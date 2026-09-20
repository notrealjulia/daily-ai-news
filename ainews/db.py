"""SQLite persistence: connection setup, schema, and every query.

This is the only module that should contain SQL. Timestamps are stored as
ISO 8601 UTC text (e.g. "2026-09-20T10:29:00Z"), which sorts correctly as a
plain string.

An article's text lives in two places on purpose:
    feed_text   what the feed itself carried (often a teaser); kept as evidence
    body        the article text later stages use, filled in by content acquisition
`body_status` says where body stands: 'pending' (not tried yet), 'ready', or
'failed' (the last attempt failed; it stays eligible for another try).
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path("ainews.db")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Added to `articles` after the first release. Defined once and used both by the
# CREATE TABLE below (fresh databases) and by _migrate (existing ones), so the two
# cannot drift apart.
BODY_COLUMNS = (
    ("body", "TEXT"),  # NULL until acquired
    ("body_source", "TEXT"),  # 'feed_content' or 'fulltext': how body was obtained
    (
        "body_status",
        "TEXT NOT NULL DEFAULT 'pending' CHECK (body_status IN ('pending', 'ready', 'failed'))",
    ),
    ("body_error", "TEXT"),  # why the latest attempt failed; NULL otherwise
    ("body_checked_at", "TEXT"),  # when we last tried
)

_BODY_COLUMNS_SQL = ",\n    ".join(f"{name} {ddl}" for name, ddl in BODY_COLUMNS)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS articles (
    id           INTEGER PRIMARY KEY,
    source       TEXT NOT NULL,           -- feed name from feeds.toml
    url          TEXT NOT NULL UNIQUE,    -- dedup key
    title        TEXT NOT NULL,
    published_at TEXT,                    -- NULL if the feed gave no usable date
    feed_text    TEXT,                    -- what the feed carried; may be a teaser or nothing
    fetched_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    {_BODY_COLUMNS_SQL}
);

CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at);

CREATE TABLE IF NOT EXISTS enrichments (
    id             INTEGER PRIMARY KEY,
    article_id     INTEGER NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    category       TEXT NOT NULL,         -- validated against the fixed list in llm.py
    relevance      INTEGER NOT NULL CHECK (relevance BETWEEN 0 AND 10),
    summary        TEXT NOT NULL,
    why_it_matters TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    -- One result per article per (model, prompt) pair; re-running with a new
    -- prompt_version adds a row instead of overwriting the old one.
    UNIQUE (article_id, model, prompt_version)
);
"""


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the database, ensuring the schema exists and is up to date.

    Rows come back as sqlite3.Row, so columns can be read by name.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # SQLite ignores foreign keys unless enabled on every connection.
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets Streamlit read while the enrich step is writing.
    conn.execute("PRAGMA journal_mode = WAL")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if missing and bring older databases up to date (idempotent)."""
    conn.executescript(SCHEMA)
    _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Upgrade an `articles` table created before content acquisition existed.

    Existing rows keep their data: `content` becomes `feed_text`, and every row starts
    with body_status 'pending', so the next `extract` run picks them up.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(articles)")}
    if "content" in columns:
        conn.execute("ALTER TABLE articles RENAME COLUMN content TO feed_text")
    for name, ddl in BODY_COLUMNS:
        if name not in columns:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {name} {ddl}")


def _format_timestamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)


# --- Ingestion ---------------------------------------------------------------


def url_exists(conn: sqlite3.Connection, url: str) -> bool:
    """Whether an article with this URL is stored (from any source)."""
    return conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone() is not None


def insert_article(
    conn: sqlite3.Connection,
    *,
    source: str,
    url: str,
    title: str,
    published_at: datetime | None,
    fetched_at: datetime,
    feed_text: str | None,
) -> bool:
    """Insert an article. Returns False if its URL is already stored.

    published_at is None for articles from feeds that give no dates.
    fetched_at is when we discovered the article.
    The article starts with body_status 'pending'.
    Does not commit; the caller decides where the transaction ends.
    """
    cursor = conn.execute(
        """
        INSERT INTO articles (source, url, title, published_at, fetched_at, feed_text)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (url) DO NOTHING
        """,
        (
            source,
            url,
            title,
            None if published_at is None else _format_timestamp(published_at),
            _format_timestamp(fetched_at),
            feed_text,
        ),
    )
    return cursor.rowcount == 1


# --- Content acquisition -----------------------------------------------------


def articles_needing_body(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every article whose body is not ready: never tried, or last attempt failed."""
    return conn.execute(
        """
        SELECT id, source, url, title, feed_text, body_status
        FROM articles
        WHERE body_status != 'ready'
        ORDER BY id
        """
    ).fetchall()


def body_status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {"pending": 0, "ready": 0, "failed": 0}
    for row in conn.execute("SELECT body_status, COUNT(*) AS n FROM articles GROUP BY body_status"):
        counts[row["body_status"]] = row["n"]
    return counts


def mark_body_ready(
    conn: sqlite3.Connection, article_id: int, *, body: str, source: str, checked_at: datetime
) -> None:
    conn.execute(
        """
        UPDATE articles
        SET body = ?, body_source = ?, body_status = 'ready', body_error = NULL,
            body_checked_at = ?
        WHERE id = ?
        """,
        (body, source, _format_timestamp(checked_at), article_id),
    )


def mark_body_failed(
    conn: sqlite3.Connection, article_id: int, *, error: str, checked_at: datetime
) -> None:
    """Record a failed attempt. The article stays eligible for the next run."""
    conn.execute(
        """
        UPDATE articles
        SET body_status = 'failed', body_error = ?, body_checked_at = ?
        WHERE id = ?
        """,
        (error, _format_timestamp(checked_at), article_id),
    )
