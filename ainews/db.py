"""SQLite persistence: connection setup and schema.

This is the only module that should contain SQL. Timestamps are stored as
ISO 8601 UTC text (e.g. "2026-09-20T10:29:00Z"), which sorts correctly as a
plain string.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path("ainews.db")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id           INTEGER PRIMARY KEY,
    source       TEXT NOT NULL,           -- feed name from feeds.toml
    url          TEXT NOT NULL UNIQUE,    -- dedup key
    title        TEXT NOT NULL,
    published_at TEXT,                    -- NULL if the feed gave no usable date
    content      TEXT,                    -- best text the feed provides; may be a teaser
    fetched_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
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
    """Open the database, ensuring the schema exists.

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
    """Create tables and indexes if they don't exist yet (idempotent)."""
    conn.executescript(SCHEMA)


def _format_timestamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)


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
    content: str | None,
) -> bool:
    """Insert an article. Returns False if its URL is already stored.

    published_at is None for articles from feeds that give no dates.
    fetched_at is when we discovered the article.
    Does not commit; the caller decides where the transaction ends.
    """
    cursor = conn.execute(
        """
        INSERT INTO articles (source, url, title, published_at, fetched_at, content)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (url) DO NOTHING
        """,
        (
            source,
            url,
            title,
            None if published_at is None else _format_timestamp(published_at),
            _format_timestamp(fetched_at),
            content,
        ),
    )
    return cursor.rowcount == 1
