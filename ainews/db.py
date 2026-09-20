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
    category       TEXT NOT NULL,         -- validated against the fixed list in enrich.py
    summary        TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    -- One result per article per (model, prompt) pair; re-running with a new
    -- prompt_version adds a row instead of overwriting the old one.
    UNIQUE (article_id, model, prompt_version)
);

-- A story run is an immutable snapshot: the stories that one clustering pass made from
-- the articles in the 24h window at that moment. It is identified by the models and
-- prompt versions involved plus a fingerprint of the input, so repeating a run with
-- unchanged input finds the existing one instead of making another.
CREATE TABLE IF NOT EXISTS story_runs (
    id                        INTEGER PRIMARY KEY,
    model                     TEXT NOT NULL,   -- LLM that grouped and wrote combined summaries
    prompt_version            TEXT NOT NULL,   -- version of the story prompts
    enrichment_model          TEXT NOT NULL,   -- which enrichments were the input
    enrichment_prompt_version TEXT NOT NULL,
    window_start              TEXT NOT NULL,
    window_end                TEXT NOT NULL,   -- when the run happened
    input_hash                TEXT NOT NULL,   -- fingerprint of the enrichment ids clustered
    created_at                TEXT NOT NULL,
    UNIQUE (model, prompt_version, enrichment_model, enrichment_prompt_version, input_hash)
);

CREATE TABLE IF NOT EXISTS stories (
    id              INTEGER PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES story_runs (id) ON DELETE CASCADE,
    category        TEXT,                      -- a non-Spam category; NULL until summary is ready
    summary         TEXT,
    summary_status  TEXT NOT NULL DEFAULT 'pending'
                    CHECK (summary_status IN ('pending', 'ready', 'failed')),
    summary_error   TEXT,                      -- why the last attempt failed; NULL otherwise
    grouping_reason TEXT                       -- why articles were grouped; NULL if just one
);

CREATE TABLE IF NOT EXISTS story_articles (
    run_id     INTEGER NOT NULL,
    story_id   INTEGER NOT NULL REFERENCES stories (id) ON DELETE CASCADE,
    article_id INTEGER NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    PRIMARY KEY (story_id, article_id),
    UNIQUE (run_id, article_id)                -- an article is in at most one story per run
);

CREATE TABLE IF NOT EXISTS digests (
    id                INTEGER PRIMARY KEY,
    run_id            INTEGER NOT NULL REFERENCES story_runs (id) ON DELETE CASCADE,
    category          TEXT NOT NULL,
    story_count       INTEGER NOT NULL,        -- stories in this category
    total_story_count INTEGER NOT NULL,        -- all (non-Spam) stories in the run
    summary           TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    UNIQUE (run_id, category, model, prompt_version)
);
"""

# Columns the first design of `enrichments` had and the current one does not.
RETIRED_ENRICHMENT_COLUMNS = ("relevance", "why_it_matters")


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
    """Upgrade tables created by earlier versions, keeping their data.

    articles:    `content` becomes `feed_text`, and every existing row starts with
                 body_status 'pending', so the next `extract` run picks it up.
    enrichments: the retired relevance and why_it_matters columns are dropped.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(articles)")}
    if "content" in columns:
        conn.execute("ALTER TABLE articles RENAME COLUMN content TO feed_text")
    for name, ddl in BODY_COLUMNS:
        if name not in columns:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {name} {ddl}")

    # Enrichment no longer scores relevance or explains why an article matters. Nothing
    # ever wrote such rows, but if any exist they keep their category and summary.
    enrichment_columns = {row["name"] for row in conn.execute("PRAGMA table_info(enrichments)")}
    for name in RETIRED_ENRICHMENT_COLUMNS:
        if name in enrichment_columns:
            conn.execute(f"ALTER TABLE enrichments DROP COLUMN {name}")


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


# --- Enrichment --------------------------------------------------------------


def articles_needing_enrichment(
    conn: sqlite3.Connection, model: str, prompt_version: str, limit: int | None = None
) -> list[sqlite3.Row]:
    """Articles with a ready body and no enrichment yet for this model + prompt_version."""
    return conn.execute(
        """
        SELECT id, source, title, url, body
        FROM articles
        WHERE body_status = 'ready'
          AND NOT EXISTS (
              SELECT 1 FROM enrichments e
              WHERE e.article_id = articles.id AND e.model = ? AND e.prompt_version = ?
          )
        ORDER BY id
        LIMIT ?
        """,
        (model, prompt_version, -1 if limit is None else limit),
    ).fetchall()


def enrichment_overview(conn: sqlite3.Connection, model: str, prompt_version: str) -> dict[str, int]:
    """Counts: ready articles, how many of those are enriched, and articles without a ready body."""
    counts = body_status_counts(conn)
    enriched = conn.execute(
        """
        SELECT COUNT(*) FROM articles
        WHERE body_status = 'ready'
          AND EXISTS (
              SELECT 1 FROM enrichments e
              WHERE e.article_id = articles.id AND e.model = ? AND e.prompt_version = ?
          )
        """,
        (model, prompt_version),
    ).fetchone()[0]
    return {
        "ready": counts["ready"],
        "enriched": enriched,
        "not_ready": counts["pending"] + counts["failed"],
    }


def insert_enrichment(
    conn: sqlite3.Connection,
    *,
    article_id: int,
    category: str,
    summary: str,
    model: str,
    prompt_version: str,
    created_at: datetime,
) -> bool:
    """Store an enrichment. Returns False if this article already has one for the pair."""
    cursor = conn.execute(
        """
        INSERT INTO enrichments (article_id, category, summary, model, prompt_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (article_id, model, prompt_version) DO NOTHING
        """,
        (article_id, category, summary, model, prompt_version, _format_timestamp(created_at)),
    )
    return cursor.rowcount == 1


# --- Story clustering --------------------------------------------------------


def story_window_articles(
    conn: sqlite3.Connection,
    *,
    window_start: datetime,
    enrichment_model: str,
    enrichment_prompt_version: str,
) -> list[sqlite3.Row]:
    """Every article in the window, with its enrichment for this version if it has one.

    An article's time is its published_at, or fetched_at when the feed gave no date.
    enrichment_id, category and summary are NULL for articles not enriched yet.
    """
    return conn.execute(
        """
        SELECT a.id AS article_id, a.source, a.title,
               e.id AS enrichment_id, e.category, e.summary
        FROM articles a
        LEFT JOIN enrichments e
               ON e.article_id = a.id AND e.model = ? AND e.prompt_version = ?
        WHERE COALESCE(a.published_at, a.fetched_at) >= ?
        ORDER BY a.id
        """,
        (enrichment_model, enrichment_prompt_version, _format_timestamp(window_start)),
    ).fetchall()


def find_story_run(
    conn: sqlite3.Connection,
    *,
    model: str,
    prompt_version: str,
    enrichment_model: str,
    enrichment_prompt_version: str,
    input_hash: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM story_runs
        WHERE model = ? AND prompt_version = ? AND enrichment_model = ?
          AND enrichment_prompt_version = ? AND input_hash = ?
        """,
        (model, prompt_version, enrichment_model, enrichment_prompt_version, input_hash),
    ).fetchone()


def create_story_run(
    conn: sqlite3.Connection,
    *,
    model: str,
    prompt_version: str,
    enrichment_model: str,
    enrichment_prompt_version: str,
    window_start: datetime,
    window_end: datetime,
    input_hash: str,
    created_at: datetime,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO story_runs (model, prompt_version, enrichment_model,
                                enrichment_prompt_version, window_start, window_end,
                                input_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model, prompt_version, enrichment_model, enrichment_prompt_version,
            _format_timestamp(window_start), _format_timestamp(window_end),
            input_hash, _format_timestamp(created_at),
        ),
    )  # fmt: skip
    return cursor.lastrowid


def create_story(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    article_ids: list[int],
    category: str | None,
    summary: str | None,
    grouping_reason: str | None,
) -> int:
    """Add a story and link its articles. It is 'ready' if it already has a summary
    (a single-article story copies its article's), otherwise 'pending'."""
    cursor = conn.execute(
        """
        INSERT INTO stories (run_id, category, summary, summary_status, grouping_reason)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, category, summary, "ready" if summary is not None else "pending", grouping_reason),
    )
    story_id = cursor.lastrowid
    for article_id in article_ids:
        conn.execute(
            "INSERT INTO story_articles (run_id, story_id, article_id) VALUES (?, ?, ?)",
            (run_id, story_id, article_id),
        )
    return story_id


def stories_needing_summary(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Stories of a run whose combined summary is not ready: never tried, or failed."""
    return conn.execute(
        "SELECT id FROM stories WHERE run_id = ? AND summary_status != 'ready' ORDER BY id",
        (run_id,),
    ).fetchall()


def story_member_articles(
    conn: sqlite3.Connection, story_id: int, *, enrichment_model: str, enrichment_prompt_version: str
) -> list[sqlite3.Row]:
    """A story's articles with the enrichment summaries the run was built from."""
    return conn.execute(
        """
        SELECT a.id AS article_id, a.source, a.title, e.summary
        FROM story_articles sa
        JOIN articles a ON a.id = sa.article_id
        JOIN enrichments e
             ON e.article_id = a.id AND e.model = ? AND e.prompt_version = ?
        WHERE sa.story_id = ?
        ORDER BY a.id
        """,
        (enrichment_model, enrichment_prompt_version, story_id),
    ).fetchall()


def mark_story_ready(conn: sqlite3.Connection, story_id: int, *, category: str, summary: str) -> None:
    conn.execute(
        """
        UPDATE stories
        SET category = ?, summary = ?, summary_status = 'ready', summary_error = NULL
        WHERE id = ?
        """,
        (category, summary, story_id),
    )


def mark_story_failed(conn: sqlite3.Connection, story_id: int, *, error: str) -> None:
    """Record a failed attempt. The story stays eligible for the next run."""
    conn.execute(
        "UPDATE stories SET summary_status = 'failed', summary_error = ? WHERE id = ?",
        (error, story_id),
    )


def get_story_run(conn: sqlite3.Connection, run_id: int | None = None) -> sqlite3.Row | None:
    """A story run by id, or the most recent one when no id is given."""
    if run_id is None:
        return conn.execute("SELECT * FROM story_runs ORDER BY id DESC LIMIT 1").fetchone()
    return conn.execute("SELECT * FROM story_runs WHERE id = ?", (run_id,)).fetchone()


def stories_in_run(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT s.id, s.category, s.summary, s.summary_status, s.summary_error, s.grouping_reason,
               (SELECT COUNT(*) FROM story_articles sa WHERE sa.story_id = s.id) AS article_count
        FROM stories s
        WHERE s.run_id = ?
        ORDER BY s.id
        """,
        (run_id,),
    ).fetchall()


def story_article_titles(conn: sqlite3.Connection, run_id: int) -> dict[int, list[tuple[str, str]]]:
    """story id -> [(source, title), ...] for every story in a run."""
    titles: dict[int, list[tuple[str, str]]] = {}
    for row in conn.execute(
        """
        SELECT sa.story_id, a.source, a.title
        FROM story_articles sa JOIN articles a ON a.id = sa.article_id
        WHERE sa.run_id = ?
        ORDER BY sa.story_id, a.id
        """,
        (run_id,),
    ):
        titles.setdefault(row["story_id"], []).append((row["source"], row["title"]))
    return titles


# --- Digests -----------------------------------------------------------------


def digest_exists(
    conn: sqlite3.Connection, *, run_id: int, category: str, model: str, prompt_version: str
) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM digests WHERE run_id = ? AND category = ? AND model = ? AND prompt_version = ?",
            (run_id, category, model, prompt_version),
        ).fetchone()
        is not None
    )


def insert_digest(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    category: str,
    story_count: int,
    total_story_count: int,
    summary: str,
    model: str,
    prompt_version: str,
    created_at: datetime,
) -> bool:
    """Store a digest. Returns False if one already exists for this run, category, model and prompt."""
    cursor = conn.execute(
        """
        INSERT INTO digests (run_id, category, story_count, total_story_count, summary,
                             model, prompt_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, category, model, prompt_version) DO NOTHING
        """,
        (run_id, category, story_count, total_story_count, summary, model, prompt_version,
         _format_timestamp(created_at)),
    )  # fmt: skip
    return cursor.rowcount == 1


# --- Read-only access for the dashboard --------------------------------------


def connect_readonly(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open an existing database strictly read-only, for the dashboard.

    Unlike connect(), this never creates the file, sets pragmas, creates tables or
    migrates anything: SQLite itself refuses every write on this connection. Raises
    sqlite3.OperationalError if the file doesn't exist.

    (Opening a WAL-mode database this way may leave empty -wal/-shm sidecar files
    next to it. They hold no data, and the database file itself is not touched.)
    """
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def latest_dashboard_run(
    conn: sqlite3.Connection, *, digest_model: str, digest_prompt_version: str
) -> sqlite3.Row | None:
    """The newest story run that is fully processed, or None.

    Fully processed means every story has a ready summary AND every category that has
    stories has a digest written with this model and prompt version. A run that is
    only partly done is never returned, however new it is.
    """
    return conn.execute(
        """
        SELECT r.* FROM story_runs r
        WHERE EXISTS (SELECT 1 FROM stories s WHERE s.run_id = r.id)
          AND NOT EXISTS (
              SELECT 1 FROM stories s
              WHERE s.run_id = r.id AND s.summary_status != 'ready'
          )
          AND NOT EXISTS (
              SELECT 1 FROM stories s
              WHERE s.run_id = r.id
                AND NOT EXISTS (
                    SELECT 1 FROM digests d
                    WHERE d.run_id = r.id AND d.category = s.category
                      AND d.model = ? AND d.prompt_version = ?
                )
          )
        ORDER BY r.id DESC
        LIMIT 1
        """,
        (digest_model, digest_prompt_version),
    ).fetchone()


def story_article_links(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Every article of every story in a run, with its URL and time (published_at, or
    fetched_at for dateless articles), earliest first within each story."""
    return conn.execute(
        """
        SELECT sa.story_id, a.id AS article_id, a.source, a.title, a.url,
               COALESCE(a.published_at, a.fetched_at) AS happened_at
        FROM story_articles sa
        JOIN articles a ON a.id = sa.article_id
        WHERE sa.run_id = ?
        ORDER BY sa.story_id, happened_at, a.id
        """,
        (run_id,),
    ).fetchall()


def digests_for_run(
    conn: sqlite3.Connection, run_id: int, *, model: str, prompt_version: str
) -> list[sqlite3.Row]:
    """A run's digests for exactly this model and prompt version (at most one per category)."""
    return conn.execute(
        """
        SELECT category, summary, created_at FROM digests
        WHERE run_id = ? AND model = ? AND prompt_version = ?
        ORDER BY id
        """,
        (run_id, model, prompt_version),
    ).fetchall()
