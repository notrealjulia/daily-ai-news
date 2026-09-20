"""Tests for the schema, migration, and the article-body state helpers."""

import sqlite3
from datetime import datetime, timezone

import pytest

from ainews import db

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)

# The schema as first released, before content acquisition existed. Databases created
# then must be upgraded in place, keeping their data.
V1_SCHEMA = """
CREATE TABLE articles (
    id           INTEGER PRIMARY KEY,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    published_at TEXT,
    content      TEXT,
    fetched_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX idx_articles_published_at ON articles (published_at);
CREATE TABLE enrichments (
    id             INTEGER PRIMARY KEY,
    article_id     INTEGER NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    category       TEXT NOT NULL,
    relevance      INTEGER NOT NULL CHECK (relevance BETWEEN 0 AND 10),
    summary        TEXT NOT NULL,
    why_it_matters TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (article_id, model, prompt_version)
);
"""


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def columns(connection, table):
    """(name, type, not null, default) for every column, in order."""
    return [
        (r["name"], r["type"], r["notnull"], r["dflt_value"])
        for r in connection.execute(f"PRAGMA table_info({table})")
    ]


def add(conn, url, *, title="T", feed_text="teaser") -> int:
    db.insert_article(
        conn, source="S", url=url, title=title,
        published_at=NOW, fetched_at=NOW, feed_text=feed_text,
    )  # fmt: skip
    return conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()["id"]


def make_v1_database(path):
    old = sqlite3.connect(path)
    old.executescript(V1_SCHEMA)
    old.execute(
        "INSERT INTO articles (source, url, title, published_at, content, fetched_at) "
        "VALUES ('Simon', 'http://x/1', 'First', '2026-09-19T10:00:00Z', 'feed text one', '2026-09-19T11:00:00Z')"
    )
    old.execute(
        "INSERT INTO articles (source, url, title, content) VALUES ('TC', 'http://x/2', 'Second', NULL)"
    )
    old.execute(
        "INSERT INTO enrichments (article_id, category, relevance, summary, why_it_matters, model, prompt_version) "
        "VALUES (1, 'Models', 7, 's', 'w', 'm', 'v1')"
    )
    old.commit()
    old.close()


# --- migration ---------------------------------------------------------------


def test_an_old_database_is_upgraded_in_place_and_keeps_its_data(tmp_path):
    path = tmp_path / "old.db"
    make_v1_database(path)

    conn = db.connect(path)

    rows = conn.execute("SELECT * FROM articles ORDER BY id").fetchall()
    assert [r["url"] for r in rows] == ["http://x/1", "http://x/2"]
    assert rows[0]["feed_text"] == "feed text one"  # `content` was renamed, not lost
    assert rows[0]["published_at"] == "2026-09-19T10:00:00Z"
    assert rows[0]["fetched_at"] == "2026-09-19T11:00:00Z"
    assert rows[1]["feed_text"] is None
    assert all(r["body_status"] == "pending" and r["body"] is None for r in rows)
    assert "content" not in rows[0].keys()
    assert conn.execute("SELECT COUNT(*) FROM enrichments").fetchone()[0] == 1
    conn.close()


def test_a_migrated_database_has_exactly_the_schema_of_a_fresh_one(tmp_path, conn):
    path = tmp_path / "old.db"
    make_v1_database(path)

    migrated = db.connect(path)

    assert columns(migrated, "articles") == columns(conn, "articles")
    assert columns(migrated, "enrichments") == columns(conn, "enrichments")
    migrated.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "old.db"
    make_v1_database(path)

    first = db.connect(path)
    first.close()
    second = db.connect(path)  # must not fail trying to add columns again

    assert second.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 2
    second.close()


def test_migrated_rows_are_picked_up_by_the_next_extraction(tmp_path):
    path = tmp_path / "old.db"
    make_v1_database(path)

    conn = db.connect(path)

    assert [r["url"] for r in db.articles_needing_body(conn)] == ["http://x/1", "http://x/2"]
    conn.close()


# --- article body state ------------------------------------------------------


def test_new_articles_start_pending_with_no_body(conn):
    a = add(conn, "http://x/1")

    row = conn.execute("SELECT * FROM articles WHERE id = ?", (a,)).fetchone()
    assert (row["body_status"], row["body"], row["body_source"], row["body_error"]) == (
        "pending", None, None, None,
    )  # fmt: skip
    assert row["feed_text"] == "teaser"


def test_status_must_be_one_of_the_known_values(conn):
    a = add(conn, "http://x/1")

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE articles SET body_status = 'bogus' WHERE id = ?", (a,))


def test_only_articles_without_a_ready_body_need_one(conn):
    pending, ready, failed = (add(conn, f"http://x/{n}") for n in (1, 2, 3))
    db.mark_body_ready(conn, ready, body="text", source="fulltext", checked_at=NOW)
    db.mark_body_failed(conn, failed, error="boom", checked_at=NOW)

    waiting = db.articles_needing_body(conn)

    assert [(r["id"], r["body_status"]) for r in waiting] == [(pending, "pending"), (failed, "failed")]
    assert db.body_status_counts(conn) == {"pending": 1, "ready": 1, "failed": 1}


def test_marking_ready_stores_the_body_and_clears_an_earlier_error(conn):
    a = add(conn, "http://x/1")
    db.mark_body_failed(conn, a, error="HTTP 403", checked_at=NOW)

    db.mark_body_ready(conn, a, body="the text", source="fulltext", checked_at=NOW)

    row = conn.execute("SELECT * FROM articles WHERE id = ?", (a,)).fetchone()
    assert (row["body"], row["body_source"], row["body_status"], row["body_error"]) == (
        "the text", "fulltext", "ready", None,
    )  # fmt: skip
    assert row["body_checked_at"] == "2026-09-20T12:00:00Z"


def test_marking_failed_records_the_reason_and_leaves_no_body(conn):
    a = add(conn, "http://x/1", feed_text="a teaser")

    db.mark_body_failed(conn, a, error="HTTP 403 Forbidden", checked_at=NOW)

    row = conn.execute("SELECT * FROM articles WHERE id = ?", (a,)).fetchone()
    assert (row["body_status"], row["body_error"], row["body"]) == ("failed", "HTTP 403 Forbidden", None)
    assert row["feed_text"] == "a teaser"  # the teaser stays as evidence, never becomes the body
