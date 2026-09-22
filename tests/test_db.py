"""Tests for the schema, migration, and the article-body state helpers."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from ainews import db

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
WINDOW_START = NOW - timedelta(hours=24)  # matches ingest.MAX_ARTICLE_AGE

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
    # The old enrichments table lost its retired columns but kept its row.
    enrichment = conn.execute("SELECT * FROM enrichments").fetchone()
    assert "relevance" not in enrichment.keys() and "why_it_matters" not in enrichment.keys()
    assert (enrichment["category"], enrichment["summary"], enrichment["model"]) == ("Models", "s", "m")
    # Databases from before clustering gain the story, digest and narration tables.
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"story_runs", "stories", "story_articles", "digests", "narrations"} <= tables
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


# --- article body state ------------------------------------------------------


def test_status_must_be_one_of_the_known_values(conn):
    a = add(conn, "http://x/1")

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE articles SET body_status = 'bogus' WHERE id = ?", (a,))


def test_only_articles_without_a_ready_body_need_one(conn):
    pending, ready, failed = (add(conn, f"http://x/{n}") for n in (1, 2, 3))
    db.mark_body_ready(conn, ready, body="text", source="fulltext", checked_at=NOW)
    db.mark_body_failed(conn, failed, error="boom", checked_at=NOW)

    waiting = db.articles_needing_body(conn, WINDOW_START)

    assert [(r["id"], r["body_status"]) for r in waiting] == [(pending, "pending"), (failed, "failed")]
    assert db.body_status_counts(conn) == {"pending": 1, "ready": 1, "failed": 1}


# --- enrichments -------------------------------------------------------------


def add_ready(conn, url) -> int:
    article_id = add(conn, url)
    db.mark_body_ready(conn, article_id, body="text", source="fulltext", checked_at=NOW)
    return article_id


def enrich_row(conn, article_id, *, model="m", prompt_version="v1", category="Other"):
    return db.insert_enrichment(
        conn, article_id=article_id, category=category, summary="s", english_title=None,
        model=model, prompt_version=prompt_version, created_at=NOW,
    )  # fmt: skip


def test_only_ready_articles_without_a_result_for_the_pair_need_enriching(conn):
    done = add_ready(conn, "http://x/done")
    waiting = add_ready(conn, "http://x/waiting")
    add(conn, "http://x/pending")  # no ready body
    failed = add(conn, "http://x/failed")
    db.mark_body_failed(conn, failed, error="boom", checked_at=NOW)
    enrich_row(conn, done)
    # Results for a different model or prompt version don't make an article "done".
    enrich_row(conn, waiting, model="other-model")
    enrich_row(conn, waiting, prompt_version="v0")

    rows = db.articles_needing_enrichment(conn, "m", "v1", WINDOW_START)

    assert [r["id"] for r in rows] == [waiting]
    assert rows[0]["body"] == "text"


def test_an_article_gets_one_result_per_model_and_prompt_version(conn):
    a = add_ready(conn, "http://x/a")

    assert enrich_row(conn, a) is True
    assert enrich_row(conn, a) is False  # same pair: not stored twice
    assert enrich_row(conn, a, model="another") is True
    assert conn.execute("SELECT COUNT(*) FROM enrichments").fetchone()[0] == 2


def test_deleting_an_article_deletes_its_enrichments(conn):
    a = add_ready(conn, "http://x/a")
    enrich_row(conn, a)

    conn.execute("DELETE FROM articles WHERE id = ?", (a,))

    assert conn.execute("SELECT COUNT(*) FROM enrichments").fetchone()[0] == 0


def test_a_story_run_holds_each_article_once_and_deleting_it_removes_its_stories_digests_and_narrations(conn):
    a, b = add(conn, "http://x/a"), add(conn, "http://x/b")
    run = db.create_story_run(
        conn, model="m", prompt_version="v", enrichment_model="e", enrichment_prompt_version="v",
        window_start=NOW, window_end=NOW, input_hash="h", created_at=NOW,
    )  # fmt: skip
    db.create_story(conn, run, article_ids=[a], category="Research", summary="s", grouping_reason=None)
    db.insert_digest(
        conn, run_id=run, category="Research", story_count=1, total_story_count=1,
        headline="h", summary="d", model="m", prompt_version="v", created_at=NOW,
    )  # fmt: skip
    db.insert_narration(
        conn, run_id=run, category="Research", script="s", model="m", prompt_version="v", created_at=NOW
    )  # fmt: skip
    conn.commit()  # so the rollback() below only undoes the failed create_story() attempt, not this fixture data

    with pytest.raises(sqlite3.IntegrityError):  # article `a` is already in a story of this run
        db.create_story(conn, run, article_ids=[a, b], category=None, summary=None, grouping_reason="x")
    conn.rollback()

    conn.execute("DELETE FROM story_runs WHERE id = ?", (run,))
    assert conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM story_articles").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM narrations").fetchone()[0] == 0


# --- choosing the backend ----------------------------------------------------


def test_local_sqlite_is_used_unless_turso_is_explicitly_selected(monkeypatch, tmp_path):
    # Having the Turso credentials around (as the developer's .env does) must not switch backends.
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://example.invalid")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "not-a-real-token")
    path = tmp_path / "local.db"

    writer = db.connect(path)
    reader = db.connect_readonly(path)

    assert isinstance(writer, sqlite3.Connection) and isinstance(reader, sqlite3.Connection)
    assert path.exists()
    writer.close()
    reader.close()


@pytest.mark.parametrize("opener", [db.connect, db.connect_readonly], ids=["connect", "connect_readonly"])
@pytest.mark.parametrize(
    ("backend", "credentials", "message"),
    [
        ("turso", {}, "needs TURSO_DATABASE_URL and TURSO_AUTH_TOKEN"),
        ("turso", {"TURSO_DATABASE_URL": "libsql://example.invalid"}, "needs TURSO_AUTH_TOKEN"),
        ("trso", {}, "must be 'sqlite' or 'turso'"),  # a typo must not quietly fall back to local SQLite
    ],
    ids=["no-credentials", "no-token", "unknown-backend"],
)
def test_the_turso_backend_fails_clearly_when_misconfigured(monkeypatch, opener, backend, credentials, message):
    monkeypatch.setenv(db.BACKEND_VARIABLE, backend)
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(db.DatabaseConfigError, match=message):
        opener()  # fails before any network access


def test_a_database_from_before_digest_headlines_gains_the_column_and_keeps_its_digests(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    run = db.create_story_run(
        conn, model="m", prompt_version="v", enrichment_model="e", enrichment_prompt_version="v",
        window_start=NOW, window_end=NOW, input_hash="h", created_at=NOW,
    )  # fmt: skip
    db.insert_digest(
        conn, run_id=run, category="Research", story_count=1, total_story_count=1,
        headline="Not stored in the old format", summary="Kept.", model="m", prompt_version="v", created_at=NOW,
    )  # fmt: skip
    conn.commit()
    conn.execute("ALTER TABLE digests DROP COLUMN headline")  # what a database from before had
    conn.commit()
    conn.close()

    migrated = db.connect(path)

    assert "headline" in {c[0] for c in columns(migrated, "digests")}
    row = migrated.execute("SELECT summary, headline FROM digests").fetchone()
    assert (row["summary"], row["headline"]) == ("Kept.", None)  # old digests stay, without a headline
    migrated.close()
    db.connect(path).close()  # and migrating again changes nothing (no duplicate column)


@pytest.mark.parametrize(
    ("environment", "settings", "expected"),
    [
        ({}, {}, False),  # local SQLite is the default
        ({}, {"AINEWS_BACKEND": "turso"}, True),  # e.g. Streamlit secrets
        ({"AINEWS_BACKEND": "turso"}, {}, True),  # the real environment, as in GitHub Actions
        ({"AINEWS_BACKEND": "sqlite"}, {"AINEWS_BACKEND": "turso"}, False),  # the environment wins
        ({}, {"TURSO_DATABASE_URL": "u", "TURSO_AUTH_TOKEN": "t"}, False),  # credentials alone never switch
    ],
    ids=["default", "settings", "environment", "environment-wins", "credentials-alone"],
)
def test_the_backend_can_be_chosen_by_the_environment_or_by_settings(monkeypatch, environment, settings, expected):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert db.uses_turso(settings) is expected


def test_the_dashboard_can_take_its_turso_settings_from_secrets(monkeypatch):
    import turso_serverless

    opened = []
    monkeypatch.setattr(turso_serverless, "connect", lambda url, auth_token: opened.append((url, auth_token)) or type("C", (), {})())
    secrets = {"AINEWS_BACKEND": "turso", "TURSO_DATABASE_URL": "libsql://from-secrets", "TURSO_AUTH_TOKEN": "secret-token"}

    db.connect_readonly(settings=secrets)

    assert opened == [("libsql://from-secrets", "secret-token")]  # no local file was involved
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "from-the-environment")
    db.connect_readonly(settings=secrets)
    assert opened[-1] == ("libsql://from-secrets", "from-the-environment")  # the real environment wins
