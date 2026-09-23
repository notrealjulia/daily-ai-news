"""Tests for category digests (`python -m ainews digest`). The LLM is a fake."""

import re
from datetime import datetime, timedelta, timezone

import pytest

from ainews import db, digest, llm
from ainews.__main__ import main

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)


class FakeLLM:
    """Answers each digest call. `respond(category, input_text)` returns a dict or raises."""

    def __init__(self, respond=None, model="digest-model"):
        self.model = model
        self.calls: list[dict] = []
        self._respond = respond or (
            lambda category, input_text: {"headline": f"Headline for {category}", "summary": f"Digest of {category}."}
        )

    def generate(self, *, instructions, input_text, schema_name, schema):
        category = input_text.split("\n", 1)[0].removeprefix("Category: ")
        self.calls.append({"category": category, "input_text": input_text})
        return self._respond(category, input_text)


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def make_run(conn, *specs, input_hash="one") -> int:
    """A story run built directly. Each spec is (category, summary, article count, status)."""
    run_id = db.create_story_run(
        conn, model="story-model", prompt_version="story-prompt", enrichment_model="e",
        enrichment_prompt_version="p", window_start=NOW - timedelta(hours=24), window_end=NOW,
        input_hash=input_hash, created_at=NOW,
    )  # fmt: skip
    for n, (category, summary, article_count, status) in enumerate(specs):
        article_ids = []
        for i in range(article_count):
            url = f"http://x/{input_hash}-{n}-{i}"
            db.insert_article(
                conn, source="Src", url=url, title=f"Article {input_hash}-{n}-{i}",
                published_at=NOW, fetched_at=NOW, feed_text=None,
            )  # fmt: skip
            article_ids.append(conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()["id"])
        ready = status == "ready"
        story_id = db.create_story(
            conn, run_id, article_ids=article_ids, category=category if ready else None,
            summary=summary if ready else None, grouping_reason=None,
        )  # fmt: skip
        if status == "failed":
            db.mark_story_failed(conn, story_id, error="boom")
    conn.commit()
    return run_id


def digest_rows(conn):
    return [
        (r["category"], r["story_count"], r["total_story_count"], r["headline"], r["summary"], r["model"],
         r["prompt_version"])
        for r in conn.execute("SELECT * FROM digests ORDER BY id")
    ]


def shows(out: str, pattern: str) -> bool:
    return re.search(pattern, out) is not None


# --- what a digest is made from ----------------------------------------------


def test_one_digest_per_category_from_stories_with_the_activity_context(conn):
    make_run(
        conn,
        ("Research", "Research story one.", 2, "ready"),
        ("Research", "Research story two.", 1, "ready"),
        ("Industry News", "An industry story.", 1, "ready"),
    )
    fake = FakeLLM()

    summary = digest.run_digests(conn, fake, now=NOW)

    assert [c["category"] for c in fake.calls] == ["Research", "Industry News"]  # none for empty ones
    assert digest_rows(conn) == [
        ("Research", 2, 3, "Headline for Research", "Digest of Research.", "digest-model", digest.DIGEST_PROMPT_VERSION),
        ("Industry News", 1, 3, "Headline for Industry News", "Digest of Industry News.", "digest-model",
         digest.DIGEST_PROMPT_VERSION),
    ]
    assert set(summary.no_stories) == {"Product Release", "Business", "Regulation & Policy", "Other"}
    text = fake.calls[0]["input_text"]
    assert "Research story one." in text and "Research story two." in text
    assert "An industry story." not in text  # only this category's stories
    assert "Article one-" not in text  # stories, not articles
    # The activity context: its count, the total, its share, and same-window counts of the others.
    assert "2 of 3" in text and "67%" in text and "Industry News 1" in text


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_a_digest_refuses_to_run_from_an_incomplete_run(conn, status):
    make_run(conn, ("Research", "Fine.", 1, "ready"), ("Research", "Not done.", 2, status))
    fake = FakeLLM()

    summary = digest.run_digests(conn, fake, now=NOW)

    assert summary.incomplete_stories == 1
    assert fake.calls == [] and digest_rows(conn) == []


def test_digests_default_to_the_latest_run_and_can_target_another(conn):
    first = make_run(conn, ("Research", "Old story.", 1, "ready"), input_hash="one")
    second = make_run(conn, ("Business", "New story.", 1, "ready"), input_hash="two")

    digest.run_digests(conn, FakeLLM(), now=NOW)
    digest.run_digests(conn, FakeLLM(), run_id=first, now=NOW)

    runs = [r["run_id"] for r in conn.execute("SELECT run_id FROM digests ORDER BY id")]
    assert runs == [second, first]


# --- reruns: versioning, isolation, retry ------------------------------------


@pytest.mark.parametrize("change", ["a-different-prompt-version", "a-new-story-run"])
def test_digests_are_not_repeated_but_a_new_prompt_or_run_gets_its_own(conn, change):
    run = make_run(conn, ("Research", "A story.", 1, "ready"))
    digest.run_digests(conn, FakeLLM(), now=NOW)

    unchanged = FakeLLM()
    again = digest.run_digests(conn, unchanged, now=NOW)
    assert unchanged.calls == [] and again.already_done == ["Research"] and len(digest_rows(conn)) == 1

    if change == "a-different-prompt-version":
        digest.run_digests(conn, FakeLLM(), run_id=run, prompt_version="digest-prompt-next", now=NOW)
    else:
        make_run(conn, ("Research", "Another story.", 1, "ready"), input_hash="two")
        digest.run_digests(conn, FakeLLM(), now=NOW)
    assert len(digest_rows(conn)) == 2  # the first digest is still there


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(llm.LLMError("RateLimitError (HTTP 429): slow down"), id="llm-error"),
        pytest.param(KeyError("boom"), id="unexpected-exception"),
        pytest.param({"headline": "", "summary": "Fine."}, id="invalid-headline"),
        pytest.param({"headline": "Fine", "summary": ""}, id="invalid-summary"),
    ],
)
def test_one_failing_category_does_not_block_the_others_and_is_retried(conn, failure):
    make_run(conn, ("Research", "R.", 1, "ready"), ("Industry News", "I.", 1, "ready"))

    def respond(category, input_text):
        if category != "Research":
            return {"headline": "Fine headline", "summary": "Fine."}
        if isinstance(failure, Exception):
            raise failure
        return failure

    first = digest.run_digests(conn, FakeLLM(respond), now=NOW)

    assert [(o.category, o.ok) for o in first.outcomes] == [("Research", False), ("Industry News", True)]
    assert [row[0] for row in digest_rows(conn)] == ["Industry News"]

    retry = FakeLLM()
    digest.run_digests(conn, retry, now=NOW)

    assert [c["category"] for c in retry.calls] == ["Research"]  # only the one that failed
    assert sorted(row[0] for row in digest_rows(conn)) == ["Industry News", "Research"]


# --- the structured answer ---------------------------------------------------


def test_the_schema_asks_for_exactly_what_the_parser_accepts():
    keys = set(digest.SCHEMA["properties"])

    assert keys == set(digest.SCHEMA["required"]) == {"headline", "summary"}
    assert digest.parse_digest({key: f"A {key}." for key in keys}) == ("A headline.", "A summary.")


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"summary": "S."}, "expected exactly headline and summary"),
        ({"headline": "H", "summary": "S.", "extra": "x"}, "expected exactly headline and summary"),
        ({"headline": "  ", "summary": "S."}, "the headline is empty"),
        ({"headline": 7, "summary": "S."}, "the headline is empty"),
        ({"headline": "Two\nlines", "summary": "S."}, "not a single line"),
        ({"headline": "word " * 40, "summary": "S."}, "the headline is .* chars"),
        ({"headline": "H", "summary": ""}, "the summary is empty"),
    ],
    ids=["no-headline", "extra-key", "blank-headline", "non-string-headline", "multi-line-headline",
         "runaway-headline", "empty-summary"],
)
def test_an_invalid_headline_or_summary_is_rejected_not_repaired(answer, message):
    with pytest.raises(digest.enrich.InvalidOutput, match=message):
        digest.parse_digest(answer)


# --- the command -------------------------------------------------------------


def test_digest_command_reports_and_refuses_an_incomplete_run(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "t.db"

    def command():
        monkeypatch.setattr(llm, "create", lambda model: FakeLLM())
        return main(["digest", "--db", str(db_path)])

    conn = db.connect(db_path)
    assert command() == 1 and "No story run to work from" in capsys.readouterr().out

    make_run(conn, ("Research", "A story.", 1, "ready"), ("Industry News", "B story.", 1, "ready"))
    assert command() == 0
    out = capsys.readouterr().out
    assert shows(out, r"digests written this run:\s+2\b") and shows(out, r"categories with no stories:\s+4\b")

    make_run(conn, ("Research", "Half done.", 1, "failed"), input_hash="two")  # now the latest run
    assert command() == 1
    out = capsys.readouterr().out
    assert "Refusing to write digests: 1 of 1 stories" in out and "python -m ainews cluster" in out


def test_digest_command_still_succeeds_when_one_categorys_digest_fails(monkeypatch, tmp_path, capsys):
    # Unlike refusing to run at all (above), a single category's failed digest call is
    # retryable and must not make the command exit non-zero.
    db_path = tmp_path / "t.db"
    conn = db.connect(db_path)
    make_run(conn, ("Research", "A story.", 1, "ready"), ("Business", "B story.", 1, "ready"))
    conn.close()

    def respond(category, input_text):
        if category == "Business":
            raise llm.LLMError("RateLimitError (HTTP 429): slow down")
        return {"headline": "H", "summary": "Digest of Research."}

    fake = FakeLLM(respond)
    monkeypatch.setattr(llm, "create", lambda model: fake)

    code = main(["digest", "--db", str(db_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "FAILED" in out and "Business" in out and "slow down" in out
    assert shows(out, r"digests written this run:\s+1\b") and shows(out, r"failed this run:\s+1\b")
