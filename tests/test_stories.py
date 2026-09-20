"""Tests for story clustering (`python -m ainews cluster`). The LLM is a fake."""

import re
from datetime import datetime, timedelta, timezone

import pytest

from ainews import db, enrich, ingest, llm, stories
from ainews.__main__ import main

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
ENRICH_MODEL, ENRICH_PROMPT = "enrich-model", "enrich-prompt"
COMBINED = {"category": "Regulation & Policy", "summary": "One combined summary."}


class FakeLLM:
    """Answers the grouping call with `groups` and every combined-summary call with `combine`.

    Either may be a value, a callable taking the input text (which may raise), or an
    exception to raise.
    """

    def __init__(self, groups=(), combine=None, model="story-model"):
        self.model = model
        self.calls: list[dict] = []
        self._groups = list(groups) if isinstance(groups, (list, tuple)) else groups
        self._combine = combine if combine is not None else dict(COMBINED)

    def generate(self, *, instructions, input_text, schema_name, schema):
        self.calls.append({"input_text": input_text, "schema_name": schema_name})
        if schema_name == stories.GROUPING_SCHEMA_NAME:
            if isinstance(self._groups, Exception):
                raise self._groups
            return {"groups": self._groups}
        if isinstance(self._combine, Exception):
            raise self._combine
        return self._combine(input_text) if callable(self._combine) else self._combine


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def add(
    conn, title, *, category="Research", summary=None, source="Src",
    published=NOW - timedelta(hours=1), fetched=NOW - timedelta(hours=2),
    enriched=True, model=ENRICH_MODEL, prompt=ENRICH_PROMPT,
) -> int:  # fmt: skip
    """An article, enriched by default. published=None makes it dateless (fetched_at counts)."""
    url = f"http://x/{title}"
    db.insert_article(
        conn, source=source, url=url, title=title,
        published_at=published, fetched_at=fetched, feed_text=None,
    )  # fmt: skip
    article_id = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()["id"]
    if enriched:
        db.insert_enrichment(
            conn, article_id=article_id, category=category,
            summary=summary or f"Summary of {title}.", model=model, prompt_version=prompt, created_at=NOW,
        )  # fmt: skip
    conn.commit()
    return article_id


def cluster(conn, fake, **kwargs):
    kwargs.setdefault("enrichment_model", ENRICH_MODEL)
    kwargs.setdefault("enrichment_prompt_version", ENRICH_PROMPT)
    return stories.run_clustering(conn, fake, now=NOW, **kwargs)


def run_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM story_runs").fetchone()[0]


def story_rows(conn, run_id):
    return [
        (r["category"], r["summary"], r["summary_status"], r["article_count"])
        for r in db.stories_in_run(conn, run_id)
    ]


def calls_of(fake, schema_name):
    return [c for c in fake.calls if c["schema_name"] == schema_name]


def shows(out: str, pattern: str) -> bool:
    return re.search(pattern, out) is not None


# --- what gets clustered -----------------------------------------------------


def test_only_recent_enriched_non_spam_articles_are_clustered(conn):
    add(conn, "At window start", published=NOW - ingest.MAX_ARTICLE_AGE)  # the boundary is in
    add(conn, "Recent")
    add(conn, "Dateless", published=None, fetched=NOW - timedelta(hours=3))  # fetched_at counts
    add(conn, "Too old", published=NOW - ingest.MAX_ARTICLE_AGE - timedelta(seconds=1))
    add(conn, "Spammy", category="Spam")
    add(conn, "Not enriched yet", enriched=False)
    add(conn, "Enriched under another version", prompt="another-prompt")
    fake = FakeLLM()  # groups nothing

    summary = cluster(conn, fake)

    (grouping,) = fake.calls  # one call, and no summary calls: nothing was grouped
    for title in ("At window start", "Recent", "Dateless"):
        assert f"Title: {title}\n" in grouping["input_text"]
    for title in ("Too old", "Spammy", "Not enriched yet", "Enriched under another version"):
        assert f"Title: {title}\n" not in grouping["input_text"]
    assert (summary.in_window, summary.candidates, summary.spam_excluded, summary.not_enriched) == (
        6, 3, 1, 2,
    )  # fmt: skip
    # Every article is its own story, copying its article's category and summary.
    assert story_rows(conn, summary.run_id) == [
        ("Research", "Summary of At window start.", "ready", 1),
        ("Research", "Summary of Recent.", "ready", 1),
        ("Research", "Summary of Dateless.", "ready", 1),
    ]


# --- grouping ----------------------------------------------------------------


def test_articles_grouped_by_the_model_become_one_story_with_a_combined_summary(conn):
    # Same event, different enrichment categories: category equality is not required.
    decoder = add(conn, "Trump announces AI Force", category="Regulation & Policy",
                  source="The Decoder", summary="Trump announced an AI Force and an AI czar.")  # fmt: skip
    techcrunch = add(conn, "Trump says rebrand AI", category="Industry News",
                     source="TechCrunch AI", summary="Trump proposed renaming AI and announced an AI Force.")  # fmt: skip
    add(conn, "Anthropic postpones IPO", category="Industry News", summary="Anthropic delays its IPO.")
    reason = "Both report the AI Force announcement."
    fake = FakeLLM(groups=[{"article_ids": [decoder, techcrunch], "reason": reason}])

    summary = cluster(conn, fake)

    combine = calls_of(fake, stories.COMBINE_SCHEMA_NAME)
    assert len(calls_of(fake, stories.GROUPING_SCHEMA_NAME)) == 1 and len(combine) == 1
    merged, single = db.stories_in_run(conn, summary.run_id)
    assert (merged["article_count"], merged["summary_status"]) == (2, "ready")
    assert (merged["category"], merged["summary"]) == ("Regulation & Policy", "One combined summary.")
    assert merged["grouping_reason"] == reason
    assert (single["category"], single["summary"]) == ("Industry News", "Anthropic delays its IPO.")
    # The relationship to the source articles is preserved.
    titles = db.story_article_titles(conn, summary.run_id)
    assert [t for _, t in titles[merged["id"]]] == ["Trump announces AI Force", "Trump says rebrand AI"]
    # The combined summary is written from its own articles only.
    assert "Trump announced an AI Force" in combine[0]["input_text"]
    assert "Trump proposed renaming AI" in combine[0]["input_text"]
    assert "Anthropic" not in combine[0]["input_text"]


@pytest.mark.parametrize(
    "bad_grouping",
    [
        pytest.param(llm.LLMError("APIConnectionError: Connection error."), id="llm-error"),
        pytest.param(lambda a, b, c: [{"article_ids": [a, 999], "reason": "x"}], id="unknown-article"),
        pytest.param(
            lambda a, b, c: [{"article_ids": [a, b], "reason": "x"}, {"article_ids": [b, c], "reason": "y"}],
            id="article-in-two-groups",
        ),
        pytest.param(lambda a, b, c: [{"article_ids": [a], "reason": "x"}], id="group-of-one"),
        pytest.param(lambda a, b, c: [{"article_ids": [a, b], "reason": "  "}], id="empty-reason"),
    ],
)
def test_a_failed_or_invalid_grouping_stores_nothing_and_a_later_run_recovers(conn, bad_grouping):
    a, b, c = (add(conn, title) for title in ("One", "Two", "Three"))
    groups = bad_grouping if isinstance(bad_grouping, Exception) else bad_grouping(a, b, c)

    first = cluster(conn, FakeLLM(groups=groups))

    assert first.grouping_error and first.run_id is None
    assert run_count(conn) == 0  # nothing is stored, so nothing is left half-done

    second = cluster(conn, FakeLLM())
    assert second.grouping_error is None and second.new_run and run_count(conn) == 1


# --- combined summaries: isolation and retry ---------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(llm.LLMError("RateLimitError (HTTP 429): slow down"), id="llm-error"),
        pytest.param(KeyError("boom"), id="unexpected-exception"),
        pytest.param({"category": "Spam", "summary": "Stories can't be Spam."}, id="invalid-answer"),
    ],
)
def test_one_failing_story_summary_does_not_block_the_others_and_is_retried_without_regrouping(
    conn, failure
):
    a1, a2, b1, b2 = (add(conn, title, summary=f"Summary {title}.") for title in ("A1", "A2", "B1", "B2"))
    groups = [
        {"article_ids": [a1, a2], "reason": "same event A"},
        {"article_ids": [b1, b2], "reason": "same event B"},
    ]

    def combine(input_text):
        if "Summary A1." not in input_text:
            return dict(COMBINED)
        if isinstance(failure, Exception):
            raise failure
        return failure

    first = cluster(conn, FakeLLM(groups=groups, combine=combine))

    assert [o.ok for o in first.outcomes] == [False, True]
    failed, ready = db.stories_in_run(conn, first.run_id)
    assert (failed["summary_status"], failed["category"]) == ("failed", None) and failed["summary_error"]
    assert ready["summary_status"] == "ready"

    retry = FakeLLM()
    second = cluster(conn, retry)

    assert second.new_run is False and second.grouping_error is None
    assert [c["schema_name"] for c in retry.calls] == [stories.COMBINE_SCHEMA_NAME]  # only the failed story
    assert {r["summary_status"] for r in db.stories_in_run(conn, second.run_id)} == {"ready"}


# --- runs are immutable snapshots --------------------------------------------


def test_rerunning_with_unchanged_input_does_nothing(conn):
    add(conn, "One")
    add(conn, "Two")
    first = cluster(conn, FakeLLM())

    fake = FakeLLM()
    second = cluster(conn, fake)

    assert fake.calls == [] and second.new_run is False
    assert second.run_id == first.run_id and run_count(conn) == 1


@pytest.mark.parametrize("change", ["a-new-article", "a-different-prompt-version"])
def test_changed_input_or_prompt_makes_a_new_run_and_keeps_the_old_one(conn, change):
    add(conn, "One")
    add(conn, "Two")
    first = cluster(conn, FakeLLM())
    before = story_rows(conn, first.run_id)
    kwargs = {}
    if change == "a-new-article":
        add(conn, "Three")
    else:
        kwargs["prompt_version"] = "story-prompt-next"

    second = cluster(conn, FakeLLM(), **kwargs)

    assert second.new_run and second.run_id != first.run_id and run_count(conn) == 2
    assert story_rows(conn, first.run_id) == before  # the old snapshot is untouched


# --- the command -------------------------------------------------------------


def test_cluster_command_reports_the_run_and_retries_failed_summaries(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "t.db"
    conn = db.connect(db_path)
    recent = datetime.now(UTC) - timedelta(hours=1)  # the command uses the real clock
    current = {"published": recent, "model": llm.DEFAULT_MODEL, "prompt": enrich.PROMPT_VERSION}
    a = add(conn, "Trump announces AI Force", **current)
    b = add(conn, "Trump says rebrand AI", **current)
    add(conn, "Unrelated", **current)
    add(conn, "Spammy", category="Spam", **current)
    conn.close()
    groups = [{"article_ids": [a, b], "reason": "Both report the AI Force announcement."}]

    def command(fake):
        monkeypatch.setattr(llm, "create", lambda model: fake)
        return main(["cluster", "--db", str(db_path)])

    code = command(FakeLLM(groups=groups, combine=llm.LLMError("RateLimitError (HTTP 429): slow down")))
    out = capsys.readouterr().out
    assert code == 1  # a summary failed
    assert "Grouped because: Both report the AI Force announcement." in out
    assert shows(out, r"excluded as Spam:\s+1\b")
    assert shows(out, r"clustered:\s+3 articles into 2 stories \(1 with several articles\)")
    assert "Retryable failures" in out and "1 x RateLimitError (HTTP 429): slow down" in out

    code = command(FakeLLM(groups=groups))
    out = capsys.readouterr().out
    assert code == 0
    assert "already existed, so not regrouped" in out
    assert shows(out, r"combined summaries:\s+ok 1, failed 0")
    assert "Summary: One combined summary." in out


@pytest.mark.parametrize("command", ["cluster", "digest"])
def test_llm_commands_stop_cleanly_without_an_api_key(monkeypatch, tmp_path, capsys, command):
    monkeypatch.delenv("OPENAI_API_KEY")
    db_path = tmp_path / "t.db"

    code = main([command, "--db", str(db_path)])

    assert code == 2 and "OPENAI_API_KEY is not set" in capsys.readouterr().out
    assert not db_path.exists()  # stopped before touching anything
