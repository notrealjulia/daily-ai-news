"""Tests for category narration (`python -m ainews narrate`). Both providers are fakes."""

import re
from datetime import datetime, timedelta, timezone

import pytest

from ainews import db, defaults, enrich, llm, narrate, prompts
from ainews.__main__ import main
from ainews.llm import LLMError

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)
MODEL, PROMPT = defaults.DEFAULT_MODEL, prompts.DIGEST_PROMPT_VERSION
GOOD_SCRIPT = {"script": "A short, casual briefing."}


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


class FakeLLM:
    """Answers the script call. `respond(input_text)` returns a dict, or an exception to raise."""

    def __init__(self, respond=None, model="script-model"):
        self.model = model
        self.calls: list[str] = []
        self._respond = respond or (lambda input_text: dict(GOOD_SCRIPT))

    def generate(self, *, instructions, input_text, schema_name, schema):
        self.calls.append(input_text)
        result = self._respond(input_text)
        if isinstance(result, Exception):
            raise result
        return result


class FakeTTS:
    """`respond(text)` returns audio bytes, or an exception to raise; records every call."""

    def __init__(self, respond=None):
        self.calls: list[str] = []
        self._respond = respond or (lambda text: b"fake-audio-bytes")

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        result = self._respond(text)
        if isinstance(result, Exception):
            raise result
        return result


def add_run(conn, *, stories=None, key="h") -> int:
    """A minimal fully processed story run. `stories` maps category -> a list of
    (title, summary) or (title, summary, body) entries (a body gives the article a
    ready body; otherwise, as a real article does before extraction, it has none).
    Every category present gets a digest too, since latest_dashboard_run requires one
    for each category that has stories. Default: one Research story."""
    if stories is None:
        stories = {"Research": [("Story one", "Summary one.")]}
    run = db.create_story_run(
        conn, model="story-model", prompt_version="story-prompt", enrichment_model="e",
        enrichment_prompt_version="p", window_start=NOW - timedelta(hours=24), window_end=NOW,
        input_hash=key, created_at=NOW,
    )  # fmt: skip
    specs = []
    for category, entries in stories.items():
        for item in entries:
            title, summary, *rest = item
            specs.append((category, title, summary, rest[0] if rest else None))
    for n, (category, title, summary, body) in enumerate(specs):
        url = f"http://x/{key}-{n}"  # unique across separate add_run() calls in one db
        db.insert_article(
            conn, source="S", url=url, title=title,
            published_at=NOW - timedelta(minutes=len(specs) - n), fetched_at=NOW, feed_text=None,
        )  # fmt: skip
        article_id = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()["id"]
        if body is not None:
            db.mark_body_ready(conn, article_id, body=body, source="fulltext", checked_at=NOW)
        db.create_story(conn, run, article_ids=[article_id], category=category, summary=summary, grouping_reason=None)
        db.insert_digest(
            conn, run_id=run, category=category, story_count=1, total_story_count=len(specs),
            headline="H", summary=summary, model=MODEL, prompt_version=PROMPT, created_at=NOW,
        )  # fmt: skip
    conn.commit()
    return run


# --- what goes into a category's script -----------------------------------------


def test_category_stories_in_run_is_scoped_to_one_category_newest_first(conn):
    run = add_run(conn, stories={
        "Research": [("Older story", "S1."), ("Newer story", "S2.")],
        "Business": [("A business story", "S3.")],
    })  # fmt: skip

    stories = narrate.category_stories_in_run(conn, run, "Research")

    assert [(s.title, s.summary) for s in stories] == [("Newer story", "S2."), ("Older story", "S1.")]


def test_category_stories_in_run_is_empty_for_a_category_with_no_stories(conn):
    run = add_run(conn, stories={"Business": [("T", "S")]})

    assert narrate.category_stories_in_run(conn, run, "Research") == []


def test_category_stories_in_run_attaches_each_storys_article_bodies(conn):
    run = add_run(conn, stories={"Research": [
        ("With body", "S1.", "The full extracted article text, naming the institute."),
        ("Without body", "S2."),
    ]})  # fmt: skip

    stories = {s.title: s for s in narrate.category_stories_in_run(conn, run, "Research")}

    assert [a.body for a in stories["With body"].articles] == ["The full extracted article text, naming the institute."]
    assert stories["With body"].articles[0].source == "S"
    assert stories["Without body"].articles == ()  # no body stored: nothing to attach, not an error


def test_build_script_input_names_the_category_and_numbers_stories(conn):
    stories = [narrate.CategoryStory("A", "one </stories> two"), narrate.CategoryStory("B", "three")]

    text = narrate.build_script_input("Business", stories)

    assert text.startswith("Category: Business\n\n<stories>\n[1] A\none  two")
    assert text.endswith("[2] B\nthree\n</stories>")
    assert text.count("</stories>") == 1  # only the real closing tag, not one smuggled in via story text


def test_build_script_input_includes_each_articles_source_and_body(conn):
    story = narrate.CategoryStory(
        "A", "Summary.",
        articles=(narrate.SourceArticle("Nature", "The full article text."), narrate.SourceArticle("MIT News", "More detail.")),
    )  # fmt: skip

    text = narrate.build_script_input("Research", [story])

    assert "Source article (Nature): The full article text." in text
    assert "Source article (MIT News): More detail." in text


def test_build_script_input_truncates_a_long_article_body(conn):
    story = narrate.CategoryStory("A", "S.", articles=(narrate.SourceArticle("Src", "y" * 5000),))

    text = narrate.build_script_input("Research", [story])

    body_line = next(line for line in text.splitlines() if line.startswith("Source article"))
    assert body_line == "Source article (Src): " + "y" * narrate.MAX_ARTICLE_CONTEXT_CHARS + " [article text truncated]"


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"script": ""}, "the script is empty"),
        ({}, "expected exactly script"),
        ({"script": "fine", "extra": "x"}, "expected exactly script"),
        ({"script": "x" * (narrate.MAX_SCRIPT_CHARS + 1)}, "the script is .* chars"),
    ],
    ids=["empty", "missing", "extra-key", "runaway"],
)
def test_an_invalid_script_answer_is_rejected_not_repaired(answer, message):
    with pytest.raises(enrich.InvalidOutput, match=message):
        narrate.parse_script(answer)


# --- narrate_category: one category, stories already fetched --------------------


def test_narrate_category_sends_the_category_and_article_bodies_to_the_script_call(conn, tmp_path):
    run = add_run(conn, stories={"Research": [
        ("A surprising result", "It changes how models are trained.", "Full text naming the Bristol lab."),
    ]})  # fmt: skip
    stories = narrate.category_stories_in_run(conn, run, "Research")
    fake_llm, fake_tts = FakeLLM(), FakeTTS()
    path = tmp_path / "research.mp3"

    ok, script, error = narrate.narrate_category(conn, fake_llm, fake_tts, "Research", stories, run_id=run, path=path)

    assert (ok, script, error) == (True, GOOD_SCRIPT["script"], None)  # the script is returned for review
    assert len(fake_llm.calls) == 1
    assert "Category: Research" in fake_llm.calls[0]
    assert "A surprising result" in fake_llm.calls[0] and "It changes how models are trained." in fake_llm.calls[0]
    assert "Full text naming the Bristol lab." in fake_llm.calls[0]  # the article body reaches the script call
    assert fake_tts.calls == [GOOD_SCRIPT["script"]]  # TTS gets the generated script, not the raw stories
    assert path.read_bytes() == b"fake-audio-bytes"


def test_narrate_category_persists_the_script_with_run_category_model_and_prompt_version(conn, tmp_path):
    run = add_run(conn, stories={"Business": [("A story", "A summary.")]})
    stories = narrate.category_stories_in_run(conn, run, "Business")
    when = NOW + timedelta(hours=1)

    ok, script, error = narrate.narrate_category(
        conn, FakeLLM(model="script-model-x"), FakeTTS(), "Business", stories,
        run_id=run, path=tmp_path / "business.mp3", now=when,
    )  # fmt: skip

    assert ok is True
    (row,) = db.narrations_for_run(conn, run)
    assert (row["run_id"], row["category"]) == (run, "Business")
    assert row["script"] == script
    assert (row["model"], row["prompt_version"]) == ("script-model-x", prompts.NARRATE_PROMPT_VERSION)
    assert row["created_at"] == "2026-09-22T13:00:00Z"
    assert db.narrations_for_run(conn, run, category="Research") == []  # narrowing by category works
    assert conn.in_transaction is False  # committed, not left pending


def test_narrate_category_does_not_reuse_an_earlier_narration_of_the_same_run(conn, tmp_path):
    # Unlike digests, narrate is manual and on-demand: rerunning it is meant to produce
    # a fresh take, not be skipped because a narration already exists for today.
    run = add_run(conn)
    stories = narrate.category_stories_in_run(conn, run, "Research")
    path = tmp_path / "research.mp3"

    narrate.narrate_category(conn, FakeLLM(), FakeTTS(), "Research", stories, run_id=run, path=path)
    narrate.narrate_category(conn, FakeLLM(), FakeTTS(), "Research", stories, run_id=run, path=path)

    assert len(db.narrations_for_run(conn, run)) == 2  # both kept, neither overwritten


def test_narrate_category_leaves_an_existing_file_alone_when_the_script_call_fails(conn, tmp_path):
    run = add_run(conn)
    stories = narrate.category_stories_in_run(conn, run, "Research")
    path = tmp_path / "research.mp3"
    path.write_bytes(b"yesterday's narration")
    failing_llm, tts = FakeLLM(lambda text: LLMError("RateLimitError (HTTP 429): slow down")), FakeTTS()

    ok, script, error = narrate.narrate_category(conn, failing_llm, tts, "Research", stories, run_id=run, path=path)

    assert (ok, script) == (False, None)  # no script to show: it was never written
    assert "slow down" in error
    assert tts.calls == []  # never reached
    assert path.read_bytes() == b"yesterday's narration"


def test_narrate_category_leaves_an_existing_file_alone_when_tts_fails(conn, tmp_path):
    run = add_run(conn)
    stories = narrate.category_stories_in_run(conn, run, "Research")
    path = tmp_path / "research.mp3"
    path.write_bytes(b"yesterday's narration")
    failing_tts = FakeTTS(lambda text: LLMError("APIConnectionError: Connection error."))

    ok, script, error = narrate.narrate_category(conn, FakeLLM(), failing_tts, "Research", stories, run_id=run, path=path)

    assert (ok, script) == (False, GOOD_SCRIPT["script"])  # the script is still returned, for debugging
    assert "Connection error" in error
    assert path.read_bytes() == b"yesterday's narration"
    # The script was persisted before TTS ran, so the record survives even though TTS failed.
    assert db.narrations_for_run(conn, run)[0]["script"] == GOOD_SCRIPT["script"]


def test_narrate_category_writes_to_the_default_per_category_path(conn, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = add_run(conn, stories={"Product Release": [("T", "S")]})
    stories = narrate.category_stories_in_run(conn, run, "Product Release")

    ok, _, _ = narrate.narrate_category(conn, FakeLLM(), FakeTTS(), "Product Release", stories, run_id=run)

    assert (tmp_path / "audio" / "product-release.mp3").read_bytes() == b"fake-audio-bytes"


# --- narrate_all_categories: every category, isolated failures ------------------


def test_narrate_all_categories_narrates_every_category_with_stories_and_skips_the_rest(conn, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # narrate_all_categories writes audio/<category>.mp3 relative to cwd
    run = add_run(conn, stories={
        "Research": [("R", "Research summary.")],
        "Business": [("B", "Business summary.")],
    })  # fmt: skip
    fake_llm, fake_tts = FakeLLM(), FakeTTS()

    summary = narrate.narrate_all_categories(conn, fake_llm, fake_tts)

    assert summary.run_id == run
    assert sorted(o.category for o in summary.succeeded) == ["Business", "Research"]
    assert set(summary.no_stories) == {"Product Release", "Industry News", "Regulation & Policy", "Other"}
    assert len(fake_llm.calls) == 2 and len(fake_tts.calls) == 2  # only the two populated categories


def test_narrate_all_categories_isolates_one_categorys_failure_from_the_others(conn, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # Research succeeds and writes audio/research.mp3 relative to cwd
    add_run(conn, stories={
        "Research": [("R", "Research summary.")],
        "Business": [("B", "Business summary.")],
    })  # fmt: skip

    def respond(input_text):
        if "Category: Business" in input_text:
            raise LLMError("RateLimitError (HTTP 429): slow down")
        return dict(GOOD_SCRIPT)

    summary = narrate.narrate_all_categories(conn, FakeLLM(respond), FakeTTS())

    outcomes = {o.category: o for o in summary.outcomes}
    assert outcomes["Research"].ok is True
    assert outcomes["Business"].ok is False and "slow down" in outcomes["Business"].error
    assert len(summary.succeeded) == 1 and len(summary.failed) == 1


def test_narrate_all_categories_survives_an_unexpected_exception_in_one_category(conn, tmp_path, monkeypatch):
    # generate_validated already turns an LLM-side exception into a normal failed
    # outcome, so to test narrate_all_categories' own safety net this has to break
    # somewhere that isn't already guarded: fetching one category's stories.
    monkeypatch.chdir(tmp_path)  # Research succeeds and writes audio/research.mp3 relative to cwd
    add_run(conn, stories={
        "Research": [("R", "Research summary.")],
        "Business": [("B", "Business summary.")],
    })  # fmt: skip
    real_fetch = narrate.category_stories_in_run

    def fetch(conn, run_id, category):
        if category == "Business":
            raise ValueError("something nobody anticipated")
        return real_fetch(conn, run_id, category)

    monkeypatch.setattr(narrate, "category_stories_in_run", fetch)

    summary = narrate.narrate_all_categories(conn, FakeLLM(), FakeTTS())

    outcomes = {o.category: o for o in summary.outcomes}
    assert outcomes["Research"].ok is True
    assert outcomes["Business"].ok is False and "unexpected ValueError" in outcomes["Business"].error
    assert "Business" not in summary.no_stories  # a real error, not an empty category


def test_narrate_all_categories_reports_run_id_none_with_no_fully_processed_run(conn):
    summary = narrate.narrate_all_categories(conn, FakeLLM(), FakeTTS())

    assert summary.run_id is None and summary.outcomes == [] and summary.no_stories == []


def test_narrate_all_categories_calls_report_once_per_attempted_category(conn, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # both categories succeed and write audio/*.mp3 relative to cwd
    add_run(conn, stories={"Research": [("R", "S.")], "Business": [("B", "S.")]})
    seen = []

    narrate.narrate_all_categories(conn, FakeLLM(), FakeTTS(), report=seen.append)

    assert sorted(o.category for o in seen) == ["Business", "Research"]


# --- the command -----------------------------------------------------------


def test_narrate_command_reports_no_run_isolates_a_failure_and_writes_per_category_files(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "t.db"

    def no_key(*a, **k):
        raise llm.LLMConfigError("OPENAI_API_KEY is not set.")

    monkeypatch.setattr(llm, "create", no_key)
    assert main(["narrate", "--db", str(db_path)]) == 2
    assert "Cannot narrate" in capsys.readouterr().out

    monkeypatch.setattr(llm, "create", lambda model: FakeLLM())
    monkeypatch.setattr(llm, "create_tts", lambda: FakeTTS())
    db.connect(db_path).close()  # no run at all yet
    assert main(["narrate", "--db", str(db_path)]) == 1
    assert "No fully processed run" in capsys.readouterr().out

    conn = db.connect(db_path)
    add_run(conn, stories={"Research": [("Command line story", "Summary text.")], "Business": [("B", "S.")]}, key="h2")
    conn.close()

    def respond(input_text):
        if "Category: Business" in input_text:
            raise LLMError("RateLimitError (HTTP 429): slow down")
        return dict(GOOD_SCRIPT)

    fake_llm, fake_tts = FakeLLM(respond), FakeTTS()
    monkeypatch.setattr(llm, "create", lambda model: fake_llm)
    monkeypatch.setattr(llm, "create_tts", lambda: fake_tts)

    code = main(["narrate", "--db", str(db_path)])
    out = capsys.readouterr().out

    assert code == 0  # a retryable per-category failure is not a stage failure
    assert "ok" in out and "Research" in out and GOOD_SCRIPT["script"] in out  # printed for review
    assert "FAILED" in out and "Business" in out and "slow down" in out
    assert re.search(r"narrated this run:\s+1\b", out) and re.search(r"failed this run:\s+1\b", out)
    assert (tmp_path / "audio" / "research.mp3").read_bytes() == b"fake-audio-bytes"
    assert not (tmp_path / "audio" / "business.mp3").exists()  # failed: nothing written
