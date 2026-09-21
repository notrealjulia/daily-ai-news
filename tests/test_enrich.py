"""Tests for the enrichment stage (`python -m ainews enrich`).

The LLM is a fake that returns whatever each test tells it to; no provider code and no
network is involved.
"""

import re
from datetime import datetime, timezone

import pytest

from ainews import db, enrich, llm
from ainews.__main__ import main

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
GOOD = {"category": "Product Release", "summary": "A company released a product.", "english_title": None}


class FakeLLM:
    """Stands in for a provider. `respond` gets the input text and returns a dict or raises."""

    def __init__(self, respond=None, model="fake-model"):
        self.model = model
        self.calls: list[dict] = []
        self._respond = respond or (lambda input_text: dict(GOOD))

    def generate(self, *, instructions, input_text, schema_name, schema):
        self.calls.append(
            {"instructions": instructions, "input_text": input_text,
             "schema_name": schema_name, "schema": schema}
        )  # fmt: skip
        return self._respond(input_text)


def by_title(**answers):
    """A `respond` that picks the answer by which article title appears in the input."""

    def respond(input_text):
        for title, answer in answers.items():
            if f"Title: {title}\n" in input_text:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise AssertionError(f"no answer prepared for: {input_text[:80]!r}")

    return respond


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def add_ready(conn, title, *, body="The article text.", source="Src", url=None) -> int:
    url = url or f"http://x/{title}"
    db.insert_article(
        conn, source=source, url=url, title=title,
        published_at=NOW, fetched_at=NOW, feed_text="teaser",
    )  # fmt: skip
    article_id = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()["id"]
    db.mark_body_ready(conn, article_id, body=body, source="fulltext", checked_at=NOW)
    conn.commit()
    return article_id


def add_without_body(conn, title, *, failed=False) -> int:
    url = f"http://x/{title}"
    db.insert_article(
        conn, source="Src", url=url, title=title,
        published_at=NOW, fetched_at=NOW, feed_text="teaser",
    )  # fmt: skip
    article_id = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()["id"]
    if failed:
        db.mark_body_failed(conn, article_id, error="HTTP 403", checked_at=NOW)
    conn.commit()
    return article_id


def stored(conn):
    return [tuple(r) for r in conn.execute(
        "SELECT article_id, category, summary, model, prompt_version FROM enrichments ORDER BY id"
    )]  # fmt: skip


def run(conn, fake, **kwargs):
    return enrich.run_enrichment(conn, fake, now=NOW, **kwargs)


def shows(out: str, pattern: str) -> bool:
    """Whether the output matches a regex (spacing between columns doesn't matter)."""
    return re.search(pattern, out) is not None


# --- what the LLM is asked -----------------------------------------------------


def test_categories_prompt_and_schema_agree():
    # Structure only, not wording: the prompt and the schema are both built from
    # CATEGORY_LIST, so a category can't be in one and missing from the other.
    for number, category in enumerate(enrich.CATEGORY_LIST, start=1):
        assert f"{number}. {category.name}" in enrich.INSTRUCTIONS
        assert category.description in enrich.INSTRUCTIONS

    schema = enrich.SCHEMA
    assert schema["properties"]["category"]["enum"] == list(enrich.CATEGORIES)
    assert set(schema["properties"]) == {"category", "summary", "english_title"}  # exactly these
    assert schema["required"] == ["category", "summary", "english_title"]
    assert schema["additionalProperties"] is False


def test_the_request_carries_the_prompt_the_schema_and_the_article(conn):
    add_ready(conn, "Big news", body="The full article body.", source="TechCrunch AI")
    fake = FakeLLM()

    run(conn, fake)

    (call,) = fake.calls
    assert call["instructions"] == enrich.INSTRUCTIONS
    assert call["schema"] == enrich.SCHEMA and call["schema_name"] == enrich.SCHEMA_NAME
    for part in ("TechCrunch AI", "Big news", "The full article body."):
        assert part in call["input_text"]


def test_long_articles_are_truncated_before_being_sent():
    body = "y" * (enrich.MAX_BODY_CHARS + 500)  # 'y' appears nowhere else in the input

    text = enrich.build_input("S", "T", body)

    assert text.count("y") == enrich.MAX_BODY_CHARS
    assert "[article text truncated]" in text


# --- successful enrichment ---------------------------------------------------


def test_a_successful_enrichment_is_stored_separately_from_the_article(conn):
    a = add_ready(conn, "Big news")
    fake = FakeLLM(model="gpt-test")

    summary = run(conn, fake)

    assert stored(conn) == [
        (a, "Product Release", "A company released a product.", "gpt-test", enrich.PROMPT_VERSION)
    ]
    assert len(summary.succeeded) == 1 and summary.failed == []
    article_columns = {r["name"] for r in conn.execute("PRAGMA table_info(articles)")}
    assert not article_columns & {"category", "summary", "model", "prompt_version"}
    created = conn.execute("SELECT created_at FROM enrichments").fetchone()["created_at"]
    assert created == "2026-09-20T12:00:00Z"


# --- English titles ----------------------------------------------------------


def test_a_translated_title_is_stored_beside_the_article_and_an_english_one_is_left_alone(conn):
    danish, english = add_ready(conn, "Danske startups rejser rekordstor kapital"), add_ready(conn, "OpenAI ships a model")
    fake = FakeLLM(by_title(**{
        "Danske startups rejser rekordstor kapital": {**GOOD, "english_title": "Danish startups raise record capital"},
        "OpenAI ships a model": {**GOOD, "english_title": None},  # already English: the model says so
    }))

    run(conn, fake)

    titles = {r["article_id"]: r["english_title"] for r in conn.execute("SELECT article_id, english_title FROM enrichments")}
    assert titles == {danish: "Danish startups raise record capital", english: None}
    # The source data is untouched, and the model was shown the original title to judge.
    originals = {r["id"]: r["title"] for r in conn.execute("SELECT id, title FROM articles")}
    assert originals == {danish: "Danske startups rejser rekordstor kapital", english: "OpenAI ships a model"}
    assert "Title: Danske startups rejser rekordstor kapital\n" in fake.calls[0]["input_text"]


# --- invalid output ----------------------------------------------------------


def test_an_invalid_category_is_a_failure_and_nothing_is_stored(conn):
    add_ready(conn, "Anything")

    summary = run(conn, FakeLLM(lambda _: {"category": "Models", "summary": "Fine.", "english_title": None}))

    assert stored(conn) == []
    (outcome,) = summary.failed
    assert outcome.error == "invalid output: invalid category 'Models'"


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"category": "product release", "summary": "Fine.", "english_title": None}, "invalid category"),  # no fuzzy matching
        ({"category": "Other", "summary": "", "english_title": None}, "summary is empty"),
        ({"category": "Other", "summary": "x" * (enrich.MAX_SUMMARY_CHARS + 1), "english_title": None}, "limit"),
        ({"category": "Other", "summary": "Fine."}, "expected exactly category, summary and english_title"),
        ({"category": "Other", "summary": "Fine.", "english_title": None, "relevance": 9}, "expected exactly"),
        ({"category": "Other", "summary": "Fine.", "english_title": "  "}, "english_title is empty"),
        ({"category": "Other", "summary": "Fine.", "english_title": 7}, "english_title is empty"),
        ({"category": "Other", "summary": "Fine.", "english_title": "Two\nlines"}, "not a single line"),
        ({"category": "Other", "summary": "Fine.", "english_title": "x" * (enrich.MAX_TITLE_CHARS + 1)}, "chars (limit"),
    ],
    ids=["category-not-matched-fuzzily", "empty-summary", "summary-too-long", "missing-english-title", "extra-key",
         "blank-english-title", "non-string-english-title", "multi-line-english-title", "runaway-english-title"],
)
def test_unacceptable_answers_are_rejected(conn, answer, message):
    add_ready(conn, "Anything")

    summary = run(conn, FakeLLM(lambda _: answer))

    assert stored(conn) == []
    assert message in summary.failed[0].error


# --- skipping what is already done -------------------------------------------


def test_an_article_is_not_enriched_again_for_the_same_model_and_prompt(conn):
    add_ready(conn, "One")
    first = FakeLLM()
    run(conn, first)
    assert len(first.calls) == 1

    second = FakeLLM()
    summary = run(conn, second)

    assert second.calls == []  # the LLM was not asked again
    assert summary.already_enriched == 1 and summary.outcomes == []
    assert len(stored(conn)) == 1


def test_spam_is_accepted_and_stored(conn):
    add_ready(conn, "Prices go up in 7 days. Get your Disrupt ticket now")

    run(conn, FakeLLM(lambda _: {"category": "Spam", "summary": "A promotion for conference tickets.", "english_title": None}))

    assert stored(conn)[0][1] == "Spam"


@pytest.mark.parametrize(
    ("second_model", "second_prompt"),
    [("model-b", enrich.PROMPT_VERSION), ("model-a", "a-later-prompt")],
    ids=["different-model", "different-prompt-version"],
)
def test_a_different_model_or_prompt_version_is_enriched_again_alongside_the_first(
    conn, second_model, second_prompt
):
    a = add_ready(conn, "One")
    run(conn, FakeLLM(model="model-a"))  # the first result, under the current prompt
    second = FakeLLM(lambda _: {"category": "Research", "summary": "Other view.", "english_title": None}, model=second_model)

    run(conn, second, prompt_version=second_prompt)

    assert len(second.calls) == 1
    assert [(r[0], r[1], r[3], r[4]) for r in stored(conn)] == [
        (a, "Product Release", "model-a", enrich.PROMPT_VERSION),
        (a, "Research", second_model, second_prompt),
    ]


# --- failures: retry and isolation -------------------------------------------


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (llm.LLMError("RateLimitError (HTTP 429): slow down"), "RateLimitError (HTTP 429): slow down"),
        (KeyError("boom"), "unexpected KeyError"),
        ({"category": "Bogus", "summary": "x", "english_title": None}, "invalid output"),
    ],
    ids=["llm-error", "unexpected-exception", "invalid-answer"],
)
def test_one_failing_article_does_not_stop_the_others(conn, failure, expected_error):
    a, _, c = (add_ready(conn, title) for title in ("One", "Two", "Three"))
    fake = FakeLLM(by_title(One=dict(GOOD), Two=failure, Three=dict(GOOD)))

    summary = run(conn, fake)

    assert [row[0] for row in stored(conn)] == [a, c]
    assert [o.ok for o in summary.outcomes] == [True, False, True]
    assert expected_error in summary.failed[0].error


def test_a_failed_article_is_retried_on_the_next_run(conn):
    a = add_ready(conn, "One")
    outage = FakeLLM(lambda _: (_ for _ in ()).throw(llm.LLMError("APIConnectionError: Connection error.")))

    first = run(conn, outage)

    assert stored(conn) == []  # nothing stored for a failure
    assert first.failed[0].error == "APIConnectionError: Connection error."

    fake = FakeLLM()
    second = run(conn, fake)

    assert len(fake.calls) == 1  # it was picked up again
    assert stored(conn)[0][0] == a and len(second.succeeded) == 1


# --- which articles are eligible ---------------------------------------------


def test_only_articles_with_a_ready_body_are_sent_to_the_llm(conn):
    ready = add_ready(conn, "Ready")
    add_without_body(conn, "Pending")
    add_without_body(conn, "FailedBody", failed=True)
    fake = FakeLLM()

    summary = run(conn, fake)

    assert [row[0] for row in stored(conn)] == [ready]
    assert len(fake.calls) == 1
    assert summary.not_ready == 2


def test_limit_caps_how_many_articles_are_processed(conn):
    for title in ("One", "Two", "Three", "Four"):
        add_ready(conn, title)
    fake = FakeLLM()

    summary = run(conn, fake, limit=3)

    assert len(fake.calls) == 3 and len(stored(conn)) == 3
    assert summary.left_for_later == 1

    later = run(conn, FakeLLM())
    assert len(later.succeeded) == 1 and later.left_for_later == 0


# --- the command ---------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Run `enrich` against a database on disk with `llm.create` replaced by a fake."""
    db_path = tmp_path / "t.db"
    created_with = []

    def run_command(fake, *extra):
        monkeypatch.setattr(llm, "create", lambda model: created_with.append(model) or fake)
        return main(["enrich", "--db", str(db_path), *extra])

    connection = db.connect(db_path)
    yield connection, run_command, created_with
    connection.close()


def test_command_summary_shows_successes_failures_skips_and_retryable_failures(cli, capsys):
    conn, run_command, _ = cli
    add_ready(conn, "Done already")
    run(conn, FakeLLM(model="fake-model"))  # so one article is already enriched
    add_ready(conn, "Good one")
    add_ready(conn, "Bad one")
    add_without_body(conn, "No body yet")
    fake = FakeLLM(
        by_title(
            **{
                "Good one": {"category": "Research", "summary": "Fine.", "english_title": None},
                "Bad one": llm.LLMError("RateLimitError (HTTP 429): slow down"),
            }
        )
    )

    code = run_command(fake)
    out = capsys.readouterr().out

    assert code == 1  # something failed
    assert "Model: fake-model" in out and f"prompt_version: {enrich.PROMPT_VERSION}" in out
    assert "1 article(s) already enriched; 2 waiting; 1 without a ready body" in out
    assert shows(out, r"\bok\s+Research\s+Src\s+Good one")
    assert shows(out, r"FAILED\s+Src\s+Bad one.*RateLimitError \(HTTP 429\): slow down")
    assert shows(out, r"already enriched \(skipped\):\s+1\b")
    assert shows(out, r"enriched this run:\s+1\s+\(Research 1\)")
    assert shows(out, r"failed this run:\s+1\b")
    assert shows(out, r"no ready body yet:\s+1\s+\(run `python -m ainews extract`\)")
    assert "Retryable failures" in out and "python -m ainews enrich" in out
    assert "1 x RateLimitError (HTTP 429): slow down" in out


@pytest.mark.parametrize(
    ("args", "expected_model"),
    [((), llm.DEFAULT_MODEL), (("--model", "custom-model"), "custom-model")],
    ids=["default-model", "chosen-model"],
)
def test_command_uses_the_chosen_or_default_model(cli, args, expected_model):
    _, run_command, created_with = cli

    run_command(FakeLLM(), *args)

    assert created_with == [expected_model]


def test_command_without_an_api_key_stops_before_touching_anything(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY")  # the real llm.create, with no key anywhere
    db_path = tmp_path / "t.db"

    code = main(["enrich", "--db", str(db_path)])

    assert code == 2
    assert "Cannot enrich: OPENAI_API_KEY is not set" in capsys.readouterr().out
    assert not db_path.exists()
