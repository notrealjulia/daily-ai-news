"""Tests for story scores (`python -m ainews score`). The LLM is a fake."""

import pytest
from test_digest import NOW, make_run

from ainews import db, llm, score
from ainews.__main__ import main


class FakeLLM:
    """Answers each batch call: by default every story gets significance 1, release 0."""

    def __init__(self, respond=None, model="score-model"):
        self.model = model
        self.calls: list[str] = []
        self._respond = respond or (lambda count: answer(*[(n, 1, 0) for n in range(1, count + 1)]))

    def generate(self, *, instructions, input_text, schema_name, schema):
        assert instructions is score.INSTRUCTIONS
        self.calls.append(input_text)
        return self._respond(input_text.count("Title: "))


def answer(*items):
    return {"stories": [{"story": n, "significance": s, "release": r, "reason": f"reason {n}"} for n, s, r in items]}


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def scores(conn):
    return [tuple(r) for r in conn.execute(
        "SELECT story_id, significance, release, reason, model, prompt_version FROM story_scores ORDER BY story_id"
    )]  # fmt: skip


def test_scores_the_ready_stories_of_the_latest_run_in_batches_from_their_content_only(conn):
    make_run(conn, ("Research", "Old run story.", 1, "ready"), input_hash="old")
    make_run(
        conn,
        ("Research", "Two-article story.", 2, "ready"),
        ("Business", "Second story.", 1, "ready"),
        ("Other", "Third story.", 1, "ready"),
        ("Other", "Not summarized yet.", 1, "pending"),
        input_hash="new",
    )
    fake = FakeLLM()

    summary = score.run_scoring(conn, fake, run_id=db.get_story_run(conn)["id"], batch_size=2, now=NOW)

    assert len(fake.calls) == 2  # 3 ready stories, batches of 2
    first = fake.calls[0]
    assert "[1] Title: Article new-0-0\nSummary: Two-article story.\nOther article titles: Article new-0-1" in first
    assert "[2] Title: Article new-1-0" in first
    assert "Src" not in first and "Old run story." not in first  # no source, no other run
    assert [(s[1], s[2], s[4], s[5]) for s in scores(conn)] == [(1, 0, "score-model", score.SCORE_PROMPT_VERSION)] * 3
    assert len(summary.succeeded) == 3 and summary.already_scored == 0


def test_only_unscored_stories_are_scored_unless_rescore_is_asked(conn):
    run_id = make_run(conn, ("Research", "A.", 1, "ready"), ("Research", "B.", 1, "ready"))
    score.run_scoring(conn, FakeLLM(lambda count: answer((1, 2, 2))), run_id=run_id, now=NOW)  # B unanswered
    assert [s[:3] for s in scores(conn)] == [(1, 2, 2)]

    fake = FakeLLM()
    summary = score.run_scoring(conn, fake, run_id=run_id, now=NOW)
    assert len(fake.calls) == 1 and fake.calls[0].count("Title: ") == 1  # only B
    assert summary.already_scored == 1
    assert [s[:3] for s in scores(conn)] == [(1, 2, 2), (2, 1, 0)]  # A kept

    summary = score.run_scoring(conn, FakeLLM(lambda count: answer((1, 0, 0), (2, 0, 1))), run_id=run_id,
                                rescore=True, now=NOW)  # fmt: skip
    assert summary.already_scored == 0
    assert [s[:3] for s in scores(conn)] == [(1, 0, 0), (2, 0, 1)]


def test_a_failed_batch_stores_nothing_and_never_stops_the_others(conn):
    run_id = make_run(conn, *[("Research", f"Story {n}.", 1, "ready") for n in range(3)])

    def respond(count):
        if count == 2:
            raise llm.LLMError("rate limited")
        return answer((1, 1, 1))

    summary = score.run_scoring(conn, FakeLLM(respond), run_id=run_id, batch_size=2, now=NOW)

    assert [o.error for o in summary.failed] == ["rate limited", "rate limited"]
    assert [s[:3] for s in scores(conn)] == [(3, 1, 1)]


@pytest.mark.parametrize(
    "raw",
    [
        {"stories": [{"story": 3, "significance": 1, "release": 0, "reason": "r"}]},  # no story 3
        {"stories": [{"story": 1, "significance": 1, "release": 0, "reason": "r"}] * 2},  # twice
        {"stories": [{"story": 1, "significance": 3, "release": 0, "reason": "r"}]},
        {"stories": [{"story": 1, "significance": True, "release": 0, "reason": "r"}]},
        {"stories": [{"story": 1, "significance": 1, "release": 0, "reason": " "}]},
        {"stories": [{"story": 1, "significance": 1, "release": 0, "reason": "x" * 301}]},
        {"answers": []},
    ],
)
def test_invalid_answers_are_rejected(raw):
    with pytest.raises(score.enrich.InvalidOutput):
        score.parse_scores(raw, 2)


def test_the_cli_exits_non_zero_only_when_every_story_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "ainews.db"
    conn = db.connect(db_path)
    make_run(conn, ("Research", "A.", 1, "ready"), ("Research", "B.", 1, "ready"))
    conn.close()

    monkeypatch.setattr(llm, "create", lambda model: FakeLLM(lambda count: answer((1, 1, 0))))
    assert main(["score", "--db", str(db_path)]) == 0  # one of two failing is isolated

    def fail(count):
        raise llm.LLMError("401 bad key")

    monkeypatch.setattr(llm, "create", lambda model: FakeLLM(fail))
    assert main(["score", "--db", str(db_path), "--all", "--rescore"]) == 1
