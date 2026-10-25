"""Story scores: significance and release relevance, 0-2 each, for every story (de-noising, step 1).

    - A story is scored from its own content only: its title, its combined summary and the
      titles of its other articles. Never the source, the article count or any history.
    - Stories are sent BATCH_SIZE at a time, numbered within the batch, in one LLM call.
    - One score per story. Stories that already have one are skipped, so a run only scores
      what is new; rescore=True replaces existing scores instead.
    - A failed batch, or a story missing from an answer, stores nothing: it is simply scored
      on the next run. One failing never stops the others.
    - Scores are only stored here; nothing is filtered by them.

This module knows nothing about any provider; see ainews.llm for that.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple

from ainews import db, enrich
from ainews.llm import StructuredLLM
from ainews.prompts import SCORE_INSTRUCTIONS as INSTRUCTIONS
from ainews.prompts import SCORE_PROMPT_VERSION
from ainews.stories import generate_validated

# INSTRUCTIONS and SCORE_PROMPT_VERSION live in ainews.prompts, along with every other
# stage's prompt; the schema below is this module's own.

SCHEMA_NAME = "story_scores"

SCHEMA = {
    "type": "object",
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "story": {"type": "integer"},
                    "significance": {"type": "integer", "enum": [0, 1, 2]},
                    "release": {"type": "integer", "enum": [0, 1, 2]},
                    "reason": {"type": "string"},
                },
                "required": ["story", "significance", "release", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["stories"],
    "additionalProperties": False,
}

BATCH_SIZE = 20

# The prompt asks for at most 20 words; this only catches an answer that has run away.
MAX_REASON_CHARS = 300


class StoryText(NamedTuple):
    title: str
    summary: str
    other_titles: list[str]


def build_input(stories: list[StoryText]) -> str:
    """The numbered stories of one batch, in the same layout as the prompt's examples."""
    parts = []
    for number, s in enumerate(stories, start=1):
        text = f"[{number}] Title: {s.title}\nSummary: {s.summary}"
        if s.other_titles:
            text += f"\nOther article titles: {' | '.join(s.other_titles)}"
        parts.append(text)
    return "<stories>\n" + "\n\n".join(parts).replace("</stories>", "") + "\n</stories>"


class Score(NamedTuple):
    significance: int
    release: int
    reason: str


def parse_scores(raw: dict, count: int) -> dict[int, Score]:
    """Validate a batch answer: story number -> Score. Stories it leaves out are simply absent."""
    if set(raw) != {"stories"} or not isinstance(raw["stories"], list):
        raise enrich.InvalidOutput(f"expected exactly a stories list, got {sorted(raw)}")
    scores: dict[int, Score] = {}
    for item in raw["stories"]:
        number = item.get("story")
        if type(number) is not int or not 1 <= number <= count:
            raise enrich.InvalidOutput(f"story number {number!r} is not between 1 and {count}")
        if number in scores:
            raise enrich.InvalidOutput(f"story {number} was answered twice")
        values = (item.get("significance"), item.get("release"))
        if any(type(v) is not int or v not in (0, 1, 2) for v in values):
            raise enrich.InvalidOutput(f"story {number} has scores {values}, expected 0, 1 or 2")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise enrich.InvalidOutput(f"story {number} has no reason")
        if len(reason.strip()) > MAX_REASON_CHARS:
            raise enrich.InvalidOutput(f"story {number}'s reason is {len(reason.strip()):,} chars")
        scores[number] = Score(*values, reason.strip())
    return scores


# --- The stage ---------------------------------------------------------------


@dataclass(frozen=True)
class ScoreOutcome:
    story_id: int
    title: str
    ok: bool
    score: Score | None
    error: str | None


@dataclass
class ScoringSummary:
    already_scored: int = 0  # stories skipped because they have a score
    outcomes: list[ScoreOutcome] = field(default_factory=list)

    @property
    def succeeded(self) -> list[ScoreOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[ScoreOutcome]:
        return [o for o in self.outcomes if not o.ok]


def _story_texts(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> dict[int, StoryText]:
    titles: dict[int, list[str]] = {}
    for run_id in sorted({r["run_id"] for r in rows}):
        for link in db.story_article_links(conn, run_id):  # earliest article first
            titles.setdefault(link["story_id"], []).append(link["title"])
    texts = {}
    for r in rows:
        first, *others = titles.get(r["id"]) or [""]
        texts[r["id"]] = StoryText(first, r["summary"], others)
    return texts


def run_scoring(
    conn: sqlite3.Connection,
    llm: StructuredLLM,
    *,
    run_id: int | None,
    rescore: bool = False,
    prompt_version: str = SCORE_PROMPT_VERSION,
    batch_size: int = BATCH_SIZE,
    now: datetime | None = None,
    report: Callable[[ScoreOutcome], None] | None = None,
) -> ScoringSummary:
    """Score the ready stories of one story run (every run when run_id is None)."""
    rows = db.stories_to_score(conn, run_id)
    todo = [r for r in rows if rescore or not r["scored"]]
    summary = ScoringSummary(already_scored=len(rows) - len(todo))
    texts = _story_texts(conn, todo)

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        scores, error = generate_validated(
            llm,
            instructions=INSTRUCTIONS,
            input_text=build_input([texts[r["id"]] for r in batch]),
            schema_name=SCHEMA_NAME,
            schema=SCHEMA,
            parse=lambda raw, count=len(batch): parse_scores(raw, count),
        )
        for number, row in enumerate(batch, start=1):
            score = scores.get(number) if scores is not None else None
            if score is not None:
                db.save_story_score(
                    conn, story_id=row["id"], significance=score.significance, release=score.release,
                    reason=score.reason, model=llm.model, prompt_version=prompt_version,
                    created_at=now or datetime.now(timezone.utc), replace=rescore,
                )  # fmt: skip
            outcome = ScoreOutcome(
                row["id"], texts[row["id"]].title, score is not None, score,
                None if score is not None else error or "no answer for this story",
            )  # fmt: skip
            summary.outcomes.append(outcome)
            if report:
                report(outcome)
        conn.commit()  # per batch, so a later crash never loses scores already paid for

    return summary
