"""Category digests: a headline and a short summary per category, made from a story run's stories.

Independent of clustering: it reads a finished story run and never regroups anything.

    - Digests are made from stories, not articles.
    - The headline and the summary come from the same LLM call and are stored together;
      an invalid headline fails the digest just as an invalid summary does.
    - It refuses to run from a run with any story whose summary isn't ready, since a
      digest silently missing stories would mislead. Run `cluster` again first.
    - Each category with at least one story gets one digest. The LLM is given that
      category's story count, the total non-Spam story count, and the story counts of the
      other categories in the same window, as context for how busy the category was. The
      UI shows the counts itself, so the digest text is told to synthesize the
      developments rather than repeat the numbers. There is no history yet, so it is also
      told never to compare with earlier days.
    - One digest per (run, category, model, prompt version). Failures store nothing, so
      that category is simply tried again on the next run; one failing never stops the
      others.

This module knows nothing about any provider; see ainews.llm for that.
"""

import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple

from ainews import db, enrich, ingest
from ainews.llm import StructuredLLM
from ainews.prompts import DIGEST_INSTRUCTIONS as INSTRUCTIONS
from ainews.prompts import DIGEST_PROMPT_VERSION
from ainews.stories import NON_SPAM_CATEGORIES, generate_validated

# INSTRUCTIONS and DIGEST_PROMPT_VERSION live in ainews.prompts, along with every other
# stage's prompt; the schema below is this module's own.

SCHEMA_NAME = "category_digest"

SCHEMA = {
    "type": "object",
    "properties": {"headline": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["headline", "summary"],
    "additionalProperties": False,
}

# The prompt asks for 6 to 12 words; this only catches an answer that has run away.
MAX_HEADLINE_CHARS = 150


def build_input(category: str, stories: list[sqlite3.Row], counts: dict[str, int]) -> str:
    """One category's stories plus the activity context (all counts are worked out here)."""
    total = sum(counts.values())
    count = len(stories)
    share = int(100 * count / total + 0.5)
    others = ", ".join(f"{name} {n}" for name, n in counts.items() if name != category) or "none"
    hours = int(ingest.MAX_ARTICLE_AGE.total_seconds() / 3600)
    lines = [
        f"Category: {category}",
        f"Stories in this category: {count} of {total} non-Spam stories in the last {hours} hours ({share}%)",
        f"Story counts of the other categories in the same window: {others}",
        "",
        "<stories>",
    ]
    for number, story in enumerate(stories, start=1):
        n = story["article_count"]
        summary = story["summary"].replace("</stories>", "")
        lines.append(f"[{number}] ({n} article{'s' if n != 1 else ''}) {summary}")
    lines.append("</stories>")
    return "\n".join(lines)


class Digest(NamedTuple):
    headline: str
    summary: str


def parse_digest(raw: dict) -> Digest:
    """Validate the digest answer: exactly a headline and a summary, each non-empty and within its cap."""
    if set(raw) != {"headline", "summary"}:
        raise enrich.InvalidOutput(f"expected exactly headline and summary, got {sorted(raw)}")
    headline, summary = raw["headline"], raw["summary"]
    if not isinstance(headline, str) or not headline.strip():
        raise enrich.InvalidOutput("the headline is empty")
    headline = headline.strip()
    if "\n" in headline:
        raise enrich.InvalidOutput("the headline is not a single line")
    if len(headline) > MAX_HEADLINE_CHARS:
        raise enrich.InvalidOutput(
            f"the headline is {len(headline):,} chars (limit {MAX_HEADLINE_CHARS:,})"
        )
    if not isinstance(summary, str) or not summary.strip():
        raise enrich.InvalidOutput("the summary is empty")
    summary = summary.strip()
    if len(summary) > enrich.MAX_SUMMARY_CHARS:
        raise enrich.InvalidOutput(
            f"the summary is {len(summary):,} chars (limit {enrich.MAX_SUMMARY_CHARS:,})"
        )
    return Digest(headline, summary)


# --- The stage ---------------------------------------------------------------


@dataclass(frozen=True)
class DigestOutcome:
    category: str
    story_count: int
    ok: bool
    error: str | None


@dataclass
class DigestSummary:
    run_id: int | None  # None if there is no story run to work from
    total_stories: int = 0
    incomplete_stories: int = 0  # stories without a ready summary; if > 0 nothing is done
    already_done: list[str] = field(default_factory=list)  # categories that had a digest
    no_stories: list[str] = field(default_factory=list)  # categories with nothing to digest
    outcomes: list[DigestOutcome] = field(default_factory=list)

    @property
    def succeeded(self) -> list[DigestOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[DigestOutcome]:
        return [o for o in self.outcomes if not o.ok]


def run_digests(
    conn: sqlite3.Connection,
    llm: StructuredLLM,
    *,
    run_id: int | None = None,
    prompt_version: str = DIGEST_PROMPT_VERSION,
    now: datetime | None = None,
    report: Callable[[DigestOutcome], None] | None = None,
) -> DigestSummary:
    """Make a digest for every category of a story run (the latest one by default)."""
    run = db.get_story_run(conn, run_id)
    if run is None:
        return DigestSummary(run_id=None)
    summary = DigestSummary(run_id=run["id"])

    stories = db.stories_in_run(conn, run["id"])
    summary.total_stories = len(stories)
    summary.incomplete_stories = sum(1 for s in stories if s["summary_status"] != "ready")
    if summary.incomplete_stories:
        return summary  # refuse: a digest must never silently miss stories

    by_category = Counter(s["category"] for s in stories)
    counts = {name: by_category[name] for name in NON_SPAM_CATEGORIES if by_category[name]}
    summary.no_stories = [name for name in NON_SPAM_CATEGORIES if not by_category[name]]

    for category in counts:
        if db.digest_exists(
            conn, run_id=run["id"], category=category, model=llm.model, prompt_version=prompt_version
        ):
            summary.already_done.append(category)
            continue

        category_stories = [s for s in stories if s["category"] == category]
        result, error = generate_validated(
            llm,
            instructions=INSTRUCTIONS,
            input_text=build_input(category, category_stories, counts),
            schema_name=SCHEMA_NAME,
            schema=SCHEMA,
            parse=parse_digest,
        )
        if result is not None:
            db.insert_digest(
                conn,
                run_id=run["id"],
                category=category,
                story_count=len(category_stories),
                total_story_count=len(stories),
                headline=result.headline,
                summary=result.summary,
                model=llm.model,
                prompt_version=prompt_version,
                created_at=now or datetime.now(timezone.utc),
            )
            conn.commit()

        outcome = DigestOutcome(category, len(category_stories), result is not None, error)
        summary.outcomes.append(outcome)
        if report:
            report(outcome)

    return summary
