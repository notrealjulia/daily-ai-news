"""AI enrichment: classify and summarize every article that has a ready body.

For each article the LLM returns exactly three things: a category from a fixed list, a
short summary, and an English version of the title if the title is not English (else
null). Results are stored in `enrichments`, separate from the article, with the model
and prompt_version that produced them. The article's own title is never changed.

    - An article is enriched once per (model, prompt_version). Change either and the
      article becomes eligible again, alongside the old result.
    - A failed article stores nothing, so it is simply tried again on the next run.
      One failure never stops the others.
    - Only articles in the last 24h (ingest.MAX_ARTICLE_AGE) are considered, the same
      window cluster uses. An article that ages out is never enriched, so a persistent
      failure does not turn into a growing backlog of ever-older retries.

This module knows nothing about any provider; see ainews.llm for that.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ainews import db, ingest
from ainews.llm import LLMError, StructuredLLM
from ainews.prompts import CATEGORY_LIST
from ainews.prompts import ENRICH_INSTRUCTIONS as INSTRUCTIONS
from ainews.prompts import ENRICH_PROMPT_VERSION as PROMPT_VERSION

# CATEGORY_LIST, INSTRUCTIONS and PROMPT_VERSION live in ainews.prompts, along with every
# other stage's prompt; only the derived, schema-facing CATEGORIES tuple, the schema
# itself, and validation are enrich's own.

# Articles longer than this are cut before being sent, to bound cost.
MAX_BODY_CHARS = 24_000
# A sanity cap on the summary (four sentences never need this much); the sentence limit
# itself is asked for in the instructions, not checked in code.
MAX_SUMMARY_CHARS = 1200
# A sanity cap on an English title; a headline never needs this many characters.
MAX_TITLE_CHARS = 250

SCHEMA_NAME = "article_enrichment"

CATEGORIES = tuple(category.name for category in CATEGORY_LIST)

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "summary": {"type": "string"},
        "english_title": {"type": ["string", "null"]},
    },
    "required": ["category", "summary", "english_title"],
    "additionalProperties": False,
}


def build_input(source: str, title: str, body: str) -> str:
    """The article as the LLM sees it."""
    body = body.replace("</article>", "")  # the article must not be able to close its own tag
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n[article text truncated]"
    return f"Source: {source}\nTitle: {title}\n\n<article>\n{body}\n</article>"


# --- Checking what the LLM returned ------------------------------------------


class InvalidOutput(ValueError):
    """The LLM answered, but not with something we can accept."""


@dataclass(frozen=True)
class Enrichment:
    category: str
    summary: str
    english_title: str | None  # None: the title was already English


def parse_category_and_summary(raw: dict) -> tuple[str, str]:
    """Validate the `category` and `summary` of an answer (combined story summaries reuse this)."""
    category, summary = raw["category"], raw["summary"]
    if not isinstance(category, str) or category not in CATEGORIES:
        raise InvalidOutput(f"invalid category {category!r}")
    if not isinstance(summary, str) or not summary.strip():
        raise InvalidOutput("the summary is empty")
    summary = summary.strip()
    if len(summary) > MAX_SUMMARY_CHARS:
        raise InvalidOutput(f"the summary is {len(summary):,} chars (limit {MAX_SUMMARY_CHARS:,})")
    return category, summary


def parse_enrichment(raw: dict) -> Enrichment:
    """Validate the LLM's answer strictly; nothing is corrected or guessed."""
    if set(raw) != {"category", "summary", "english_title"}:
        raise InvalidOutput(f"expected exactly category, summary and english_title, got {sorted(raw)}")
    category, summary = parse_category_and_summary(raw)
    english_title = raw["english_title"]
    if english_title is not None:
        if not isinstance(english_title, str) or not english_title.strip():
            raise InvalidOutput("the english_title is empty (it must be null for an English title)")
        english_title = english_title.strip()
        if "\n" in english_title:
            raise InvalidOutput("the english_title is not a single line")
        if len(english_title) > MAX_TITLE_CHARS:
            raise InvalidOutput(f"the english_title is {len(english_title):,} chars (limit {MAX_TITLE_CHARS:,})")
    return Enrichment(category, summary, english_title)


# --- The stage ---------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    article_id: int
    source: str
    title: str
    ok: bool
    category: str | None  # when ok
    error: str | None  # when not ok


@dataclass
class EnrichmentSummary:
    already_enriched: int  # ready articles that already have a result for this model + prompt
    not_ready: int  # articles with no ready body yet, so not eligible (run `extract`)
    left_for_later: int = 0  # eligible, but beyond --limit
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def succeeded(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.ok]


def run_enrichment(
    conn: sqlite3.Connection,
    llm: StructuredLLM,
    *,
    prompt_version: str = PROMPT_VERSION,
    limit: int | None = None,
    now: datetime | None = None,
    report: Callable[[Outcome], None] | None = None,
) -> EnrichmentSummary:
    """Enrich every article in the 24h window that has a ready body and no result for this model + prompt.

    `report` is called after each article, so a long run can show progress. Each result
    is committed as it happens.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    window_start = now - ingest.MAX_ARTICLE_AGE
    overview = db.enrichment_overview(conn, llm.model, prompt_version, window_start)
    waiting = db.articles_needing_enrichment(conn, llm.model, prompt_version, window_start, limit)
    summary = EnrichmentSummary(
        already_enriched=overview["enriched"],
        not_ready=overview["not_ready"],
        left_for_later=overview["ready"] - overview["enriched"] - len(waiting),
    )

    for row in waiting:
        result: Enrichment | None = None
        try:
            raw = llm.generate(
                instructions=INSTRUCTIONS,
                input_text=build_input(row["source"], row["title"], row["body"]),
                schema_name=SCHEMA_NAME,
                schema=SCHEMA,
            )
            result = parse_enrichment(raw)
            error = None
        except LLMError as e:
            error = str(e)
        except InvalidOutput as e:
            error = f"invalid output: {e}"
        except Exception as e:  # one article must never stop the rest
            error = f"unexpected {type(e).__name__}: {e}"

        if result is not None:
            db.insert_enrichment(
                conn,
                article_id=row["id"],
                category=result.category,
                summary=result.summary,
                english_title=result.english_title,
                model=llm.model,
                prompt_version=prompt_version,
                created_at=now,
            )
            conn.commit()

        outcome = Outcome(
            article_id=row["id"],
            source=row["source"],
            title=row["title"],
            ok=result is not None,
            category=None if result is None else result.category,
            error=error,
        )
        summary.outcomes.append(outcome)
        if report:
            report(outcome)

    return summary
