"""Story clustering: group the window's enriched articles into stories.

A story is one underlying event, announcement, release, research result or
development. Articles are grouped only when they clearly describe the same thing;
similar topics are not enough, and category plays no part (the model is not even shown
it). When in doubt, articles stay separate.

    - Input: articles in the 24h window that are enriched (for the current enrichment
      version) and not Spam. The window uses published_at, or fetched_at for dateless
      articles.
    - One LLM call groups the whole window. It returns only groups of two or more, so
      anything it doesn't mention is its own story. If the call fails or its answer is
      invalid, nothing is stored and the next run tries again.
    - A single-article story copies its article's category and summary; no LLM call.
    - A multi-article story gets one combined summary and category from a second call,
      made per story. These are independent: one failing never blocks the others, and a
      rerun retries only the failed ones without regrouping.
    - The result is an immutable snapshot (a "story run") identified by the models,
      prompt version and a fingerprint of the input. Repeating a run with unchanged
      input finds the existing snapshot instead of making another.

This module knows nothing about any provider; see ainews.llm for that.
"""

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ainews import db, enrich, ingest
from ainews.llm import LLMError, StructuredLLM
from ainews.prompts import COMBINE_INSTRUCTIONS, GROUPING_INSTRUCTIONS, STORY_PROMPT_VERSION

# GROUPING_INSTRUCTIONS, COMBINE_INSTRUCTIONS and STORY_PROMPT_VERSION live in
# ainews.prompts, along with every other stage's prompt; the schemas below are this
# module's own, and reference NON_SPAM_CATEGORIES the same way enrich's schema does.

GROUPING_SCHEMA_NAME = "story_grouping"
COMBINE_SCHEMA_NAME = "story_summary"

SPAM = "Spam"
NON_SPAM_CATEGORIES = tuple(name for name in enrich.CATEGORIES if name != SPAM)


# --- What the LLM is asked ---------------------------------------------------

GROUPING_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_ids": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string"},
                },
                "required": ["article_ids", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}

COMBINE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(NON_SPAM_CATEGORIES)},
        "summary": {"type": "string"},
    },
    "required": ["category", "summary"],
    "additionalProperties": False,
}


def _article_block(source: str, title: str, summary: str, article_id: int | None = None) -> str:
    # The article must not be able to close its own tag.
    title, summary = title.replace("</article>", ""), summary.replace("</article>", "")
    opening = "<article>" if article_id is None else f'<article id="{article_id}">'
    return f"{opening}\nSource: {source}\nTitle: {title}\nSummary: {summary}\n</article>"


def build_grouping_input(candidates: list[sqlite3.Row]) -> str:
    return "\n\n".join(
        _article_block(r["source"], r["title"], r["summary"], r["article_id"]) for r in candidates
    )


def build_combine_input(members: list[sqlite3.Row]) -> str:
    return "\n\n".join(_article_block(r["source"], r["title"], r["summary"]) for r in members)


# --- Checking what the LLM returned ------------------------------------------


@dataclass(frozen=True)
class Group:
    article_ids: tuple[int, ...]
    reason: str


def parse_groups(raw: dict, valid_ids: set[int]) -> list[Group]:
    """Validate the grouping answer strictly; nothing is corrected or guessed."""
    if set(raw) != {"groups"} or not isinstance(raw["groups"], list):
        raise enrich.InvalidOutput("expected exactly a list called groups")
    groups, seen = [], set()
    for item in raw["groups"]:
        if not isinstance(item, dict) or set(item) != {"article_ids", "reason"}:
            raise enrich.InvalidOutput("each group needs exactly article_ids and reason")
        ids, reason = item["article_ids"], item["reason"]
        if not isinstance(ids, list) or not all(type(i) is int for i in ids):
            raise enrich.InvalidOutput("article_ids must be a list of integers")
        if len(set(ids)) != len(ids) or len(ids) < 2:
            raise enrich.InvalidOutput(f"a group needs at least two different articles, got {ids}")
        if unknown := sorted(set(ids) - valid_ids):
            raise enrich.InvalidOutput(f"unknown article ids {unknown}")
        if repeated := sorted(seen & set(ids)):
            raise enrich.InvalidOutput(f"articles {repeated} are in more than one group")
        if not isinstance(reason, str) or not reason.strip():
            raise enrich.InvalidOutput("a group needs a reason")
        seen.update(ids)
        groups.append(Group(tuple(sorted(ids)), reason.strip()))
    return groups


def parse_story_summary(raw: dict) -> enrich.Enrichment:
    """A combined summary has exactly a category and a summary, validated like an enrichment's, and can't be Spam."""
    if set(raw) != {"category", "summary"}:
        raise enrich.InvalidOutput(f"expected exactly category and summary, got {sorted(raw)}")
    category, summary = enrich.parse_category_and_summary(raw)
    if category == SPAM:
        raise enrich.InvalidOutput("a story cannot be Spam")
    return enrich.Enrichment(category, summary, english_title=None)  # stories have no title of their own


def generate_validated(llm: StructuredLLM, *, instructions, input_text, schema_name, schema, parse):
    """Ask the LLM and validate its answer: (result, None), or (None, reason it failed).

    Any failure, including one nobody anticipated, is returned rather than raised, so
    that one item failing can never stop the others.
    """
    try:
        raw = llm.generate(
            instructions=instructions, input_text=input_text, schema_name=schema_name, schema=schema
        )
        return parse(raw), None
    except LLMError as e:
        return None, str(e)
    except enrich.InvalidOutput as e:
        return None, f"invalid output: {e}"
    except Exception as e:
        return None, f"unexpected {type(e).__name__}: {e}"


# --- The stage ---------------------------------------------------------------


@dataclass(frozen=True)
class StoryOutcome:
    story_id: int
    titles: tuple[str, ...]
    ok: bool
    category: str | None
    error: str | None


@dataclass
class ClusteringSummary:
    in_window: int  # articles in the window
    not_enriched: int  # ...that have no enrichment for this version yet
    spam_excluded: int  # ...that are Spam
    candidates: int  # ...that were clustered
    run_id: int | None = None  # None if there was nothing to cluster or grouping failed
    new_run: bool = False  # False if an identical run already existed
    grouping_error: str | None = None
    outcomes: list[StoryOutcome] = field(default_factory=list)  # combined summaries attempted

    @property
    def succeeded(self) -> list[StoryOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[StoryOutcome]:
        return [o for o in self.outcomes if not o.ok]


def _input_fingerprint(candidates: list[sqlite3.Row]) -> str:
    # Enrichment rows never change, so their ids identify the exact input.
    ids = sorted(r["enrichment_id"] for r in candidates)
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def _store_run(conn, key, window_start, now, candidates, groups) -> int:
    run_id = db.create_story_run(
        conn, **key, window_start=window_start, window_end=now, created_at=now
    )
    grouped = {i for group in groups for i in group.article_ids}
    stories = [(min(g.article_ids), list(g.article_ids), None, None, g.reason) for g in groups]
    stories += [
        (r["article_id"], [r["article_id"]], r["category"], r["summary"], None)
        for r in candidates
        if r["article_id"] not in grouped
    ]  # a single-article story copies its article's category and summary
    for _, article_ids, category, summary, reason in sorted(stories, key=lambda s: s[0]):
        db.create_story(
            conn, run_id, article_ids=article_ids, category=category, summary=summary,
            grouping_reason=reason,
        )  # fmt: skip
    conn.commit()
    return run_id


def run_clustering(
    conn: sqlite3.Connection,
    llm: StructuredLLM,
    *,
    enrichment_model: str,
    enrichment_prompt_version: str,
    prompt_version: str = STORY_PROMPT_VERSION,
    now: datetime | None = None,
    report: Callable[[StoryOutcome], None] | None = None,
) -> ClusteringSummary:
    """Cluster the window's articles into a story run, then complete its story summaries.

    `now` is passed in so the window can be tested exactly. `report` is called after each
    combined summary, so a long run can show progress.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    window_start = now - ingest.MAX_ARTICLE_AGE
    rows = db.story_window_articles(
        conn,
        window_start=window_start,
        enrichment_model=enrichment_model,
        enrichment_prompt_version=enrichment_prompt_version,
    )
    enriched = [r for r in rows if r["enrichment_id"] is not None]
    candidates = [r for r in enriched if r["category"] != SPAM]
    summary = ClusteringSummary(
        in_window=len(rows),
        not_enriched=len(rows) - len(enriched),
        spam_excluded=len(enriched) - len(candidates),
        candidates=len(candidates),
    )
    if not candidates:
        return summary

    key = {
        "model": llm.model,
        "prompt_version": prompt_version,
        "enrichment_model": enrichment_model,
        "enrichment_prompt_version": enrichment_prompt_version,
        "input_hash": _input_fingerprint(candidates),
    }
    run = db.find_story_run(conn, **key)
    if run is None:
        valid_ids = {r["article_id"] for r in candidates}
        groups, error = generate_validated(
            llm,
            instructions=GROUPING_INSTRUCTIONS,
            input_text=build_grouping_input(candidates),
            schema_name=GROUPING_SCHEMA_NAME,
            schema=GROUPING_SCHEMA,
            parse=lambda raw: parse_groups(raw, valid_ids),
        )
        if error:
            summary.grouping_error = error  # nothing is stored; the next run tries again
            return summary
        summary.run_id = _store_run(conn, key, window_start, now, candidates, groups)
        summary.new_run = True
    else:
        summary.run_id = run["id"]

    titles = db.story_article_titles(conn, summary.run_id)
    for story in db.stories_needing_summary(conn, summary.run_id):
        members = db.story_member_articles(
            conn,
            story["id"],
            enrichment_model=enrichment_model,
            enrichment_prompt_version=enrichment_prompt_version,
        )
        result, error = generate_validated(
            llm,
            instructions=COMBINE_INSTRUCTIONS,
            input_text=build_combine_input(members),
            schema_name=COMBINE_SCHEMA_NAME,
            schema=COMBINE_SCHEMA,
            parse=parse_story_summary,
        )
        if result is not None:
            db.mark_story_ready(conn, story["id"], category=result.category, summary=result.summary)
        else:
            db.mark_story_failed(conn, story["id"], error=error)
        conn.commit()

        outcome = StoryOutcome(
            story_id=story["id"],
            titles=tuple(title for _, title in titles.get(story["id"], [])),
            ok=result is not None,
            category=None if result is None else result.category,
            error=error,
        )
        summary.outcomes.append(outcome)
        if report:
            report(outcome)

    return summary
