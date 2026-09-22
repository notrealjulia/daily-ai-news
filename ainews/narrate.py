"""Audio narration: a short spoken briefing per category, from today's stories, via OpenAI TTS.

MVP scope, deliberately small:
    - Every non-Spam category that has stories today gets its own narration: title,
      summary, and each source article's full extracted body, not the category digest,
      which is a written summary and reads badly aloud. The article text is what lets
      the script identify things the summary alone doesn't name clearly (which
      institution, which company, where).
    - Two LLM calls per category: a structured call writes the spoken script from that
      category's stories, then TTS turns that exact script into audio. The script is
      persisted (`narrations`, versioned like digests: run, category, model,
      prompt_version) before TTS ever runs, so what is in the database and what is in
      the audio always agree. It is never displayed anywhere (not in Streamlit).
    - Unlike digests, there is no dedup: every narrate run writes a new row and makes a
      fresh script, since narrate is a manual, on-demand action meant to be rerun for a
      new take, not skipped because a narration already exists for today.
    - One fixed local file per category (see defaults.audio_path), e.g.
      "audio/research.mp3", "audio/product-release.mp3". Every run overwrites them; the
      audio itself is never stored in the database, only the script that made it.
    - A category with no stories is skipped, not a failure. A failed category (either
      call) leaves its existing audio file exactly as it was, and never stops the
      others: one category's trouble is isolated the same way one article's or one
      story's is elsewhere in the pipeline.

This module knows nothing about any provider; see ainews.llm for that.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ainews import db, enrich
from ainews.defaults import DEFAULT_MODEL, audio_path
from ainews.llm import LLMError, StructuredLLM, TextToSpeech
from ainews.prompts import DIGEST_PROMPT_VERSION, NARRATE_INSTRUCTIONS, NARRATE_PROMPT_VERSION
from ainews.stories import NON_SPAM_CATEGORIES, generate_validated

SCHEMA_NAME = "category_narration"

SCHEMA = {
    "type": "object",
    "properties": {"script": {"type": "string"}},
    "required": ["script"],
    "additionalProperties": False,
}

# The prompt asks for about 200-250 words (roughly 1,300-1,700 characters); this only
# catches an answer that has run well away from that, not the exact target.
MAX_SCRIPT_CHARS = 3000

# A day's stories in one category can together pull in a lot of source text; this bounds
# each article's contribution so the request stays a sane size regardless of how many
# stories there are. Smaller than enrich.MAX_BODY_CHARS, which budgets for one article
# per call; this one's shared across every article of every story in the same request.
MAX_ARTICLE_CONTEXT_CHARS = 4000


@dataclass(frozen=True)
class SourceArticle:
    source: str
    body: str  # the extracted article text (see extract.py); the summary alone can omit context


@dataclass(frozen=True)
class CategoryStory:
    title: str  # the earliest article's title, in English - see db.story_article_links
    summary: str
    articles: tuple[SourceArticle, ...] = ()


def current_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """The newest fully processed story run - the same one the dashboard and digests use."""
    return db.latest_dashboard_run(
        conn, digest_model=DEFAULT_MODEL, digest_prompt_version=DIGEST_PROMPT_VERSION
    )


def category_stories_in_run(conn: sqlite3.Connection, run_id: int, category: str) -> list[CategoryStory]:
    """This run's stories in one category (title, summary, source articles), newest
    first: the same stories and titles the dashboard shows for that category."""
    links_by_story: dict[int, list[sqlite3.Row]] = {}
    for link in db.story_article_links(conn, run_id):
        links_by_story.setdefault(link["story_id"], []).append(link)
    articles_by_story: dict[int, list[SourceArticle]] = {}
    for row in db.story_article_bodies(conn, run_id):
        article = SourceArticle(source=row["source"], body=row["body"])
        articles_by_story.setdefault(row["story_id"], []).append(article)

    found = []
    for story in db.stories_in_run(conn, run_id):
        if story["category"] != category:
            continue
        links = links_by_story.get(story["id"], [])
        if not links:
            continue
        newest = max(link["happened_at"] for link in links)
        view = CategoryStory(
            title=links[0]["title"],  # earliest article first
            summary=story["summary"],
            articles=tuple(articles_by_story.get(story["id"], [])),
        )
        found.append((newest, story["id"], view))

    found.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in found]


def build_script_input(category: str, stories: list[CategoryStory]) -> str:
    lines = [f"Category: {category}", "", "<stories>"]
    for number, story in enumerate(stories, start=1):
        title = story.title.replace("</stories>", "")  # a story must not close its own tag
        summary = story.summary.replace("</stories>", "")
        lines.append(f"[{number}] {title}\n{summary}")
        for article in story.articles:
            source = article.source.replace("</stories>", "")
            body = article.body.replace("</stories>", "")
            if len(body) > MAX_ARTICLE_CONTEXT_CHARS:
                body = body[:MAX_ARTICLE_CONTEXT_CHARS] + " [article text truncated]"
            lines.append(f"Source article ({source}): {body}")
    lines.append("</stories>")
    return "\n".join(lines)


def parse_script(raw: dict) -> str:
    """Validate the script answer strictly; nothing is corrected or guessed."""
    if set(raw) != {"script"}:
        raise enrich.InvalidOutput(f"expected exactly script, got {sorted(raw)}")
    script = raw["script"]
    if not isinstance(script, str) or not script.strip():
        raise enrich.InvalidOutput("the script is empty")
    script = script.strip()
    if len(script) > MAX_SCRIPT_CHARS:
        raise enrich.InvalidOutput(f"the script is {len(script):,} chars (limit {MAX_SCRIPT_CHARS:,})")
    return script


def generate_script(llm: StructuredLLM, category: str, stories: list[CategoryStory]) -> tuple[str | None, str | None]:
    """(script, None) on success, or (None, reason) on failure."""
    return generate_validated(
        llm,
        instructions=NARRATE_INSTRUCTIONS,
        input_text=build_script_input(category, stories),
        schema_name=SCHEMA_NAME,
        schema=SCHEMA,
        parse=parse_script,
    )


def narrate_category(
    conn: sqlite3.Connection,
    llm: StructuredLLM,
    tts: TextToSpeech,
    category: str,
    stories: list[CategoryStory],
    *,
    run_id: int,
    path: Path | None = None,
    now: datetime | None = None,
) -> tuple[bool, str | None, str | None]:
    """Narrate one category's stories, which the caller has already fetched (so
    narrate_all_categories fetches each category's stories only once). (ok, script,
    error): `script` is the generated text whenever one was successfully written, even
    if the later TTS call then failed - it is also, by then, already in the
    `narrations` table, the only place it is kept; it is never shown in Streamlit. On
    success, `path` (default: defaults.audio_path(category)) now holds the new
    narration (any prior file is replaced); on any failure, `path` is left untouched."""
    script, error = generate_script(llm, category, stories)
    if script is None:
        return False, None, f"could not write a narration script: {error}"

    db.insert_narration(
        conn,
        run_id=run_id,
        category=category,
        script=script,
        model=llm.model,
        prompt_version=NARRATE_PROMPT_VERSION,
        created_at=now or datetime.now(timezone.utc),
    )
    conn.commit()  # persisted before TTS runs, so the record survives even if TTS fails

    try:
        audio = tts.synthesize(script)  # exactly the script just persisted
    except LLMError as e:
        return False, script, str(e)
    out = path or audio_path(category)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio)
    return True, script, None


@dataclass(frozen=True)
class NarrationOutcome:
    category: str
    ok: bool
    script: str | None
    error: str | None


@dataclass
class NarrationSummary:
    run_id: int | None  # None if there is no fully processed run to work from
    no_stories: list[str] = field(default_factory=list)  # categories with nothing to narrate
    outcomes: list[NarrationOutcome] = field(default_factory=list)

    @property
    def succeeded(self) -> list[NarrationOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[NarrationOutcome]:
        return [o for o in self.outcomes if not o.ok]


def narrate_all_categories(
    conn: sqlite3.Connection,
    llm: StructuredLLM,
    tts: TextToSpeech,
    *,
    now: datetime | None = None,
    report: Callable[[NarrationOutcome], None] | None = None,
) -> NarrationSummary:
    """Narrate every non-Spam category that has stories in the current run, each into
    its own fixed audio/<category>.mp3. One category's failure - no stories, an
    unexpected error, a failed script call or a failed TTS call - never stops the
    others; `report` is called after each attempted category, so a long run can show
    progress."""
    run = current_run(conn)
    if run is None:
        return NarrationSummary(run_id=None)

    summary = NarrationSummary(run_id=run["id"])
    for category in NON_SPAM_CATEGORIES:
        try:
            stories = category_stories_in_run(conn, run["id"], category)
            if not stories:
                summary.no_stories.append(category)
                continue
            ok, script, error = narrate_category(conn, llm, tts, category, stories, run_id=run["id"], now=now)
        except Exception as e:  # one category must never stop the rest
            ok, script, error = False, None, f"unexpected {type(e).__name__}: {e}"
        outcome = NarrationOutcome(category, ok, script, error)
        summary.outcomes.append(outcome)
        if report:
            report(outcome)
    return summary
