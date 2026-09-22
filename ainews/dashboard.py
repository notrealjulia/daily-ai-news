"""What the read-only dashboard shows, worked out from SQLite (and the source names in feeds.toml).

`app.py` (Streamlit) only renders what this module returns. There is no SQL here (see
db.py) and no Streamlit, so the logic can be tested without a browser. This module
imports nothing from the pipeline or the LLM code: only `db`, and the import-free
settings in `defaults.py` (the default model, and where `narrate` writes each category's
narration) and `prompts.py` (the current digest prompt version). That is what makes the
dashboard incapable of running a stage or calling OpenAI, and a test enforces it.

The dashboard shows the newest story run that is fully processed (every story summarized
and every category digested with the current digest prompt and default model). A run
that is only partly done is never shown.
"""

import re
import sqlite3
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from urllib.parse import quote, urlparse

from ainews import db
from ainews.defaults import DEFAULT_MODEL, audio_path
from ainews.prompts import DIGEST_PROMPT_VERSION

# The six categories shown, in the 2-column grid's reading order (row by row).
# Spam is deliberately absent: it is never displayed.
CATEGORY_GRID = (
    ("Product Release", "Industry News"),
    ("Research", "Business"),
    ("Regulation & Policy", "Other"),
)
CATEGORY_ORDER = tuple(name for row in CATEGORY_GRID for name in row)


@dataclass(frozen=True)
class Source:
    name: str
    url: str | None  # None if the article's URL isn't a plain http(s) link


@dataclass(frozen=True)
class StoryView:
    title: str  # the earliest article's title, in English (stories have no title of their own yet)
    summary: str
    sources: tuple[Source, ...]  # one per article, earliest first


@dataclass(frozen=True)
class CategoryView:
    name: str
    headline: str | None  # None for a digest written before headlines existed
    digest: str | None
    stories: tuple[StoryView, ...]  # newest first; not ranked

    @property
    def story_count(self) -> int:
        return len(self.stories)


@dataclass(frozen=True)
class Dashboard:
    last_updated: datetime  # timezone-aware UTC
    window_hours: int
    total_stories: int
    total_articles: int
    categories: dict[str, CategoryView]  # keyed by name, in CATEGORY_ORDER


# --- Loading -----------------------------------------------------------------


def _parse_timestamp(text: str) -> datetime:
    return datetime.strptime(text, db.TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def _safe_url(url: str) -> str | None:
    return url if urlparse(url).scheme in ("http", "https") else None


def load_dashboard(conn: sqlite3.Connection) -> Dashboard | None:
    """The dashboard for the newest fully processed story run, or None if there isn't one."""
    run = db.latest_dashboard_run(
        conn, digest_model=DEFAULT_MODEL, digest_prompt_version=DIGEST_PROMPT_VERSION
    )
    if run is None:
        return None

    links_by_story: dict[int, list[sqlite3.Row]] = {}
    for link in db.story_article_links(conn, run["id"]):
        links_by_story.setdefault(link["story_id"], []).append(link)
    digest_rows = db.digests_for_run(
        conn, run["id"], model=DEFAULT_MODEL, prompt_version=DIGEST_PROMPT_VERSION
    )
    digests = {row["category"]: row for row in digest_rows}

    grouped: dict[str, list[tuple[str, int, StoryView]]] = {name: [] for name in CATEGORY_ORDER}
    for story in db.stories_in_run(conn, run["id"]):
        links = links_by_story.get(story["id"], [])
        if story["category"] not in grouped or not links:
            continue  # Spam, or anything outside the six categories, is never shown
        view = StoryView(
            title=links[0]["title"],  # earliest article first
            summary=story["summary"],
            sources=tuple(Source(link["source"], _safe_url(link["url"])) for link in links),
        )
        newest = max(link["happened_at"] for link in links)
        grouped[story["category"]].append((newest, story["id"], view))

    categories = {}
    for name in CATEGORY_ORDER:
        newest_first = sorted(grouped[name], key=lambda item: (item[0], item[1]), reverse=True)
        digest = digests.get(name)
        categories[name] = CategoryView(
            name=name,
            headline=digest["headline"] if digest else None,
            digest=digest["summary"] if digest else None,
            stories=tuple(item[2] for item in newest_first),
        )

    shown = [story for category in categories.values() for story in category.stories]
    # "Last updated" is the latest stored timestamp behind what is on screen.
    stamps = [_parse_timestamp(run["created_at"])] + [
        _parse_timestamp(row["created_at"]) for row in digest_rows
    ]
    window = _parse_timestamp(run["window_end"]) - _parse_timestamp(run["window_start"])
    return Dashboard(
        last_updated=max(stamps),
        window_hours=int(window.total_seconds() // 3600),
        total_stories=len(shown),
        total_articles=sum(len(story.sources) for story in shown),
        categories=categories,
    )


def open_dashboard(
    path: str | Path = db.DEFAULT_DB_PATH, settings: Mapping[str, str] | None = None
) -> Dashboard | None:
    """Open the database read-only and load the dashboard.

    `settings` (Streamlit's secrets, when deployed) is passed on to db.connect_readonly.

    Returns None if there is nothing to show: no database file, a database from before
    clustering existed (no story tables), or no fully processed run. It never writes to,
    creates, or upgrades the database.
    """
    try:
        conn = db.connect_readonly(path, settings)
    except sqlite3.OperationalError:
        return None
    try:
        return load_dashboard(conn)
    except sqlite3.OperationalError:  # e.g. "no such table: story_runs"
        return None
    finally:
        conn.close()


# --- Text for the page -------------------------------------------------------

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]<>$#~&])")


def escape_markdown(text: str) -> str:
    """Make text from the web safe to show as markdown.

    Titles, summaries and digests are untrusted text. Escaping stops markdown from
    reformatting them (and `$...$` from being typeset as math, which matters for
    "$2 trillion" and "$100 billion"); HTML is never enabled, so tags stay plain text.
    Text is shown as one line.
    """
    text = _MARKDOWN_SPECIAL.sub(r"\\\1", " ".join(text.split()))
    text = re.sub(r"^([-+])", r"\\\1", text)  # a leading "- " would start a list
    return re.sub(r"^(\d+)([.)])", r"\1\\\2", text)  # ...and so would "1. "


def _source_markdown(source: Source) -> str:
    name = escape_markdown(source.name)
    if source.url is None:
        return name
    # Percent-encode anything that could end the link early, such as ")" or spaces.
    return f"[{name}]({quote(source.url, safe=':/?#[]@!$&' + chr(39) + '*+,;=%~-._')})"


def story_markdown(story: StoryView) -> str:
    """One story: title, summary, and its sources, each linked to its article."""
    sources = " · ".join(_source_markdown(source) for source in story.sources)
    return f"**{escape_markdown(story.title)}**  \n{escape_markdown(story.summary)}  \nSources: {sources}"


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def expander_label(story_count: int) -> str:
    return f"View {_plural(story_count, 'story', 'stories')}"


def database_label(settings: Mapping[str, str] | None = None) -> str:
    """Which database the page is reading, for the message shown when there is no run."""
    return "Turso" if db.uses_turso(settings) else "the local SQLite file"


def empty_text(window_hours: int) -> str:
    return f"No stories in the last {window_hours} hours."


def sources_caption(path: str | Path = "feeds.toml") -> str | None:
    """e.g. "Sources monitored: OpenAI · Google DeepMind", or None if there are none to show.

    The names come straight from feeds.toml, the source of truth, in file order. It is
    read here rather than through ingest.load_feeds, which would import the RSS parser.
    A missing or unreadable file just means no footer.
    """
    try:
        with open(path, "rb") as f:
            feeds = tomllib.load(f).get("feeds", [])
    except (OSError, tomllib.TOMLDecodeError):
        return None
    names = [escape_markdown(feed["name"]) for feed in feeds if "name" in feed]
    return f"Sources monitored: {' · '.join(names)}" if names else None


def category_audio_path(category: str) -> Path | None:
    """A category's narration file (see ainews.narrate), or None if it hasn't been made yet.

    One fixed path per category, checked for existence only; never generated here.
    """
    path = audio_path(category)
    return path if path.is_file() else None


def format_header(dashboard: Dashboard, tz: tzinfo | None = None) -> str:
    """e.g. "Last updated: Sep 20, 18:00 · Last 24 hours · 8 stories from 9 articles".

    The time is shown in `tz`, by default the local time of the machine running the app.
    """
    local = dashboard.last_updated.astimezone(tz)
    return (
        f"Last updated: {local:%b} {local.day}, {local:%H:%M} · Last {dashboard.window_hours} hours · "
        f"{_plural(dashboard.total_stories, 'story', 'stories')} from "
        f"{_plural(dashboard.total_articles, 'article', 'articles')}"
    )
