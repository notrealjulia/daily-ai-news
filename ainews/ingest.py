"""RSS ingestion: fetch feeds, normalize entries, store recent ones.

Deterministic: no LLM involved. Given the same feed and the same clock,
it always makes the same decisions, which is what makes it testable.
"""

import calendar
import sqlite3
import tomllib
import urllib.request
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import feedparser

from ainews import db

# Only entries published within this long before "now" are ingested.
MAX_ARTICLE_AGE = timedelta(hours=24)

DEFAULT_FEEDS_PATH = Path("feeds.toml")
USER_AGENT = "ainews/0.1 (personal RSS reader)"
FETCH_TIMEOUT_SECONDS = 20


class FeedError(Exception):
    """A feed could not be fetched or parsed."""


# How a feed's article text is acquired (see ainews.extract).
STRATEGIES = ("feed_content", "fulltext")


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    strategy: str  # one of STRATEGIES
    # fulltext only: extracted text is cut at the first paragraph equal to one of these
    # (a site's trailing subscription block, say).
    stop_markers: tuple[str, ...] = ()


@dataclass
class FeedStats:
    """What happened to each entry in a feed.

    Every entry is counted exactly once at each level:
        found                = in_window + too_old + bad_date + undated
        in_window + undated  = inserted + already_present + no_url + not_examined

    A feed is either dated (some entry has a usable date) or dateless (none does).
    In a dated feed, entries are in_window, too_old, or bad_date (their own date
    is missing). In a dateless feed, every entry is `undated` and is handled by
    the newest-first rule described in _ingest_dateless_feed.
    """

    found: int = 0
    in_window: int = 0
    too_old: int = 0
    bad_date: int = 0  # missing/unparseable date, in a feed that has other dated entries
    undated: int = 0  # entries of a dateless feed
    no_url: int = 0  # would be ingested, but has no link to dedup on
    inserted: int = 0
    already_present: int = 0
    not_examined: int = 0  # dateless feed: older than the first known URL, so never looked at

    def __add__(self, other: "FeedStats") -> "FeedStats":
        return FeedStats(
            **{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        )


# --- Config -----------------------------------------------------------------


def load_feeds(path: str | Path = DEFAULT_FEEDS_PATH) -> list[Feed]:
    """Read and validate the [[feeds]] tables from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    feeds = []
    for i, item in enumerate(data.get("feeds", []), start=1):
        try:
            name, url, strategy = item["name"], item["url"], item["strategy"]
        except KeyError as missing:
            raise ValueError(f"{path}: feed #{i} is missing {missing}") from None
        if strategy not in STRATEGIES:
            raise ValueError(
                f"{path}: feed {name!r} has strategy {strategy!r}; "
                f"expected one of: {', '.join(STRATEGIES)}"
            )
        stop_markers = item.get("stop_markers", [])
        if not isinstance(stop_markers, list) or not all(isinstance(m, str) for m in stop_markers):
            raise ValueError(f"{path}: feed {name!r}: stop_markers must be a list of strings")
        if stop_markers and strategy != "fulltext":
            raise ValueError(f"{path}: feed {name!r}: stop_markers only apply to strategy 'fulltext'")
        if any(feed.name == name for feed in feeds):
            raise ValueError(f"{path}: more than one feed is named {name!r}")
        feeds.append(Feed(name, url, strategy, tuple(stop_markers)))
    return feeds


# --- Fetching ---------------------------------------------------------------


def fetch_feed(url: str) -> feedparser.FeedParserDict:
    """Download and parse a feed.

    We do the HTTP ourselves (rather than letting feedparser do it) so that we
    control the timeout and User-Agent and get clear errors.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            body = response.read()
    except OSError as e:  # URLError, HTTPError and timeouts are all OSErrors
        raise FeedError(f"{url}: {e}") from e

    parsed = feedparser.parse(body)
    # feedparser leaves `version` empty when the response isn't recognisably RSS or
    # Atom, for example an ordinary web page. That is a failure, not a feed that
    # happens to have zero entries (which has a version and simply no entries).
    if not parsed.get("version"):
        raise FeedError(
            f"{url}: not an RSS/Atom feed (the response was not recognised as one; "
            "it may be an ordinary web page)"
        )
    # "bozo" means malformed XML. That's fine if we still got entries out of it,
    # but if we got nothing the feed is broken.
    if parsed.bozo and not parsed.entries:
        raise FeedError(f"{url}: not a parseable feed ({parsed.bozo_exception})")
    return parsed


# --- Normalizing entries ----------------------------------------------------


def entry_published_at(entry: feedparser.FeedParserDict) -> datetime | None:
    """The entry's publication time as an aware UTC datetime, or None.

    feedparser has already converted whatever the feed said (+0200, EST, Z...)
    to a UTC struct_time, and leaves it as None if the date was missing,
    unparseable, or an RSS date with no timezone.

    We fall back to `updated` because some feeds (Atom without <published>,
    RSS using dc:date) only provide that.
    """
    # dict.get reads the stored keys directly. FeedParserDict.get() would also
    # apply feedparser's deprecated updated->published aliasing, which would
    # make this fallback depend on behavior that's being removed.
    parsed = dict.get(entry, "published_parsed") or dict.get(entry, "updated_parsed")
    if parsed is None:
        return None
    try:
        # timegm reads the tuple as UTC. (time.mktime would read it as the
        # machine's local time, which is wrong on any non-UTC computer.)
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


class _TextExtractor(HTMLParser):
    _SKIPPED = {"script", "style"}
    _BLOCKS = {
        "p", "br", "div", "li", "ul", "ol", "tr", "blockquote", "pre",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }  # fmt: skip

    def __init__(self) -> None:
        super().__init__()  # convert_charrefs=True: &amp; etc. are decoded for us
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, _attrs: list) -> None:
        if tag in self._SKIPPED:
            self._skip_depth += 1
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def strip_html(html: str) -> str:
    """Reduce HTML to plain text, keeping paragraph-ish line breaks."""
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    lines = (" ".join(line.split()) for line in "".join(extractor.parts).splitlines())
    return "\n".join(line for line in lines if line)


def entry_url(entry: feedparser.FeedParserDict) -> str:
    """The entry's link, or "" if it has none. This is the dedup key."""
    return (entry.get("link") or "").strip()


def entry_content(entry: feedparser.FeedParserDict) -> str | None:
    """Best available text for the entry: full content if given, else the summary."""
    contents = entry.get("content")
    raw = (contents[0].get("value") if contents else None) or entry.get("summary")
    text = strip_html(raw) if raw else ""
    return text or None


# --- Ingesting --------------------------------------------------------------


def _store_entry(
    conn: sqlite3.Connection,
    source: str,
    entry: feedparser.FeedParserDict,
    url: str,
    *,
    published_at: datetime | None,
    now: datetime,
) -> bool:
    """Insert one entry. Returns False if its URL was already stored."""
    return db.insert_article(
        conn,
        source=source,
        url=url,
        title=(entry.get("title") or "").strip() or "(untitled)",
        published_at=published_at,
        fetched_at=now,
        feed_text=entry_content(entry),
    )


def ingest_parsed_feed(
    conn: sqlite3.Connection,
    source: str,
    parsed: feedparser.FeedParserDict,
    *,
    now: datetime,
    max_age: timedelta = MAX_ARTICLE_AGE,
) -> FeedStats:
    """Store the new entries of an already-parsed feed and report what happened.

    A feed where at least one entry has a usable date is "dated": entries are
    kept if published within `max_age` of `now`. A feed where no entry has one is
    "dateless" and is handled by the newest-first rule instead.

    `now` is passed in (rather than read from the clock here) so the window
    boundary can be tested exactly. It is also recorded as fetched_at.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)

    entries = list(parsed.entries)
    dated = [(entry, entry_published_at(entry)) for entry in entries]
    stats = FeedStats(found=len(entries))

    if entries and all(published_at is None for _, published_at in dated):
        _ingest_dateless_feed(conn, source, entries, now, stats)
    else:
        _ingest_dated_feed(conn, source, dated, now, now - max_age, stats)

    conn.commit()
    return stats


def _ingest_dated_feed(
    conn: sqlite3.Connection,
    source: str,
    dated_entries: list[tuple[feedparser.FeedParserDict, datetime | None]],
    now: datetime,
    cutoff: datetime,
    stats: FeedStats,
) -> None:
    for entry, published_at in dated_entries:
        if published_at is None:
            stats.bad_date += 1
            continue
        if published_at < cutoff:  # exactly at the cutoff counts as in the window
            stats.too_old += 1
            continue
        stats.in_window += 1

        url = entry_url(entry)
        if not url:
            stats.no_url += 1
            continue

        if _store_entry(conn, source, entry, url, published_at=published_at, now=now):
            stats.inserted += 1
        else:
            stats.already_present += 1


def _ingest_dateless_feed(
    conn: sqlite3.Connection,
    source: str,
    entries: list[feedparser.FeedParserDict],
    now: datetime,
    stats: FeedStats,
) -> None:
    """Ingest what is new in a feed that gives no dates, using position instead.

    Assumes the feed is ordered newest to oldest. Walk down from the top:
    every unseen URL is new since our last run, and the first URL already in the
    database marks where the previous run stopped, so everything below it is
    older and is not looked at.

    If no URL in the feed is in the database, we've never processed it (or none
    of what we took survived). Ingest only the newest entry so the first run
    doesn't import the whole backlog.

    Articles are stored with published_at NULL and fetched_at = now.
    """
    stats.undated = len(entries)

    unseen = []
    reached_known_url = False
    for position, entry in enumerate(entries):
        url = entry_url(entry)
        if not url:
            stats.no_url += 1
            continue
        if db.url_exists(conn, url):
            stats.already_present += 1
            stats.not_examined = len(entries) - position - 1
            reached_known_url = True
            break
        unseen.append((entry, url))

    if not reached_known_url:  # first run for this feed: newest entry only
        stats.not_examined = max(len(unseen) - 1, 0)
        unseen = unseen[:1]

    # Insert oldest first, so the newest entry is written last. If we were ever
    # interrupted part-way, the top of the feed would still be unseen and the
    # next run would fill in the rest. Newest-first would instead leave a gap
    # below a "known" top entry that no later run would ever revisit.
    for entry, url in reversed(unseen):
        if _store_entry(conn, source, entry, url, published_at=None, now=now):
            stats.inserted += 1
        else:  # same URL listed twice in one feed
            stats.already_present += 1


def ingest_feed(
    conn: sqlite3.Connection,
    feed: Feed,
    *,
    now: datetime | None = None,
    max_age: timedelta = MAX_ARTICLE_AGE,
) -> FeedStats:
    """Fetch one feed over the network and ingest it."""
    return ingest_parsed_feed(
        conn,
        feed.name,
        fetch_feed(feed.url),
        now=now or datetime.now(timezone.utc),
        max_age=max_age,
    )
