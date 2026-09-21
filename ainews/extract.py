"""Content acquisition: give every stored article its `body`, the text later stages use.

Each feed in feeds.toml has a strategy:

    feed_content   the text the feed itself carries becomes the body (no network)
    fulltext       fetch the article page and extract its main text (Trafilatura)

`python -m ainews extract` handles every article whose body is not yet 'ready':
articles never tried (pending) and articles whose last attempt failed. A failure is
recorded and the article stays eligible, so the next run simply tries it again; there
is no backoff, attempt limit or queue. Articles that are 'ready' are never touched
again.

A failed fulltext extraction never falls back to the feed's teaser: the article just
has no body until an extraction succeeds. One article failing never stops the others.

This module also holds the page-fetching and extraction helpers, which the
`inspect-feed` tool reuses so that it evaluates exactly what this stage will do.
"""

import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import trafilatura

from ainews import db, ingest

MAX_PAGE_BYTES = 5_000_000


# --- Fetching and extracting a page -----------------------------------------


@dataclass(frozen=True)
class PageFetch:
    html: bytes | None
    note: str  # what happened, in a line
    problem: str | None  # set when the page could not be fetched


def fetch_page(url: str) -> PageFetch:
    """Fetch an article page, describing (never raising on) any failure."""
    request = urllib.request.Request(url, headers={"User-Agent": ingest.USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=ingest.FETCH_TIMEOUT_SECONDS) as response:
            body = response.read(MAX_PAGE_BYTES)
            status = getattr(response, "status", None)
            note = f"HTTP {status}" if status else "no HTTP status"
            note += f", {len(body):,} bytes, {response.headers.get_content_type()}"
            if response.url != url:
                note += f", redirected to {response.url}"
            return PageFetch(body, note, None)
    except urllib.error.HTTPError as e:
        problem = f"HTTP {e.code} {e.reason}"
        if e.headers is not None and e.headers.get("cf-mitigated") == "challenge":
            problem += " - Cloudflare bot challenge (often intermittent; not worked around)"
        elif e.code in (401, 403):
            problem += " - the server refuses this client (not worked around)"
        return PageFetch(None, problem, problem)
    except OSError as e:  # URLError, timeouts, connection failures
        return PageFetch(None, str(e), str(e))


def extract_article_text(html: bytes, url: str) -> tuple[str, str | None]:
    """(text, problem). Trafilatura, the same settings every time."""
    try:
        text = trafilatura.extract(html, url=url, include_comments=False)
    except Exception as e:  # third-party parser on arbitrary HTML: report, don't crash
        return "", f"Trafilatura raised {type(e).__name__}: {e}"
    text = (text or "").strip()
    return text, (None if text else "Trafilatura returned no text")


def paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n") if p.strip()]


def normalize_text(s: str) -> str:
    return " ".join(s.lower().split())


def clean_extracted_text(text: str, title: str, stop_markers: tuple[str, ...] = ()) -> str:
    """Deterministic cleanup of extracted text.

    Drops the headline when it is repeated as the first paragraph, and cuts the text at
    the first paragraph equal to one of `stop_markers` (that paragraph and everything
    after it). Comparison ignores case and whitespace.
    """
    paras = paragraphs(text)
    if paras and normalize_text(paras[0]) == normalize_text(title):
        paras = paras[1:]
    markers = {normalize_text(marker) for marker in stop_markers}
    for index, para in enumerate(paras):
        if normalize_text(para) in markers:
            paras = paras[:index]
            break
    return "\n".join(paras)


# --- Acquiring one article's body -------------------------------------------


def acquire_body(
    feed: ingest.Feed, *, url: str, title: str, feed_text: str | None
) -> tuple[str | None, str | None]:
    """(body, None) on success, or (None, reason) on failure."""
    if feed.strategy == "feed_content":
        if feed_text:
            return feed_text, None
        return None, "the feed carries no text for this entry"

    if feed.strategy == "fulltext":
        page = fetch_page(url)
        if page.html is None:
            return None, page.problem
        text, problem = extract_article_text(page.html, url)
        if problem:
            return None, problem
        body = clean_extracted_text(text, title, feed.stop_markers)
        return (body, None) if body else (None, "no text left after cleanup")

    raise ValueError(f"unknown strategy {feed.strategy!r} for feed {feed.name!r}")


# --- The stage ---------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    article_id: int
    source: str
    title: str
    strategy: str
    ok: bool
    chars: int | None  # length of the body, when ok
    error: str | None  # why it failed, when not ok
    recovered: bool  # succeeded now after failing on an earlier run


@dataclass
class ExtractionSummary:
    already_complete: int  # 'ready' before this run started; not touched
    outcomes: list[Outcome] = field(default_factory=list)
    # Waiting articles whose source is not in feeds.toml, so we don't know their strategy.
    not_processed: list[tuple[str, str]] = field(default_factory=list)  # (source, title)

    @property
    def succeeded(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.ok]

    @property
    def recovered(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.recovered]


def run_extraction(
    conn: sqlite3.Connection,
    feeds: list[ingest.Feed],
    *,
    now: datetime | None = None,
    report: Callable[[Outcome], None] | None = None,
) -> ExtractionSummary:
    """Acquire the body of every article that doesn't have one yet.

    `report` is called after each article, so a long run can show progress.
    Each result is committed as it happens, so an interruption loses nothing.
    """
    by_name = {feed.name: feed for feed in feeds}
    summary = ExtractionSummary(already_complete=db.body_status_counts(conn)["ready"])
    requested_from: set[str] = set()  # feeds whose site was already asked for a page this run

    for row in db.articles_needing_body(conn):
        feed = by_name.get(row["source"])
        if feed is None:
            summary.not_processed.append((row["source"], row["title"]))
            continue

        # A feed can ask for a pause between its page requests (request_delay_seconds).
        # This only spaces requests out; retrying is unchanged (a failure waits for the next run).
        if feed.strategy == "fulltext" and feed.request_delay_seconds:
            if feed.name in requested_from:
                time.sleep(feed.request_delay_seconds)
            requested_from.add(feed.name)

        try:
            body, error = acquire_body(
                feed, url=row["url"], title=row["title"], feed_text=row["feed_text"]
            )
        except Exception as e:  # one article must never stop the rest
            body, error = None, f"unexpected {type(e).__name__}: {e}"

        checked_at = now or datetime.now(timezone.utc)
        if body is not None:
            db.mark_body_ready(
                conn, row["id"], body=body, source=feed.strategy, checked_at=checked_at
            )
        else:
            db.mark_body_failed(conn, row["id"], error=error, checked_at=checked_at)
        conn.commit()

        outcome = Outcome(
            article_id=row["id"],
            source=row["source"],
            title=row["title"],
            strategy=feed.strategy,
            ok=body is not None,
            chars=None if body is None else len(body),
            error=error,
            recovered=body is not None and row["body_status"] == "failed",
        )
        summary.outcomes.append(outcome)
        if report:
            report(outcome)

    return summary
