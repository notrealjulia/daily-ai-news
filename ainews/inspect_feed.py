"""Onboarding tool for a new feed: `python -m ainews inspect-feed URL`.

feeds.toml is configuration; this is how you decide what goes in it. Given a feed
URL it gathers the evidence for choosing between two ways of getting article text:

    feed_content   use the text the feed itself carries
    fulltext       fetch the article page and extract its main text (Trafilatura)

It reports evidence only. It never reads or edits feeds.toml and never recommends a
strategy. Its "observations" are deterministic hints (say, the headline repeated on
line 1) meant to help you spot cleanup rules. Failures are reported as findings and
never worked around.
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser

from ainews import ingest
from ainews.extract import extract_article_text, fetch_page, normalize_text, paragraphs

MAX_SAMPLES = 5
RSS_TEXT_FULL_LIMIT = 1200  # show a feed's whole text up to this many characters

# Boilerplate hints: a short paragraph near the start or end of the extracted text
# that contains one of these. Long paragraphs are ignored, because real article text
# mentions "related work" or "comments" all the time.
BOILERPLATE_KEYWORDS = (
    "subscribe", "newsletter", "sign up", "recent articles", "related", "read more",
    "share this", "follow us", "advertisement", "all rights reserved", "©", "cookie",
    "comments",
)  # fmt: skip
BOILERPLATE_MAX_CHARS = 250
BOILERPLATE_HEAD_WINDOW = 3
BOILERPLATE_TAIL_WINDOW = 15

TRUNCATION_HINT = re.compile(
    r"(\.\.\.|…|\[…\]|\[\.\.\.\]|read more|continue reading|appeared first on[^\n]*)\W*$",
    re.IGNORECASE,
)
_MONTH = (
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
)
_YEAR = r"\b(?:19|20)\d\d\b"
_DATE_LINE_MAX_CHARS = 45


# --- Feed-level facts -------------------------------------------------------


@dataclass(frozen=True)
class FeedOverview:
    title: str | None
    format: str  # feedparser's label, e.g. "rss20" or "atom10"
    entries: int
    dated_entries: int
    dateless: bool  # how ingestion would treat it: no entry has a usable date
    oldest: datetime | None
    newest: datetime | None
    out_of_order_pairs: int  # adjacent dated entries not in newest-first order
    in_window: int  # entries ingestion would keep right now
    no_link: int
    common_time_of_day: tuple[str, float] | None  # ("12:00:00", share of dated entries)
    with_content_field: int  # entries carrying a full-content element
    with_summary_only: int
    with_no_text: int
    text_chars: tuple[int, ...]  # stripped-text length of each entry that has any
    truncation_hints: int  # entries whose text ends like a teaser ("…", "Read more")


def entry_text_field(entry: feedparser.FeedParserDict) -> str:
    """Which element ingest.entry_content would take text from. Keep in step with it."""
    contents = entry.get("content")
    if contents and contents[0].get("value"):
        return "content"
    return "summary" if entry.get("summary") else "none"


def summarize_feed(
    parsed: feedparser.FeedParserDict,
    now: datetime,
    max_age: timedelta = ingest.MAX_ARTICLE_AGE,
) -> FeedOverview:
    entries = list(parsed.entries)
    dates = [ingest.entry_published_at(e) for e in entries]
    dated = [d for d in dates if d is not None]
    texts = [ingest.entry_content(e) or "" for e in entries]
    fields = [entry_text_field(e) if t else "none" for e, t in zip(entries, texts)]

    common = None
    if dated:
        time_of_day, count = Counter(d.strftime("%H:%M:%S") for d in dated).most_common(1)[0]
        common = (time_of_day, count / len(dated))

    return FeedOverview(
        title=parsed.feed.get("title"),
        format=parsed.get("version") or "unknown",
        entries=len(entries),
        dated_entries=len(dated),
        dateless=bool(entries) and not dated,
        oldest=min(dated, default=None),
        newest=max(dated, default=None),
        out_of_order_pairs=sum(1 for a, b in zip(dated, dated[1:]) if a < b),
        in_window=sum(1 for d in dated if d >= now - max_age),
        no_link=sum(1 for e in entries if not ingest.entry_url(e)),
        common_time_of_day=common,
        with_content_field=fields.count("content"),
        with_summary_only=fields.count("summary"),
        with_no_text=fields.count("none"),
        text_chars=tuple(len(t) for t in texts if t),
        truncation_hints=sum(1 for t in texts if t and TRUNCATION_HINT.search(t)),
    )


def render_overview(o: FeedOverview, now: datetime) -> list[str]:
    lines = [
        f"title:          {o.title}",
        f"format:         {o.format}",
        f"entries:        {o.entries} ({o.no_link} without a link)",
    ]
    if o.newest:
        age_hours = (now - o.newest).total_seconds() / 3600
        lines.append(f"dates:          {o.dated_entries} of {o.entries} entries have a usable date")
        lines.append(
            f"date range:     {o.oldest:%Y-%m-%d %H:%M}Z .. {o.newest:%Y-%m-%d %H:%M}Z "
            f"(newest is {age_hours:.1f}h old)"
        )
        lines.append(
            f"order:          {o.out_of_order_pairs} of {max(o.dated_entries - 1, 0)} adjacent "
            "pairs are not newest-first"
        )
        if o.common_time_of_day:
            time_of_day, share = o.common_time_of_day
            lines.append(
                f"time of day:    {share:.0%} of dated entries are stamped {time_of_day}Z "
                "(a high share can mean the publisher only supplies a date)"
            )
    else:
        lines.append("dates:          no entry has a usable date")
    hours = ingest.MAX_ARTICLE_AGE.total_seconds() / 3600
    treatment = "dateless (newest-first walk)" if o.dateless else "dated (time window)"
    lines.append(f"ingestion:      would be treated as {treatment}")
    if not o.dateless:
        lines.append(f"in window:      {o.in_window} entries within the last {hours:g}h")

    lines.append(
        f"text in feed:   {o.with_content_field} full-content, {o.with_summary_only} "
        f"summary-only, {o.with_no_text} with no text"
    )
    if o.text_chars:
        chars = o.text_chars
        lines.append(
            f"text length:    min {min(chars):,} / median {int(statistics.median(chars)):,} / "
            f"max {max(chars):,} chars (over the {len(chars)} entries that have text)"
        )
        lines.append(f"teaser hints:   {o.truncation_hints} entries end with '…', 'Read more', etc.")
    return lines


# --- Choosing what to look at ------------------------------------------------


@dataclass(frozen=True)
class Sample:
    entry: feedparser.FeedParserDict
    position: int  # 1-based position in the feed
    published: datetime | None


def pick_samples(parsed: feedparser.FeedParserDict, count: int) -> tuple[list[Sample], bool]:
    """The `count` newest entries that have a link, and whether we had to guess.

    Newest means latest date. If no entry has a date we take entries from the top of
    the feed instead (assuming newest-first) and return True as the second value.
    """
    candidates = [
        Sample(entry, position, ingest.entry_published_at(entry))
        for position, entry in enumerate(parsed.entries, start=1)
        if ingest.entry_url(entry)
    ]
    dated = [s for s in candidates if s.published is not None]
    if not dated:
        return candidates[:count], True
    dated.sort(key=lambda s: (s.published, -s.position), reverse=True)
    return dated[:count], False


# --- Reading the extracted text ---------------------------------------------


def _looks_like_date_line(line: str) -> bool:
    if len(line) > _DATE_LINE_MAX_CHARS:
        return False
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", line):
        return True
    return bool(re.search(_MONTH, line, re.IGNORECASE) and re.search(_YEAR, line))


@dataclass(frozen=True)
class ExtractionObservations:
    paragraph_count: int
    title_repeated: bool  # paragraph 1 is the headline
    date_lines: tuple[tuple[int, str], ...]  # (paragraph number, text) near the start
    boilerplate: tuple[tuple[int, str, str], ...]  # (paragraph number, keyword, text)


def observe_extraction(text: str, title: str) -> ExtractionObservations:
    paras = paragraphs(text)
    total = len(paras)

    date_lines = tuple(
        (number, para) for number, para in enumerate(paras[:3], start=1) if _looks_like_date_line(para)
    )

    boilerplate = []
    window = set(range(1, BOILERPLATE_HEAD_WINDOW + 1)) | set(
        range(max(total - BOILERPLATE_TAIL_WINDOW + 1, 1), total + 1)
    )
    for number in sorted(window & set(range(1, total + 1))):
        para = paras[number - 1]
        if len(para) > BOILERPLATE_MAX_CHARS:
            continue
        lowered = para.lower()
        keyword = next((k for k in BOILERPLATE_KEYWORDS if k in lowered), None)
        if keyword:
            boilerplate.append((number, keyword, para))

    return ExtractionObservations(
        paragraph_count=total,
        title_repeated=bool(paras) and normalize_text(paras[0]) == normalize_text(title),
        date_lines=date_lines,
        boilerplate=tuple(boilerplate),
    )


def describe_observations(obs: ExtractionObservations, width: int) -> list[str]:
    def cut(s: str) -> str:
        return s if len(s) <= width else s[:width] + "…"

    lines = []
    if obs.title_repeated:
        lines.append("headline repeated as paragraph 1")
    for number, para in obs.date_lines:
        lines.append(f"date-like line at paragraph {number}: {cut(para)!r}")
    for number, keyword, para in obs.boilerplate:
        lines.append(f"possible boilerplate at paragraph {number} (matched {keyword!r}): {cut(para)!r}")
    return lines


def outline(text: str, head: int, tail: int, width: int) -> list[str]:
    """The first `head` and last `tail` paragraphs, numbered, each cut to `width` chars."""
    paras = paragraphs(text)

    def fmt(number: int) -> str:
        para = paras[number - 1]
        return f"[{number}/{len(paras)}] {para if len(para) <= width else para[:width] + '…'}"

    if len(paras) <= head + tail:
        return [fmt(n) for n in range(1, len(paras) + 1)]
    omitted = len(paras) - head - tail
    return (
        [fmt(n) for n in range(1, head + 1)]
        + [f"... {omitted} paragraphs omitted ..."]
        + [fmt(n) for n in range(len(paras) - tail + 1, len(paras) + 1)]
    )


def _indented(lines: list[str]) -> str:
    return "\n".join("    | " + line for line in lines)


# --- The command ------------------------------------------------------------


@dataclass(frozen=True)
class SampleResult:
    number: int
    rss_chars: int
    extracted_chars: int | None
    result: str  # "ok" or what went wrong
    observation_count: int


def _inspect_sample(
    number: int,
    total: int,
    sample: Sample,
    *,
    head: int,
    tail: int,
    width: int,
    save_dir: Path | None,
) -> SampleResult:
    entry = sample.entry
    url = ingest.entry_url(entry)
    title = (entry.get("title") or "").strip()
    rss_text = ingest.entry_content(entry) or ""

    print(f"SAMPLE {number}/{total}")
    print(f"  title:       {title}")
    print(f"  url:         {url}")
    published = f"{sample.published:%Y-%m-%dT%H:%M:%SZ}" if sample.published else "(no date)"
    print(f"  published:   {published}   (position {sample.position} in the feed)")
    print(f"  feed text:   {rss_text and entry_text_field(entry) or 'none'}, {len(rss_text):,} chars")
    if not rss_text:
        print("    | (the feed carries no text for this entry)")
    elif len(rss_text) <= RSS_TEXT_FULL_LIMIT:
        print(_indented(rss_text.splitlines()))
    else:
        print(_indented(rss_text[:500].splitlines()))
        print(f"    | ... [{len(rss_text) - 800:,} chars omitted] ...")
        print(_indented(rss_text[-300:].splitlines()))

    page = fetch_page(url)
    print(f"  page fetch:  {page.note}")
    text, problem = "", page.problem
    if page.html is not None:
        text, problem = extract_article_text(page.html, url)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"sample{number}.feed.txt").write_text(rss_text, encoding="utf-8")
        if text:
            (save_dir / f"sample{number}.extracted.txt").write_text(text, encoding="utf-8")

    if problem:
        print(f"  extraction:  FAILED - {problem}\n")
        return SampleResult(number, len(rss_text), None, f"FAILED: {problem}", 0)

    ratio = len(text) / len(rss_text) if rss_text else None
    print(
        f"  extraction:  {len(text):,} chars"
        + (f" ({ratio:.1f}x the feed text)" if ratio else " (feed has no text to compare)")
    )
    observations = describe_observations(observe_extraction(text, title), width)
    if rss_text and len(text) <= len(rss_text):
        observations.append("extraction is not longer than the feed text (it may have missed the article)")
    print("  observations (deterministic hints, not verdicts):")
    for line in observations or ["(none found)"]:
        print(f"    - {line}")
    print(f"  extracted text, first {head} and last {tail} paragraphs:")
    print(_indented(outline(text, head, tail, width)))
    print()
    return SampleResult(number, len(rss_text), len(text), "ok", len(observations))


def run(
    url: str,
    *,
    samples: int = 1,
    head: int = 4,
    tail: int = 8,
    width: int = 200,
    save_dir: Path | None = None,
    now: datetime | None = None,
) -> int:
    """Inspect a feed and print the evidence. Returns 1 only if the feed itself is unusable."""
    now = now or datetime.now(timezone.utc)
    print(f"Inspecting {url}\n")

    try:
        parsed = ingest.fetch_feed(url)
    except ingest.FeedError as e:
        print(f"FAILED to load the feed: {e}")
        return 1
    if not parsed.entries:
        print("The feed loaded but contains no entries.")
        return 1

    print("FEED")
    for line in render_overview(summarize_feed(parsed, now), now):
        print(f"  {line}")
    print()

    picked, guessed = pick_samples(parsed, samples)
    if not picked:
        print("No entry in this feed has a link, so there is nothing to inspect.")
        return 1
    if guessed:
        print("NOTE: no entry has a usable date, so the samples below are the top of the feed")
        print("      (assuming newest-first), not verified newest articles.\n")

    results = [
        _inspect_sample(n, len(picked), sample, head=head, tail=tail, width=width, save_dir=save_dir)
        for n, sample in enumerate(picked, start=1)
    ]

    print("=" * 72)
    print("EVIDENCE SUMMARY (no strategy is chosen here)")
    print(f"{'sample':>6} {'feed chars':>11} {'extracted':>10}  {'observations':>12}  result")
    for r in results:
        extracted = "-" if r.extracted_chars is None else f"{r.extracted_chars:,}"
        print(f"{r.number:>6} {r.rss_chars:>11,} {extracted:>10}  {r.observation_count:>12}  {r.result}")
    print(
        "\nCompare the feed text with the extracted text above and read the start and end of\n"
        "the extraction for cleanup markers. Then decide feed_content or fulltext yourself."
    )
    return 0
