"""Command line entry point: `python -m ainews ingest | extract | enrich | inspect-feed`."""

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ainews import db, ingest

if TYPE_CHECKING:  # imported lazily below: extract needs Trafilatura, llm needs the OpenAI SDK
    from ainews import enrich, extract


def cmd_ingest(args: argparse.Namespace) -> int:
    feeds = ingest.load_feeds(args.feeds)
    conn = db.connect(args.db)

    # One clock reading for the whole run, so every feed uses the same cutoff.
    now = datetime.now(timezone.utc)
    cutoff = now - ingest.MAX_ARTICLE_AGE
    hours = ingest.MAX_ARTICLE_AGE.total_seconds() / 3600
    print(f"Window: last {hours:g}h (published at or after {cutoff:%Y-%m-%dT%H:%M:%SZ})\n")

    width = max(len(feed.name) for feed in feeds)
    total = ingest.FeedStats()
    failures = 0
    for feed in feeds:
        try:
            stats = ingest.ingest_feed(conn, feed, now=now)
        except ingest.FeedError as e:
            failures += 1
            print(f"{feed.name:<{width}}  FAILED: {e}")
            continue
        total += stats
        print(f"{feed.name:<{width}}  {_format_stats(stats)}")

    print(f"\n{'TOTAL':<{width}}  {_format_stats(total)}")
    if failures:
        print(f"\n{failures} feed(s) failed.")
    return 1 if failures else 0


def cmd_inspect_feed(args: argparse.Namespace) -> int:
    # Imported here so that `ingest` runs without needing Trafilatura's dependencies.
    from ainews import inspect_feed

    # Extracted text has curly quotes etc.; don't let a Windows console codec crash us.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return inspect_feed.run(
        args.url,
        samples=args.samples,
        head=args.head,
        tail=args.tail,
        save_dir=args.save_dir,
    )


def cmd_extract(args: argparse.Namespace) -> int:
    # Imported here so that `ingest` runs without needing Trafilatura's dependencies.
    from ainews import extract

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    feeds = ingest.load_feeds(args.feeds)
    conn = db.connect(args.db)

    counts = db.body_status_counts(conn)
    waiting = counts["pending"] + counts["failed"]
    print(
        f"{counts['ready']} article(s) already complete; {waiting} waiting "
        f"({counts['pending']} new, {counts['failed']} retrying after an earlier failure)\n"
    )

    width = max((len(feed.name) for feed in feeds), default=0)
    summary = extract.run_extraction(
        conn, feeds, report=lambda outcome: print(_format_outcome(outcome, width), flush=True)
    )

    print()
    for line in _format_extraction_summary(summary):
        print(line)
    return 1 if summary.failed else 0


def _format_outcome(o: "extract.Outcome", width: int) -> str:
    title = o.title if len(o.title) <= 50 else o.title[:49] + "…"
    if o.ok:
        status = "ok" + ("*" if o.recovered else "")
        detail = f"{o.chars:,} chars"
    else:
        status, detail = "FAILED", o.error
    return f"  {status:<7} {o.strategy:<12} {o.source:<{width}}  {title}  -  {detail}"


def _format_extraction_summary(s: "extract.ExtractionSummary") -> list[str]:
    by_strategy = Counter(o.strategy for o in s.succeeded)
    breakdown = ", ".join(f"{name} {count}" for name, count in sorted(by_strategy.items()))
    extracted_note = f"({breakdown})" if breakdown else ""
    if s.recovered:
        extracted_note += f"  [* {len(s.recovered)} recovered from an earlier failure]"

    lines = [
        "SUMMARY",
        f"  already complete (skipped): {s.already_complete:>4}",
        f"  extracted this run:         {len(s.succeeded):>4}  {extracted_note}".rstrip(),
        f"  failed this run:            {len(s.failed):>4}",
        f"  not processed:              {len(s.not_processed):>4}"
        + ("  (source not in feeds.toml)" if s.not_processed else ""),
    ]
    if s.failed:
        lines.append("")
        lines.append("Retryable failures (they stay eligible; run `python -m ainews extract` again):")
        for reason, count in Counter(o.error for o in s.failed).most_common():
            lines.append(f"  {count} x {reason}")
    return lines


def cmd_enrich(args: argparse.Namespace) -> int:
    # Imported here so that `ingest` runs without needing the OpenAI SDK installed.
    from ainews import enrich, llm

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        client = llm.create(args.model or llm.DEFAULT_MODEL)
    except llm.LLMConfigError as e:
        print(f"Cannot enrich: {e}")
        return 2
    conn = db.connect(args.db)

    overview = db.enrichment_overview(conn, client.model, enrich.PROMPT_VERSION)
    waiting = overview["ready"] - overview["enriched"]
    print(f"Model: {client.model}   prompt_version: {enrich.PROMPT_VERSION}")
    print(
        f"{overview['enriched']} article(s) already enriched; {waiting} waiting"
        + (f" (processing at most {args.limit})" if args.limit is not None else "")
        + f"; {overview['not_ready']} without a ready body\n"
    )

    summary = enrich.run_enrichment(
        conn,
        client,
        limit=args.limit,
        report=lambda outcome: print(_format_enrich_outcome(outcome), flush=True),
    )

    print()
    for line in _format_enrichment_summary(summary):
        print(line)
    return 1 if summary.failed else 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _format_enrich_outcome(o: "enrich.Outcome") -> str:
    title = o.title if len(o.title) <= 50 else o.title[:49] + "…"
    if o.ok:
        return f"  {'ok':<7} {o.category:<19} {o.source[:16]:<16}  {title}"
    return f"  {'FAILED':<7} {o.source[:16]:<16}  {title}  -  {o.error}"


def _format_enrichment_summary(s: "enrich.EnrichmentSummary") -> list[str]:
    by_category = Counter(o.category for o in s.succeeded)
    breakdown = ", ".join(f"{name} {count}" for name, count in sorted(by_category.items()))
    lines = [
        "SUMMARY",
        f"  already enriched (skipped): {s.already_enriched:>4}",
        f"  enriched this run:          {len(s.succeeded):>4}" + (f"  ({breakdown})" if breakdown else ""),
        f"  failed this run:            {len(s.failed):>4}",
    ]
    if s.left_for_later:
        lines.append(f"  left for a later run:       {s.left_for_later:>4}  (--limit)")
    if s.not_ready:
        lines.append(
            f"  no ready body yet:          {s.not_ready:>4}  (run `python -m ainews extract`)"
        )
    if s.failed:
        lines.append("")
        lines.append("Retryable failures (nothing was stored; run `python -m ainews enrich` again):")
        for reason, count in Counter(o.error for o in s.failed).most_common():
            lines.append(f"  {count} x {reason}")
    return lines


def _format_stats(s: ingest.FeedStats) -> str:
    return (
        f"found {s.found:>4} | in window {s.in_window:>3} | too old {s.too_old:>4} | "
        f"bad date {s.bad_date:>3} | undated {s.undated:>4} | no url {s.no_url:>3} | "
        f"inserted {s.inserted:>3} | already present {s.already_present:>3} | "
        f"not examined {s.not_examined:>4}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ainews")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subcommands.add_parser("ingest", help="fetch feeds and store recent articles")
    ingest_parser.add_argument("--feeds", type=Path, default=ingest.DEFAULT_FEEDS_PATH)
    ingest_parser.add_argument("--db", type=Path, default=db.DEFAULT_DB_PATH)
    ingest_parser.set_defaults(func=cmd_ingest)

    extract_parser = subcommands.add_parser(
        "extract",
        help="acquire article text: fetch pages for fulltext feeds, use feed text otherwise",
        description="Give every article without a ready body its text, using each feed's "
        "strategy from feeds.toml. Failed articles stay eligible and are retried on the "
        "next run; articles that are already complete are skipped.",
    )
    extract_parser.add_argument("--feeds", type=Path, default=ingest.DEFAULT_FEEDS_PATH)
    extract_parser.add_argument("--db", type=Path, default=db.DEFAULT_DB_PATH)
    extract_parser.set_defaults(func=cmd_extract)

    enrich_parser = subcommands.add_parser(
        "enrich",
        help="classify and summarize articles that have a ready body, using an LLM",
        description="For each article with a ready body, ask the LLM for a category and a "
        "summary and store them. An article is enriched once per model and prompt version; "
        "failed articles store nothing and are retried on the next run.",
    )
    enrich_parser.add_argument("--db", type=Path, default=db.DEFAULT_DB_PATH)
    enrich_parser.add_argument(
        "--model", default=None, help="OpenAI model id (default: the one set in ainews/llm.py)"
    )
    enrich_parser.add_argument(
        "--limit", type=_positive_int, help="enrich at most this many articles this run"
    )
    enrich_parser.set_defaults(func=cmd_enrich)

    inspect_parser = subcommands.add_parser(
        "inspect-feed",
        help="report the evidence for choosing feed_content or fulltext for a feed URL",
        description="Inspect a feed's newest articles: compare the text in the feed with "
        "text extracted from the article page. Reports evidence only; edits nothing.",
    )
    inspect_parser.add_argument("url", help="RSS or Atom feed URL")
    inspect_parser.add_argument(
        "--samples", type=int, choices=range(1, 6), default=1, metavar="N",
        help="how many of the newest articles to examine, 1-5 (default 1)",
    )  # fmt: skip
    inspect_parser.add_argument("--head", type=int, default=4, help="paragraphs shown from the start")
    inspect_parser.add_argument("--tail", type=int, default=8, help="paragraphs shown from the end")
    inspect_parser.add_argument("--save-dir", type=Path, help="also write the full texts to this folder")
    inspect_parser.set_defaults(func=cmd_inspect_feed)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
