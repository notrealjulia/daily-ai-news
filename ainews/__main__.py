"""Command line entry point: `python -m ainews ingest`."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from ainews import db, ingest


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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
