"""Command line entry point: `python -m ainews ingest | extract | enrich | narrate | inspect-feed`."""

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ainews import db, ingest

if TYPE_CHECKING:  # imported lazily below: extract needs Trafilatura, llm needs the OpenAI SDK
    from ainews import digest, enrich, extract, narrate, stories


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

    # The same window run_extraction itself will use, so this banner matches what the
    # run below actually does.
    window_start = datetime.now(timezone.utc) - ingest.MAX_ARTICLE_AGE
    waiting_rows = db.articles_needing_body(conn, window_start)
    new = sum(1 for r in waiting_rows if r["body_status"] == "pending")
    already_complete = db.body_status_counts(conn)["ready"]
    print(
        f"{already_complete} article(s) already complete; {len(waiting_rows)} waiting "
        f"({new} new, {len(waiting_rows) - new} retrying after an earlier failure)\n"
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

    # The same window run_enrichment itself will use, so this banner matches the run below.
    window_start = datetime.now(timezone.utc) - ingest.MAX_ARTICLE_AGE
    overview = db.enrichment_overview(conn, client.model, enrich.PROMPT_VERSION, window_start)
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


def cmd_cluster(args: argparse.Namespace) -> int:
    from ainews import enrich, llm, stories

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        client = llm.create(args.model or llm.DEFAULT_MODEL)
    except llm.LLMConfigError as e:
        print(f"Cannot cluster: {e}")
        return 2
    conn = db.connect(args.db)

    now = datetime.now(timezone.utc)
    hours = ingest.MAX_ARTICLE_AGE.total_seconds() / 3600
    cutoff = now - ingest.MAX_ARTICLE_AGE
    print(f"Model: {client.model}   story prompt: {stories.STORY_PROMPT_VERSION}")
    print(f"Reads enrichments: {llm.DEFAULT_MODEL} / {enrich.PROMPT_VERSION}")
    print(f"Window: last {hours:g}h (published, or first seen, at or after {cutoff:%Y-%m-%dT%H:%M:%SZ})\n")

    summary = stories.run_clustering(
        conn,
        client,
        enrichment_model=llm.DEFAULT_MODEL,
        enrichment_prompt_version=enrich.PROMPT_VERSION,
        now=now,
        report=lambda outcome: print(_format_story_outcome(outcome), flush=True),
    )

    if summary.run_id is not None:
        print()
        for line in _format_story_listing(conn, summary.run_id):
            print(line)
    print()
    for line in _format_clustering_summary(conn, summary):
        print(line)
    return 1 if summary.grouping_error or summary.failed else 0


def _format_story_outcome(o: "stories.StoryOutcome") -> str:
    label = f"story {o.story_id} ({len(o.titles)} articles)"
    if o.ok:
        return f"  {'ok':<7} {label}: combined summary written, category {o.category}"
    return f"  {'FAILED':<7} {label}: {o.error}"


def _format_story_listing(conn: sqlite3.Connection, run_id: int) -> list[str]:
    titles = db.story_article_titles(conn, run_id)
    lines = []
    for story in db.stories_in_run(conn, run_id):
        members = titles[story["id"]]
        label = story["category"] if story["summary_status"] == "ready" else f"summary {story['summary_status']}"
        if len(members) == 1:
            source, title = members[0]
            lines.append(f"  Story {story['id']} [{label}]  {source}: {title}")
            continue
        lines.append(f"  Story {story['id']} [{label}]  {len(members)} articles")
        lines += [f"      {source}: {title}" for source, title in members]
        if story["summary"]:
            lines.append(f"      Summary: {story['summary']}")
        lines.append(f"      Grouped because: {story['grouping_reason']}")
    return lines


def _format_clustering_summary(conn: sqlite3.Connection, s: "stories.ClusteringSummary") -> list[str]:
    lines = [
        "SUMMARY",
        f"  articles in the window:   {s.in_window:>4}",
        f"  excluded as Spam:         {s.spam_excluded:>4}",
    ]
    if s.not_enriched:
        lines.append(
            f"  not enriched yet:         {s.not_enriched:>4}  (run `python -m ainews enrich`)"
        )
    if not s.candidates:
        lines.append("  Nothing to cluster: no enriched, non-Spam articles in the window.")
        return lines
    if s.grouping_error:
        lines.append(f"  Grouping failed: {s.grouping_error}")
        lines.append("  Nothing was stored; run `python -m ainews cluster` again.")
        return lines

    run_stories = db.stories_in_run(conn, s.run_id)
    several = sum(1 for story in run_stories if story["article_count"] > 1)
    lines.append(
        f"  clustered:                {s.candidates:>4} articles into {len(run_stories)} stories "
        f"({several} with several articles)"
    )
    lines.append(
        f"  story run:                #{s.run_id}"
        + (" (new)" if s.new_run else " (already existed, so not regrouped)")
    )
    lines.append(f"  combined summaries:       ok {len(s.succeeded)}, failed {len(s.failed)}")
    if s.failed:
        lines.append("")
        lines.append("Retryable failures (the stories stay; run `python -m ainews cluster` again):")
        for reason, count in Counter(o.error for o in s.failed).most_common():
            lines.append(f"  {count} x {reason}")
    return lines


def cmd_digest(args: argparse.Namespace) -> int:
    from ainews import digest, llm

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        client = llm.create(args.model or llm.DEFAULT_MODEL)
    except llm.LLMConfigError as e:
        print(f"Cannot write digests: {e}")
        return 2
    conn = db.connect(args.db)

    run = db.get_story_run(conn, args.run)
    if run is None:
        print(
            f"No story run with id {args.run}."
            if args.run is not None
            else "No story run to work from. Run `python -m ainews cluster` first."
        )
        return 1
    print(f"Model: {client.model}   digest prompt: {digest.DIGEST_PROMPT_VERSION}")
    print(f"Story run #{run['id']} (window ending {run['window_end']})\n")

    summary = digest.run_digests(
        conn,
        client,
        run_id=run["id"],
        report=lambda outcome: print(_format_digest_outcome(outcome), flush=True),
    )
    if summary.incomplete_stories:
        print(
            f"Refusing to write digests: {summary.incomplete_stories} of {summary.total_stories} "
            f"stories in run #{run['id']} have no ready summary.\n"
            "Run `python -m ainews cluster` again to finish them first."
        )
        return 1

    print()
    for line in _format_digest_summary(summary):
        print(line)
    return 1 if summary.failed else 0


def cmd_narrate(args: argparse.Namespace) -> int:
    from ainews import llm, narrate

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        client = llm.create(args.model or llm.DEFAULT_MODEL)  # writes each script
        tts = llm.create_tts()  # turns each script into audio
    except llm.LLMConfigError as e:
        print(f"Cannot narrate: {e}")
        return 2
    conn = db.connect(args.db)

    summary = narrate.narrate_all_categories(
        conn, client, tts, report=lambda outcome: print(_format_narration_outcome(outcome), flush=True)
    )
    if summary.run_id is None:
        print("No fully processed run to narrate. Run `python -m ainews digest` first.")
        return 1

    print()
    for line in _format_narration_summary(summary):
        print(line)
    return 1 if summary.failed else 0


def _format_narration_outcome(o: "narrate.NarrationOutcome") -> str:
    if not o.ok:
        return f"  {'FAILED':<7} {o.category:<20} - {o.error}"
    words = len(o.script.split())
    line = f"  {'ok':<7} {o.category:<20} {words:>3} words"
    # Never stored as text anywhere else, so printing it here is the only way to review it.
    return f"{line}\n      {o.script}\n"


def _format_narration_summary(s: "narrate.NarrationSummary") -> list[str]:
    lines = [
        "SUMMARY",
        f"  narrated this run:          {len(s.succeeded):>4}",
        f"  failed this run:            {len(s.failed):>4}",
        f"  categories with no stories: {len(s.no_stories):>4}",
    ]
    if s.failed:
        lines.append("")
        lines.append(
            "Retryable failures (existing audio, if any, was left untouched; "
            "run `python -m ainews narrate` again):"
        )
        for reason, count in Counter(o.error for o in s.failed).most_common():
            lines.append(f"  {count} x {reason}")
    return lines


def _format_digest_outcome(o: "digest.DigestOutcome") -> str:
    stories = f"{o.story_count} stor{'y' if o.story_count == 1 else 'ies'}"
    if o.ok:
        return f"  {'ok':<7} {o.category:<20} {stories}"
    return f"  {'FAILED':<7} {o.category:<20} {stories}  -  {o.error}"


def _format_digest_summary(s: "digest.DigestSummary") -> list[str]:
    lines = [
        "SUMMARY",
        f"  stories in the run:         {s.total_stories:>4}",
        f"  digests written this run:   {len(s.succeeded):>4}",
        f"  failed this run:            {len(s.failed):>4}",
        f"  already had a digest:       {len(s.already_done):>4}  (skipped)",
        f"  categories with no stories: {len(s.no_stories):>4}",
    ]
    if s.failed:
        lines.append("")
        lines.append("Retryable failures (nothing was stored; run `python -m ainews digest` again):")
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

    cluster_parser = subcommands.add_parser(
        "cluster",
        help="group the last 24h of enriched articles into stories",
        description="Group enriched, non-Spam articles from the 24h window into stories, "
        "keeping articles separate unless they clearly describe the same event. The result "
        "is an immutable snapshot (a story run); repeating it with unchanged input finds "
        "the existing run, and failed story summaries are retried.",
    )
    cluster_parser.add_argument("--db", type=Path, default=db.DEFAULT_DB_PATH)
    cluster_parser.add_argument(
        "--model", default=None, help="OpenAI model id (default: the one set in ainews/llm.py)"
    )
    cluster_parser.set_defaults(func=cmd_cluster)

    digest_parser = subcommands.add_parser(
        "digest",
        help="write one digest per category from a story run",
        description="Write a short digest for each category of a finished story run (the "
        "latest by default), from its stories. Refuses to run if the story run is incomplete.",
    )
    digest_parser.add_argument("--db", type=Path, default=db.DEFAULT_DB_PATH)
    digest_parser.add_argument(
        "--model", default=None, help="OpenAI model id (default: the one set in ainews/llm.py)"
    )
    digest_parser.add_argument("--run", type=int, default=None, help="story run id (default: latest)")
    digest_parser.set_defaults(func=cmd_digest)

    narrate_parser = subcommands.add_parser(
        "narrate",
        help="turn today's stories into a spoken briefing per category, audio/<category>.mp3",
        description="For every non-Spam category with stories today: write a short spoken "
        "briefing script (title + summary + source article text, not the digest), persist "
        "it, then read it aloud with OpenAI TTS and overwrite that category's fixed MP3 "
        "file (e.g. audio/research.mp3, audio/product-release.mp3). One category's failure "
        "never stops the others. Each script is printed here for review, but never shown "
        "in the dashboard.",
    )
    narrate_parser.add_argument("--db", type=Path, default=db.DEFAULT_DB_PATH)
    narrate_parser.add_argument(
        "--model", default=None, help="OpenAI model id for the scripts (default: the one set in ainews/llm.py)"
    )
    narrate_parser.set_defaults(func=cmd_narrate)

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
