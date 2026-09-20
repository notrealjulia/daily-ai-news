"""Tests for the inspect-feed onboarding command.

Pure functions are tested directly. The command itself is run against a small local
HTTP server, so redirects, blocked pages and the full report are exercised without
touching the internet.
"""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import feedparser
import pytest
from helpers import article_page, ok

from ainews import ingest, inspect_feed

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)


def ago(**delta) -> str:
    """An RSS date string for a moment `delta` before NOW."""
    return format_datetime(NOW - timedelta(**delta))


def feed_xml(*items: dict) -> str:
    """RSS with one <item> per dict. Keys: title, link, date, summary, content."""
    body = ""
    for item in items:
        body += f"<item><title>{item.get('title', 't')}</title>"
        if "link" in item:
            body += f"<link>{item['link']}</link>"
        if "date" in item:
            body += f"<pubDate>{item['date']}</pubDate>"
        if "summary" in item:
            body += f"<description>{item['summary']}</description>"
        if "content" in item:
            body += f"<content:encoded><![CDATA[{item['content']}]]></content:encoded>"
        body += "</item>"
    return (
        "<rss version='2.0' xmlns:content='http://purl.org/rss/1.0/modules/content/'>"
        f"<channel><title>Test Feed</title>{body}</channel></rss>"
    )


def parse(*items: dict) -> feedparser.FeedParserDict:
    return feedparser.parse(feed_xml(*items))


# --- picking samples ---------------------------------------------------------


def test_samples_are_the_newest_by_date_not_by_feed_position():
    parsed = parse(
        {"title": "older", "link": "http://x/older", "date": ago(hours=5)},
        {"title": "newest", "link": "http://x/newest", "date": ago(hours=1)},
        {"title": "middle", "link": "http://x/middle", "date": ago(hours=3)},
    )

    samples, guessed = inspect_feed.pick_samples(parsed, 2)

    assert [s.entry.title for s in samples] == ["newest", "middle"]
    assert [s.position for s in samples] == [2, 3]
    assert guessed is False


def test_dateless_feed_samples_come_from_the_top_and_are_flagged():
    parsed = parse(
        {"title": "A", "link": "http://x/A"},
        {"title": "B", "link": "http://x/B"},
        {"title": "C", "link": "http://x/C"},
    )

    samples, guessed = inspect_feed.pick_samples(parsed, 2)

    assert [s.entry.title for s in samples] == ["A", "B"]
    assert guessed is True


# --- feed-wide facts ---------------------------------------------------------


def test_feed_overview_counts():
    parsed = parse(
        {"title": "a", "link": "http://x/a", "date": ago(hours=1), "content": "<p>" + "full " * 40 + "</p>"},
        {"title": "b", "link": "http://x/b", "date": ago(hours=2), "summary": "a teaser that trails off…"},
        {"title": "c", "link": "http://x/c", "date": ago(hours=30)},
        {"title": "d", "link": "http://x/d", "summary": "no date on this one"},
        {"title": "e", "date": ago(hours=3), "summary": "and no link on this one"},
    )

    o = inspect_feed.summarize_feed(parsed, NOW)

    assert o.entries == 5 and o.dated_entries == 4
    assert o.dateless is False
    assert o.newest == NOW - timedelta(hours=1)
    assert o.oldest == NOW - timedelta(hours=30)
    assert o.in_window == 3  # a, b, e; c is 30h old
    assert o.no_link == 1
    assert (o.with_content_field, o.with_summary_only, o.with_no_text) == (1, 3, 1)
    assert len(o.text_chars) == 4
    assert o.truncation_hints == 1  # only b ends with an ellipsis
    assert o.out_of_order_pairs == 1  # c (30h old) is followed by e (3h old)
    assert o.common_time_of_day is not None and o.common_time_of_day[1] == pytest.approx(0.25)


# --- reading extracted text --------------------------------------------------

LONG = "A long body paragraph that keeps going so it is clearly article text. " * 6


@pytest.mark.parametrize(
    ("text", "expected"),
    [(f"My Headline\n{LONG}", True), (f"{LONG}\n{LONG}", False)],
    ids=["repeated-ignoring-case", "not-repeated"],
)
def test_headline_repeated_on_line_one_is_noticed(text, expected):
    assert inspect_feed.observe_extraction(text, "my headline").title_repeated is expected


@pytest.mark.parametrize(
    ("line", "is_date"),
    [
        ("Updated September 17, 2026", True),
        ("19th September 2026", True),
        ("We shipped it in May.", False),
        ("Market outlook 2026", False),  # 'Mar' must not match 'Market'
    ],
)
def test_date_like_lines_near_the_start_are_recognised_and_ordinary_ones_are_not(line, is_date):
    obs = inspect_feed.observe_extraction(f"Title\n{line}\n{LONG}", "Title")

    assert obs.date_lines == (((2, line),) if is_date else ())


def test_boilerplate_at_the_end_is_reported_with_its_paragraph_number():
    body = "\n".join([LONG] * 5)
    text = f"{body}\nAI News Without the Hype\nSubscribe to our newsletter\nSubscribe now"

    obs = inspect_feed.observe_extraction(text, "Title")

    assert [(n, kw) for n, kw, _ in obs.boilerplate] == [(7, "subscribe"), (8, "subscribe")]
    assert obs.paragraph_count == 8


def test_keywords_in_long_paragraphs_or_mid_article_are_not_boilerplate():
    # a long paragraph that merely mentions a keyword
    text = f"{LONG}\n{LONG} This is related work, and the comments are discussed.\n{LONG}"
    assert inspect_feed.observe_extraction(text, "Title").boilerplate == ()

    # a short keyword paragraph far from either end of a long article
    paragraphs = [LONG] * 60
    paragraphs[30] = "Subscribe now"
    assert inspect_feed.observe_extraction("\n".join(paragraphs), "Title").boilerplate == ()


def test_outline_of_a_long_text_shows_both_ends_and_says_what_was_skipped():
    text = "\n".join(f"paragraph {n}" for n in range(1, 21))

    lines = inspect_feed.outline(text, head=2, tail=3, width=50)

    assert lines == [
        "[1/20] paragraph 1",
        "[2/20] paragraph 2",
        "... 15 paragraphs omitted ...",
        "[18/20] paragraph 18",
        "[19/20] paragraph 19",
        "[20/20] paragraph 20",
    ]


# --- the whole command -------------------------------------------------------


def two_article_site(routes, base):
    routes["/feed.xml"] = ok(
        feed_xml(
            {"title": "Older post", "link": f"{base}/older", "date": ago(hours=6), "summary": "old teaser"},
            {"title": "Newest post", "link": f"{base}/newest", "date": ago(hours=1), "summary": "new teaser"},
        ),
        "application/rss+xml",
    )
    routes["/newest"] = ok(article_page("Newest post", "NEWEST"))
    routes["/older"] = ok(article_page("Older post", "OLDER"))


def test_report_contains_the_evidence_and_decides_nothing(server, capsys, monkeypatch, tmp_path):
    base, routes = server
    two_article_site(routes, base)
    monkeypatch.chdir(tmp_path)
    # The command must not consult feeds.toml at all.
    monkeypatch.setattr(ingest, "load_feeds", lambda *a, **k: pytest.fail("read feeds.toml"))

    code = inspect_feed.run(base + "/feed.xml", now=NOW)
    out = capsys.readouterr().out

    assert code == 0
    assert "Test Feed" in out and "would be treated as dated" in out
    assert "Newest post" in out and "Older post" not in out  # one sample by default: the newest
    assert "new teaser" in out  # the feed's own text
    assert "extraction:" in out and "NEWEST paragraph 1" in out  # the start of the extracted text
    assert "NEWEST paragraph 6" in out  # ...and the end
    assert "EVIDENCE SUMMARY" in out and "no strategy is chosen" in out
    assert "recommend" not in out.lower()
    assert list(tmp_path.iterdir()) == []  # wrote nothing


def test_a_blocked_page_is_reported_as_a_finding_not_hidden(server, capsys):
    base, routes = server
    two_article_site(routes, base)
    routes["/newest"] = (403, {"cf-mitigated": "challenge"}, b"challenge")

    code = inspect_feed.run(base + "/feed.xml", now=NOW)
    out = capsys.readouterr().out

    assert code == 0  # the inspection worked; the page is simply the finding
    assert "extraction:  FAILED" in out and "Cloudflare" in out
    assert "FAILED: HTTP 403" in out.split("EVIDENCE SUMMARY")[1]
    assert "NEWEST paragraph" not in out  # no silent fallback to anything


def test_an_extraction_no_longer_than_the_feed_text_is_flagged(server, capsys):
    base, routes = server
    routes["/feed.xml"] = ok(
        feed_xml({"title": "T", "link": f"{base}/p", "date": ago(hours=1), "summary": "a fairly long teaser " * 5})
    )
    routes["/p"] = ok("<html><body><p>hi</p></body></html>")  # Trafilatura returns just "hi"

    inspect_feed.run(base + "/feed.xml", now=NOW)

    assert "extraction is not longer than the feed text" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("path", "message"),
    [("/nothing-here", "404"), ("/site", "not an RSS/Atom feed")],
    ids=["unreachable-feed", "web-page-that-is-not-a-feed"],
)
def test_a_feed_that_cannot_be_used_fails_clearly(server, capsys, path, message):
    base, routes = server
    routes["/site"] = ok("<html><body><h1>Just a website</h1><p>No feed here.</p></body></html>")

    code = inspect_feed.run(base + path, now=NOW)
    out = capsys.readouterr().out

    assert code == 1
    assert "FAILED to load the feed" in out and message in out


