"""Tests for content acquisition (`python -m ainews extract`).

Everything runs offline against the local HTTP server fixture from conftest.py.
"""

import re
from datetime import datetime, timezone

import pytest
from helpers import article_page, ok

from ainews import db, extract, ingest
from ainews.__main__ import main

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
CLOUDFLARE = (403, {"cf-mitigated": "challenge"}, b"challenge")

FULLTEXT = ingest.Feed("Site", "http://unused/feed", "fulltext")
FEED_CONTENT = ingest.Feed("Notes", "http://unused/feed", "feed_content")


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def add(conn, url, *, source="Site", title="A title", feed_text=None) -> int:
    db.insert_article(
        conn, source=source, url=url, title=title,
        published_at=NOW, fetched_at=NOW, feed_text=feed_text,
    )  # fmt: skip
    conn.commit()
    return conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()["id"]


def article(conn, article_id):
    return conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()


def run(conn, *feeds, **kwargs):
    return extract.run_extraction(conn, list(feeds), now=NOW, **kwargs)


# --- cleanup ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "title", "stop_markers", "expected"),
    [
        pytest.param(
            "My Headline\nBody one\nBody two", "my headline", (), "Body one\nBody two",
            id="repeated-headline-dropped-ignoring-case",
        ),
        pytest.param(
            "Body one\nBody two", "My Headline", (), "Body one\nBody two",
            id="first-paragraph-kept-when-it-is-not-the-headline",
        ),
        pytest.param(
            "Body one\nBody two\nAI News Without the Hype\nSubscribe now\nFull archive",
            "Headline", ("AI News Without the Hype",), "Body one\nBody two",
            id="cut-at-a-stop-marker-and-the-marker-goes-too",
        ),
        pytest.param(
            "Body\n  PROMO   block ", "T", ("promo block",), "Body",
            id="stop-marker-ignores-case-and-spacing",
        ),
        pytest.param(
            "Headline\nBody\nPromo\nMore promo", "Headline", ("Promo",), "Body",
            id="headline-and-stop-marker-together",
        ),
        pytest.param(
            "Body mentions the Promo block in passing\nMore body", "T", ("Promo",),
            "Body mentions the Promo block in passing\nMore body",
            id="stop-marker-only-matches-a-whole-paragraph",
        ),
    ],
)  # fmt: skip
def test_extracted_text_is_cleaned(text, title, stop_markers, expected):
    assert extract.clean_extracted_text(text, title, stop_markers) == expected


# --- fetching a page -------------------------------------------------------


def test_fetch_returns_the_page_and_follows_redirects(server):
    base, routes = server
    routes["/old"] = (302, {"Location": "/new"}, b"")
    routes["/new"] = ok("<html>new</html>")

    page = extract.fetch_page(base + "/old")

    assert page.problem is None and page.html == b"<html>new</html>"
    assert "HTTP 200" in page.note and f"redirected to {base}/new" in page.note


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        # Intermittent in practice; must be reported (so it is retried), never worked around.
        (CLOUDFLARE, "Cloudflare bot challenge (often intermittent; not worked around)"),
        ((403, {}, b"no"), "refuses this client (not worked around)"),
        ((404, {}, b"missing"), "HTTP 404"),
    ],
    ids=["cloudflare-challenge", "plain-403", "404"],
)
def test_http_failures_are_described_not_raised(server, response, expected):
    base, routes = server
    routes["/page"] = response

    page = extract.fetch_page(base + "/page")

    assert page.html is None and expected in page.problem


def test_fetch_reports_an_unreachable_page_instead_of_raising(tmp_path):
    page = extract.fetch_page((tmp_path / "nope.html").as_uri())
    assert page.html is None and page.problem


# --- acquiring bodies ------------------------------------------------------


def test_fulltext_stores_the_cleaned_page_text_and_keeps_the_feed_text(server, conn):
    base, routes = server
    routes["/a"] = ok(article_page("A title", "ART"))
    a = add(conn, base + "/a", feed_text="just a teaser")

    summary = run(conn, FULLTEXT)

    row = article(conn, a)
    assert row["body_status"] == "ready" and row["body_source"] == "fulltext"
    assert row["body"].startswith("ART paragraph 1")  # the repeated headline is gone
    assert "ART paragraph 6" in row["body"]
    assert row["body_error"] is None
    assert row["body_checked_at"] == "2026-09-20T12:00:00Z"
    assert row["feed_text"] == "just a teaser"  # kept as evidence
    assert len(summary.succeeded) == 1 and summary.failed == []


def test_a_feed_with_a_request_delay_is_paused_between_its_page_requests_only(server, conn, monkeypatch):
    base, routes = server
    pauses = []
    monkeypatch.setattr(extract, "time", type("FakeTime", (), {"sleep": pauses.append}))
    slow = ingest.Feed("Slow", "http://unused/feed", "fulltext", request_delay_seconds=2.5)
    for path in ("/s1", "/s2", "/s3", "/o1", "/o2"):
        routes[path] = ok(article_page("T", path[1:].upper()))
    routes["/s2"] = CLOUDFLARE  # a refused request still counts as a request
    # Articles alternate between the delayed feed and another one (FULLTEXT has no delay).
    for path in ("/s1", "/o1", "/s2", "/o2", "/s3"):
        add(conn, base + path, source="Slow" if path.startswith("/s") else "Site", title="T")

    summary = run(conn, slow, FULLTEXT)

    assert pauses == [2.5, 2.5]  # three Slow requests -> two pauses; none for the other feed
    assert [o.ok for o in summary.outcomes] == [True, True, False, True, True]  # nothing else changed


def test_feed_content_uses_the_feed_text_and_never_touches_the_network(conn):
    # The URL points at a closed port: any attempt to fetch it would fail.
    a = add(conn, "http://127.0.0.1:1/never", source="Notes", feed_text="the whole post")

    run(conn, FEED_CONTENT)

    row = article(conn, a)
    assert (row["body_status"], row["body_source"], row["body"]) == (
        "ready", "feed_content", "the whole post",
    )  # fmt: skip


# Make the article's very first paragraph a stop marker, so nothing is left after cleanup.
FIRST_PARAGRAPH = (
    "ART paragraph 1: the quick brown fox jumps over the lazy dog while a group of "
    "researchers measure how consistently agents repeat their results across many "
    "independent runs of the same task."
)


@pytest.mark.parametrize(
    ("feed", "page", "message"),
    [
        pytest.param(FEED_CONTENT, None, "carries no text", id="feed-content-without-feed-text"),
        pytest.param(
            FULLTEXT, "<html><body></body></html>", "Trafilatura returned no text",
            id="page-with-nothing-extractable",
        ),
        pytest.param(
            ingest.Feed("Site", "http://unused/feed", "fulltext", (FIRST_PARAGRAPH,)),
            article_page("A title", "ART"), "no text left after cleanup",
            id="nothing-left-after-cleanup",
        ),
    ],
)  # fmt: skip
def test_articles_whose_text_cannot_be_acquired_are_marked_failed(server, conn, feed, page, message):
    base, routes = server
    if page is not None:  # feed_content never fetches, so it needs no page
        routes["/a"] = ok(page)
    a = add(conn, base + "/a", source=feed.name, feed_text=None)

    summary = run(conn, feed)

    row = article(conn, a)
    assert row["body_status"] == "failed" and row["body"] is None
    assert message in row["body_error"]
    assert len(summary.failed) == 1


# --- state: what runs again, what does not ---------------------------------


def test_ready_articles_are_not_processed_again(server, conn):
    base, routes = server
    routes["/a"] = ok(article_page("A title", "FIRST"))
    a = add(conn, base + "/a")
    run(conn, FULLTEXT)
    assert routes.hits["/a"] == 1

    routes["/a"] = ok(article_page("A title", "SECOND"))  # the page changes afterwards
    summary = run(conn, FULLTEXT)

    assert routes.hits["/a"] == 1  # not fetched again
    assert summary.already_complete == 1 and summary.outcomes == []
    assert "FIRST" in article(conn, a)["body"] and "SECOND" not in article(conn, a)["body"]


def test_a_failure_stays_eligible_and_succeeds_on_a_later_run(server, conn):
    base, routes = server
    routes["/a"] = CLOUDFLARE
    a = add(conn, base + "/a", feed_text="just a teaser")

    first = run(conn, FULLTEXT)
    row = article(conn, a)
    assert row["body_status"] == "failed"
    assert row["body"] is None  # no silent fallback to the teaser
    assert "Cloudflare" in row["body_error"] and row["body_checked_at"] == "2026-09-20T12:00:00Z"
    assert len(first.failed) == 1 and first.recovered == []

    second = run(conn, FULLTEXT)  # still refused: tried again, still failing
    assert routes.hits["/a"] == 2 and len(second.failed) == 1

    routes["/a"] = ok(article_page("A title", "ART"))
    third = run(conn, FULLTEXT)

    row = article(conn, a)
    assert routes.hits["/a"] == 3
    assert row["body_status"] == "ready" and row["body_error"] is None
    assert row["body"].startswith("ART paragraph 1")
    assert len(third.succeeded) == 1 and len(third.recovered) == 1


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [("cloudflare-block", "Cloudflare"), ("unexpected-exception", "unexpected ValueError")],
)
def test_one_failing_article_does_not_block_the_others(
    server, conn, monkeypatch, failure, expected_error
):
    base, routes = server
    routes["/a"] = ok(article_page("A", "AAA"))
    routes["/b"] = CLOUDFLARE
    routes["/c"] = ok(article_page("C", "CCC"))
    ids = [add(conn, base + path, title=path[1:].upper()) for path in ("/a", "/b", "/c")]
    if failure == "unexpected-exception":  # something no HTTP-level handling anticipates
        real_fetch = extract.fetch_page

        def fetch(url):
            if url.endswith("/b"):
                raise ValueError("something nobody anticipated")
            return real_fetch(url)

        monkeypatch.setattr(extract, "fetch_page", fetch)

    summary = run(conn, FULLTEXT)

    assert [article(conn, i)["body_status"] for i in ids] == ["ready", "failed", "ready"]
    assert [o.ok for o in summary.outcomes] == [True, False, True]
    assert expected_error in article(conn, ids[1])["body_error"]


# --- the command -----------------------------------------------------------


def shows(out: str, pattern: str) -> bool:
    """Whether the output matches a regex (spacing between columns doesn't matter)."""
    return re.search(pattern, out) is not None


def write_feeds(path):
    path.write_text(
        '[[feeds]]\nname = "Site"\nurl = "http://unused/site"\nstrategy = "fulltext"\n\n'
        '[[feeds]]\nname = "Notes"\nurl = "http://unused/notes"\nstrategy = "feed_content"\n',
        encoding="utf-8",
    )
    return path


def test_command_summary_shows_successes_failures_skips_and_retryable_failures(
    server, tmp_path, capsys
):
    base, routes = server
    routes["/done"] = ok(article_page("Done", "DONE"))
    routes["/good"] = ok(article_page("Good", "GOOD"))
    routes["/blocked"] = CLOUDFLARE
    db_path, feeds_path = tmp_path / "t.db", write_feeds(tmp_path / "feeds.toml")

    conn = db.connect(db_path)
    done = add(conn, base + "/done", title="Already done")
    db.mark_body_ready(conn, done, body="text", source="fulltext", checked_at=NOW)
    add(conn, base + "/good", title="Good article")
    add(conn, base + "/blocked", title="Blocked article")
    add(conn, base + "/notes", source="Notes", title="A note", feed_text="note text")
    conn.commit()
    conn.close()

    code = main(["extract", "--feeds", str(feeds_path), "--db", str(db_path)])
    out = capsys.readouterr().out

    assert code == 1  # something failed
    assert "1 article(s) already complete; 3 waiting (3 new, 0 retrying" in out
    assert shows(out, r"\bok\s+fulltext\s+Site\s+Good article")
    assert shows(out, r"FAILED\s+fulltext\s+Site\s+Blocked article.*Cloudflare")
    assert shows(out, r"already complete \(skipped\):\s+1\b")
    assert shows(out, r"extracted this run:\s+2\s+\(feed_content 1, fulltext 1\)")
    assert shows(out, r"failed this run:\s+1\b")
    assert "Retryable failures" in out and "1 x HTTP 403 Forbidden - Cloudflare" in out
    assert "python -m ainews extract" in out

    routes["/blocked"] = ok(article_page("Blocked", "OPEN"))  # the challenge goes away
    code = main(["extract", "--feeds", str(feeds_path), "--db", str(db_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "3 article(s) already complete; 1 waiting (0 new, 1 retrying after an earlier failure)" in out
    assert "ok*" in out and "[* 1 recovered from an earlier failure]" in out
    assert shows(out, r"failed this run:\s+0\b") and "Retryable failures" not in out


