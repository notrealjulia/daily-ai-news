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


def test_repeated_headline_is_dropped():
    assert extract.clean_extracted_text("My Headline\nBody one\nBody two", "my headline") == (
        "Body one\nBody two"
    )


def test_first_paragraph_is_kept_when_it_is_not_the_headline():
    assert extract.clean_extracted_text("Body one\nBody two", "My Headline") == "Body one\nBody two"


def test_text_is_cut_at_a_stop_marker_and_the_marker_goes_too():
    text = "Body one\nBody two\nAI News Without the Hype\nSubscribe now\nFull archive"

    cleaned = extract.clean_extracted_text(text, "Headline", ("AI News Without the Hype",))

    assert cleaned == "Body one\nBody two"


def test_stop_markers_ignore_case_and_spacing():
    cleaned = extract.clean_extracted_text("Body\n  PROMO   block ", "T", ("promo block",))
    assert cleaned == "Body"


def test_headline_and_stop_marker_together():
    text = "Headline\nBody\nPromo\nMore promo"
    assert extract.clean_extracted_text(text, "Headline", ("Promo",)) == "Body"


def test_a_stop_marker_only_matches_a_whole_paragraph():
    text = "Body mentions the Promo block in passing\nMore body"
    assert extract.clean_extracted_text(text, "T", ("Promo",)) == text


def test_cleanup_can_leave_nothing():
    assert extract.clean_extracted_text("Headline\nPromo\nMore", "Headline", ("Promo",)) == ""


# --- fetching a page -------------------------------------------------------


def test_fetch_describes_a_successful_page(server):
    base, routes = server
    routes["/page"] = ok("<html>hi</html>")

    page = extract.fetch_page(base + "/page")

    assert page.problem is None and page.html == b"<html>hi</html>"
    assert "HTTP 200" in page.note and "text/html" in page.note


def test_fetch_notes_a_redirect(server):
    base, routes = server
    routes["/old"] = (302, {"Location": "/new"}, b"")
    routes["/new"] = ok("<html>new</html>")

    page = extract.fetch_page(base + "/old")

    assert page.problem is None
    assert f"redirected to {base}/new" in page.note


def test_fetch_recognises_a_cloudflare_challenge_and_does_not_work_around_it(server):
    base, routes = server
    routes["/page"] = CLOUDFLARE

    page = extract.fetch_page(base + "/page")

    assert page.html is None
    assert "Cloudflare" in page.problem and "not worked around" in page.problem


def test_fetch_reports_a_plain_403_and_a_404(server):
    base, routes = server
    routes["/forbidden"] = (403, {}, b"no")

    assert "refuses this client" in extract.fetch_page(base + "/forbidden").problem
    assert "HTTP 404" in extract.fetch_page(base + "/missing").problem


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


def test_stop_markers_cut_the_extracted_article(server, conn):
    base, routes = server
    long = "The quick brown fox jumps over the lazy dog while researchers measure results. " * 3
    promo = "Subscribe to THE TEST SITE for ad-free reading, a weekly newsletter and full archive access."
    routes["/a"] = ok(
        "<html><body><article><h1>A title</h1>"
        f"<p>BODY one. {long}</p><p>BODY two. {long}</p><p>BODY three. {long}</p>"
        f"<p>{promo}</p><p>TRAILING. {long}</p></article></body></html>"
    )
    a = add(conn, base + "/a")
    feed = ingest.Feed("Site", "http://unused/feed", "fulltext", (promo,))

    run(conn, feed)

    body = article(conn, a)["body"]
    assert "BODY three" in body
    assert "Subscribe" not in body and "TRAILING" not in body


def test_feed_content_uses_the_feed_text_and_never_touches_the_network(conn):
    # The URL points at a closed port: any attempt to fetch it would fail.
    a = add(conn, "http://127.0.0.1:1/never", source="Notes", feed_text="the whole post")

    run(conn, FEED_CONTENT)

    row = article(conn, a)
    assert (row["body_status"], row["body_source"], row["body"]) == (
        "ready", "feed_content", "the whole post",
    )  # fmt: skip


def test_feed_content_with_no_feed_text_fails_clearly(conn):
    a = add(conn, "http://127.0.0.1:1/never", source="Notes", feed_text=None)

    summary = run(conn, FEED_CONTENT)

    row = article(conn, a)
    assert row["body_status"] == "failed" and row["body"] is None
    assert "carries no text" in row["body_error"]
    assert len(summary.failed) == 1


def test_a_page_with_nothing_extractable_fails(server, conn):
    base, routes = server
    routes["/a"] = ok("<html><body></body></html>")
    a = add(conn, base + "/a")

    run(conn, FULLTEXT)

    row = article(conn, a)
    assert row["body_status"] == "failed"
    assert "Trafilatura returned no text" in row["body_error"]


def test_text_that_is_all_cleanup_leaves_the_article_failed(server, conn):
    base, routes = server
    routes["/a"] = ok(article_page("A title", "ART"))
    a = add(conn, base + "/a")
    # Make the article's very first paragraph a stop marker, so nothing is left.
    first_paragraph = (
        "ART paragraph 1: the quick brown fox jumps over the lazy dog while a group of "
        "researchers measure how consistently agents repeat their results across many "
        "independent runs of the same task."
    )
    feed = ingest.Feed("Site", "http://unused/feed", "fulltext", (first_paragraph,))

    run(conn, feed)

    row = article(conn, a)
    assert row["body_status"] == "failed" and "no text left after cleanup" in row["body_error"]


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


def test_one_failed_article_does_not_block_the_others(server, conn):
    base, routes = server
    routes["/a"] = ok(article_page("A", "AAA"))
    routes["/b"] = CLOUDFLARE
    routes["/c"] = ok(article_page("C", "CCC"))
    ids = [add(conn, base + path, title=path[1:].upper()) for path in ("/a", "/b", "/c")]

    summary = run(conn, FULLTEXT)

    statuses = [article(conn, i)["body_status"] for i in ids]
    assert statuses == ["ready", "failed", "ready"]
    assert [o.ok for o in summary.outcomes] == [True, False, True]


def test_an_unexpected_error_in_one_article_does_not_stop_the_run(server, conn, monkeypatch):
    base, routes = server
    routes["/a"] = ok(article_page("A", "AAA"))
    routes["/c"] = ok(article_page("C", "CCC"))
    ids = [add(conn, base + path) for path in ("/a", "/boom", "/c")]
    real_fetch = extract.fetch_page

    def fetch(url):
        if url.endswith("/boom"):
            raise ValueError("something nobody anticipated")
        return real_fetch(url)

    monkeypatch.setattr(extract, "fetch_page", fetch)

    summary = run(conn, FULLTEXT)

    assert [article(conn, i)["body_status"] for i in ids] == ["ready", "failed", "ready"]
    assert "unexpected ValueError: something nobody anticipated" in article(conn, ids[1])["body_error"]
    assert len(summary.failed) == 1


def test_articles_from_unconfigured_sources_are_left_alone_and_counted(server, conn):
    base, routes = server
    routes["/a"] = ok(article_page("A", "AAA"))
    known = add(conn, base + "/a", source="Site")
    unknown = add(conn, base + "/x", source="Removed feed", title="Orphan")

    summary = run(conn, FULLTEXT)

    assert article(conn, known)["body_status"] == "ready"
    assert article(conn, unknown)["body_status"] == "pending"
    assert summary.not_processed == [("Removed feed", "Orphan")]
    assert routes.hits["/x"] == 0


def test_each_result_is_reported_as_it_happens(server, conn):
    base, routes = server
    routes["/a"] = ok(article_page("A", "AAA"))
    routes["/b"] = CLOUDFLARE
    for path in ("/a", "/b"):
        add(conn, base + path)
    seen = []

    run(conn, FULLTEXT, report=seen.append)

    assert [o.ok for o in seen] == [True, False]


def test_each_result_is_committed_immediately(server, conn):
    base, routes = server
    routes["/a"] = ok(article_page("A", "AAA"))
    a = add(conn, base + "/a")
    committed_states = []

    def peek(outcome):  # report runs after the commit, so a second connection would see it
        committed_states.append(article(conn, a)["body_status"])

    run(conn, FULLTEXT, report=peek)

    assert committed_states == ["ready"]
    assert conn.in_transaction is False


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


def test_command_with_nothing_waiting_makes_no_requests(server, tmp_path, capsys):
    base, routes = server
    db_path, feeds_path = tmp_path / "t.db", write_feeds(tmp_path / "feeds.toml")
    conn = db.connect(db_path)
    a = add(conn, base + "/a")
    db.mark_body_ready(conn, a, body="text", source="fulltext", checked_at=NOW)
    conn.commit()
    conn.close()

    code = main(["extract", "--feeds", str(feeds_path), "--db", str(db_path)])
    out = capsys.readouterr().out

    assert code == 0 and sum(routes.hits.values()) == 0
    assert "1 article(s) already complete; 0 waiting" in out
    assert shows(out, r"extracted this run:\s+0\b")


def test_command_rejects_a_feeds_file_without_strategies(tmp_path):
    feeds_path = tmp_path / "feeds.toml"
    feeds_path.write_text('[[feeds]]\nname = "Site"\nurl = "http://unused"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'strategy'"):
        main(["extract", "--feeds", str(feeds_path), "--db", str(tmp_path / "t.db")])


def test_source_configuration_matches_what_the_evidence_showed():
    feeds = {feed.name: feed for feed in ingest.load_feeds("feeds.toml")}

    assert feeds["OpenAI"].strategy == "fulltext"  # its page fetch is intermittent, not blocked
    assert feeds["Simon Willison"].strategy == "feed_content"
    assert all(feeds[name].strategy == "fulltext" for name in feeds if name != "Simon Willison")
    assert feeds["The Decoder"].stop_markers == ("AI News Without the Hype – Curated by Humans",)
