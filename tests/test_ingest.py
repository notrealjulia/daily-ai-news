"""Tests for the deterministic ingestion logic.

Feeds are built as XML strings and parsed by the real feedparser, so the tests
exercise feedparser's actual timestamp handling. Nothing touches the network.
"""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import feedparser
import pytest

from ainews import db, ingest

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
CUTOFF = NOW - timedelta(hours=24)  # 2026-09-19 12:00:00 UTC
SECOND = timedelta(seconds=1)


# --- helpers ----------------------------------------------------------------


def rfc822(moment: datetime, offset_hours: float = 0) -> str:
    """Format an instant as an RSS date in the given UTC offset, e.g. '... +0200'."""
    return format_datetime(moment.astimezone(timezone(timedelta(hours=offset_hours))))


def rss(*items: tuple[str, str | None, str | None]) -> feedparser.FeedParserDict:
    """Build a parsed RSS feed from (title, link, pubDate-text) triples.

    A None link or pubDate omits that element entirely.
    """
    body = ""
    for title, link, pub in items:
        body += f"<item><title>{title}</title>"
        if link is not None:
            body += f"<link>{link}</link>"
        if pub is not None:
            body += f"<pubDate>{pub}</pubDate>"
        body += "</item>"
    return feedparser.parse(
        f"<rss version='2.0'><channel><title>t</title>{body}</channel></rss>"
    )


def atom(entry_xml: str) -> feedparser.FeedParserDict:
    return feedparser.parse(
        "<feed xmlns='http://www.w3.org/2005/Atom'><title>t</title>"
        f"<entry><title>a</title><link href='http://x/1'/>{entry_xml}</entry></feed>"
    )


def one_item(pub: str | None, link: str | None = "http://x/1"):
    return rss(("a", link, pub))


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def run(conn, parsed, *, now=NOW, **kwargs) -> ingest.FeedStats:
    return ingest.ingest_parsed_feed(conn, "src", parsed, now=now, **kwargs)


def count_articles(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]


def stored_urls(conn) -> list[str]:
    """Stored URLs, in the order they were inserted."""
    return [r["url"] for r in conn.execute("SELECT url FROM articles ORDER BY id")]


def assert_stats_add_up(stats: ingest.FeedStats) -> None:
    """The accounting FeedStats promises, whatever kind of feed it was."""
    assert stats.found == stats.in_window + stats.too_old + stats.bad_date + stats.undated
    assert stats.in_window + stats.undated == (
        stats.inserted + stats.already_present + stats.no_url + stats.not_examined
    )


# --- the window setting -----------------------------------------------------


def test_default_window_is_24_hours():
    assert ingest.MAX_ARTICLE_AGE == timedelta(hours=24)


def test_window_is_configurable(conn):
    thirty_hours_old = one_item(rfc822(NOW - timedelta(hours=30)))
    assert run(conn, thirty_hours_old).too_old == 1
    assert run(conn, thirty_hours_old, max_age=timedelta(hours=48)).inserted == 1


# --- window boundary --------------------------------------------------------


@pytest.mark.parametrize(
    ("published", "expect_inserted"),
    [
        (CUTOFF - SECOND, False),  # one second too old
        (CUTOFF, True),  # exactly at the cutoff is in (>=)
        (CUTOFF + SECOND, True),
        (NOW, True),
    ],
    ids=["1s-before-cutoff", "exactly-at-cutoff", "1s-after-cutoff", "now"],
)
def test_window_boundary(conn, published, expect_inserted):
    stats = run(conn, one_item(rfc822(published)))

    assert stats.found == 1
    assert stats.inserted == (1 if expect_inserted else 0)
    assert stats.too_old == (0 if expect_inserted else 1)
    assert count_articles(conn) == (1 if expect_inserted else 0)


# --- timezone handling ------------------------------------------------------
# Each case names the same UTC instant (the cutoff, or one second before it)
# but writes it with a different offset. If offsets were ignored, the wall-clock
# time would be compared against the UTC cutoff and many of these would flip.

OFFSETS = [-12, -8, -5, 0, 2, 5.5, 14]


@pytest.mark.parametrize("offset", OFFSETS)
def test_cutoff_instant_is_in_window_in_any_timezone(conn, offset):
    assert run(conn, one_item(rfc822(CUTOFF, offset))).inserted == 1


@pytest.mark.parametrize("offset", OFFSETS)
def test_one_second_before_cutoff_is_too_old_in_any_timezone(conn, offset):
    assert run(conn, one_item(rfc822(CUTOFF - SECOND, offset))).too_old == 1


def test_named_timezone_is_converted():
    # 07:00 EST is 12:00 UTC, which is exactly the cutoff.
    entry = one_item("Sat, 19 Sep 2026 07:00:00 EST").entries[0]
    assert ingest.entry_published_at(entry) == CUTOFF


def test_result_is_aware_utc():
    entry = one_item(rfc822(CUTOFF, 2)).entries[0]
    published = ingest.entry_published_at(entry)
    assert published == CUTOFF
    assert published.utcoffset() == timedelta(0)


def test_stored_timestamp_is_utc_text(conn):
    run(conn, one_item(rfc822(CUTOFF, 2)))  # written with a +0200 offset
    row = conn.execute("SELECT published_at FROM articles").fetchone()
    assert row["published_at"] == "2026-09-19T12:00:00Z"


def test_atom_offsets_and_z_suffix():
    with_offset = atom("<published>2026-09-19T14:00:00+02:00</published>")
    with_z = atom("<published>2026-09-19T12:00:00Z</published>")
    assert ingest.entry_published_at(with_offset.entries[0]) == CUTOFF
    assert ingest.entry_published_at(with_z.entries[0]) == CUTOFF


def test_atom_prefers_published_over_updated(conn):
    # Published long ago, edited recently: the article is old.
    parsed = atom(
        "<published>2026-01-01T00:00:00Z</published>"
        "<updated>2026-09-20T11:00:00Z</updated>"
    )
    assert run(conn, parsed).too_old == 1


def test_falls_back_to_updated_when_no_published(conn):
    parsed = atom("<updated>2026-09-20T11:00:00Z</updated>")
    assert run(conn, parsed).inserted == 1


def test_naive_now_is_rejected(conn):
    with pytest.raises(ValueError):
        run(conn, one_item(rfc822(NOW)), now=datetime(2026, 9, 20, 12, 0, 0))


# --- missing / invalid dates in a dated feed: skipped, never assumed recent --

UNUSABLE_DATES = [
    pytest.param(None, id="missing"),  # no <pubDate> at all
    pytest.param("not a date", id="garbage"),
    pytest.param("", id="empty"),
    pytest.param("Sat, 19 Sep 2026 18:00:00", id="no-timezone"),  # RSS date lacking a zone
]


@pytest.mark.parametrize("pub", UNUSABLE_DATES)
def test_undated_entry_in_a_dated_feed_is_skipped(conn, pub):
    # The feed has one good date, so it is a dated feed and the bad entry is
    # skipped, not assumed recent and not treated as a dateless feed.
    parsed = rss(
        ("good", "http://x/good", rfc822(NOW - timedelta(hours=1))),
        ("bad", "http://x/bad", pub),
    )

    stats = run(conn, parsed)

    assert (stats.found, stats.inserted, stats.bad_date, stats.undated) == (2, 1, 1, 0)
    assert stored_urls(conn) == ["http://x/good"]
    assert_stats_add_up(stats)


# --- deduplication and reporting --------------------------------------------


def test_second_run_finds_everything_already_present(conn):
    parsed = one_item(rfc822(NOW - timedelta(hours=1)))

    first = run(conn, parsed)
    second = run(conn, parsed)

    assert (first.inserted, first.already_present) == (1, 0)
    assert (second.inserted, second.already_present) == (0, 1)
    assert count_articles(conn) == 1


def test_duplicate_url_within_one_feed(conn):
    recent = rfc822(NOW - timedelta(hours=1))
    stats = run(conn, rss(("a", "http://x/1", recent), ("b", "http://x/1", recent)))

    assert (stats.inserted, stats.already_present) == (1, 1)


def test_entry_without_link_is_counted_not_stored(conn):
    stats = run(conn, one_item(rfc822(NOW), link=None))

    assert (stats.in_window, stats.no_url, stats.inserted) == (1, 1, 0)


def test_stats_for_a_mixed_feed(conn):
    recent = rfc822(NOW - timedelta(hours=2))
    old = rfc822(NOW - timedelta(days=3))
    parsed = rss(
        ("new-1", "http://x/1", recent),
        ("new-2", "http://x/2", recent),
        ("old-1", "http://x/3", old),
        ("old-2", "http://x/4", old),
        ("old-3", "http://x/5", old),
        ("nodate", "http://x/6", None),
        ("nolink", None, recent),
    )
    # Pre-existing row, so one of the in-window entries is a duplicate.
    db.insert_article(
        conn, source="src", url="http://x/1", title="new-1",
        published_at=NOW - timedelta(hours=2), fetched_at=NOW, content=None,
    )  # fmt: skip

    stats = run(conn, parsed)

    assert stats == ingest.FeedStats(
        found=7, in_window=3, too_old=3, bad_date=1, no_url=1, inserted=1, already_present=1
    )
    assert_stats_add_up(stats)


def test_stats_can_be_summed():
    a = ingest.FeedStats(found=2, inserted=1, too_old=1)
    b = ingest.FeedStats(found=3, inserted=2, bad_date=1)
    assert a + b == ingest.FeedStats(found=5, inserted=3, too_old=1, bad_date=1)


# --- dateless feeds: walk from the top until the first known URL ------------
# undated_feed("A", "B", "C") lists A first, i.e. A is the newest entry.


def undated_feed(*names: str) -> feedparser.FeedParserDict:
    return rss(*((name, f"http://x/{name}", None) for name in names))


def urls(*names: str) -> list[str]:
    return [f"http://x/{name}" for name in names]


def test_dateless_first_run_inserts_only_the_newest_entry(conn):  # requirement 1
    stats = run(conn, undated_feed("A", "B", "C", "D"))

    assert stored_urls(conn) == urls("A")
    assert (stats.found, stats.undated, stats.inserted, stats.not_examined) == (4, 4, 1, 3)
    assert_stats_add_up(stats)


def test_dateless_rerun_with_nothing_new_inserts_nothing(conn):  # requirement 2
    feed = undated_feed("A", "B", "C", "D")
    run(conn, feed)

    stats = run(conn, feed)

    assert stored_urls(conn) == urls("A")
    assert (stats.inserted, stats.already_present, stats.not_examined) == (0, 1, 3)
    assert_stats_add_up(stats)


def test_dateless_one_new_entry_is_inserted(conn):  # requirement 3
    run(conn, undated_feed("A", "B", "C"))  # first run stores A

    stats = run(conn, undated_feed("E", "A", "B", "C"))

    assert stored_urls(conn) == urls("A", "E")
    assert (stats.inserted, stats.already_present, stats.not_examined) == (1, 1, 2)
    assert_stats_add_up(stats)


def test_dateless_several_new_entries_are_inserted_up_to_the_first_known(conn):  # requirement 4
    run(conn, undated_feed("D", "E", "F"))  # first run stores D

    stats = run(conn, undated_feed("A", "B", "C", "D", "E", "F"))

    assert sorted(stored_urls(conn)) == urls("A", "B", "C", "D")
    assert (stats.inserted, stats.already_present, stats.not_examined) == (3, 1, 2)
    assert_stats_add_up(stats)


def test_dateless_entries_below_a_known_url_are_not_processed(conn):  # requirement 5
    run(conn, undated_feed("D"))  # D is known

    # Z and Y are unseen, but they sit below the known D, so they are left alone.
    stats = run(conn, undated_feed("A", "D", "Z", "Y"))

    assert stored_urls(conn) == urls("D", "A")
    assert (stats.inserted, stats.already_present, stats.not_examined) == (1, 1, 2)
    assert_stats_add_up(stats)


def test_dateless_articles_have_no_published_at_and_record_discovery_time(conn):
    run(conn, undated_feed("A"), now=NOW)
    run(conn, undated_feed("B", "A"), now=NOW + timedelta(hours=1))

    rows = {r["url"]: r for r in conn.execute("SELECT * FROM articles")}
    assert rows["http://x/A"]["published_at"] is None
    assert rows["http://x/B"]["published_at"] is None
    assert rows["http://x/A"]["fetched_at"] == "2026-09-20T12:00:00Z"
    assert rows["http://x/B"]["fetched_at"] == "2026-09-20T13:00:00Z"


def test_new_dateless_entries_are_stored_oldest_first(conn):
    run(conn, undated_feed("D"))
    run(conn, undated_feed("A", "B", "C", "D"))

    # Row order is oldest to newest, so the newest entry is always written last.
    assert stored_urls(conn) == urls("D", "C", "B", "A")


def test_interrupted_ingest_leaves_no_gap_that_a_later_run_would_miss(conn, monkeypatch):
    run(conn, undated_feed("D"))
    real_insert = db.insert_article
    calls = 0

    def crash_on_last_insert(conn, **fields):
        nonlocal calls
        calls += 1
        if calls == 3:  # A, B and C are new; die after writing two of them
            raise RuntimeError("simulated crash")
        return real_insert(conn, **fields)

    monkeypatch.setattr(db, "insert_article", crash_on_last_insert)
    with pytest.raises(RuntimeError):
        run(conn, undated_feed("A", "B", "C", "D"))
    conn.commit()  # worst case: whatever was written before the crash is kept

    monkeypatch.setattr(db, "insert_article", real_insert)
    run(conn, undated_feed("A", "B", "C", "D"))

    assert sorted(stored_urls(conn)) == urls("A", "B", "C", "D")


@pytest.mark.parametrize("pub", UNUSABLE_DATES)
def test_feed_where_no_entry_has_a_usable_date_is_dateless(conn, pub):
    stats = run(conn, rss(("A", "http://x/A", pub), ("B", "http://x/B", pub)))

    assert (stats.undated, stats.bad_date, stats.inserted) == (2, 0, 1)
    assert stored_urls(conn) == urls("A")
    assert_stats_add_up(stats)


def test_dateless_entry_without_link_is_skipped_and_the_next_one_is_newest(conn):
    stats = run(conn, rss(("x", None, None), ("A", "http://x/A", None), ("B", "http://x/B", None)))

    assert stored_urls(conn) == urls("A")
    assert (stats.no_url, stats.inserted, stats.not_examined) == (1, 1, 1)
    assert_stats_add_up(stats)


# --- dated feeds keep their 24-hour behavior (requirement 6) ----------------
# The window and timezone tests above already cover the boundary; these check
# that the dateless rules don't leak into dated feeds.


def test_dated_feed_first_run_inserts_every_recent_entry_not_just_the_newest(conn):
    parsed = rss(
        ("a", "http://x/a", rfc822(NOW - timedelta(hours=1))),
        ("b", "http://x/b", rfc822(NOW - timedelta(hours=2))),
        ("c", "http://x/c", rfc822(NOW - timedelta(hours=3))),
    )

    stats = run(conn, parsed)

    assert sorted(stored_urls(conn)) == urls("a", "b", "c")
    assert (stats.inserted, stats.undated, stats.not_examined) == (3, 0, 0)


def test_dated_feed_does_not_stop_at_a_known_url(conn):
    hours_ago = lambda h: rfc822(NOW - timedelta(hours=h))  # noqa: E731
    run(conn, rss(("known", "http://x/known", hours_ago(2))))

    stats = run(
        conn,
        rss(
            ("new-1", "http://x/new-1", hours_ago(1)),
            ("known", "http://x/known", hours_ago(2)),
            ("new-2", "http://x/new-2", hours_ago(3)),
        ),
    )

    assert sorted(stored_urls(conn)) == urls("known", "new-1", "new-2")
    assert (stats.inserted, stats.already_present, stats.not_examined) == (2, 1, 0)


def test_dated_feed_still_uses_the_window_not_newest_first(conn):
    parsed = rss(
        ("new", "http://x/new", rfc822(NOW - timedelta(hours=1))),
        ("old", "http://x/old", rfc822(NOW - timedelta(days=3))),
    )

    stats = run(conn, parsed)

    assert stored_urls(conn) == urls("new")
    assert (stats.too_old, stats.undated) == (1, 0)


# --- content normalization --------------------------------------------------


def test_strip_html_removes_tags_scripts_and_decodes_entities():
    html = "<p>Hello &amp; <b>welcome</b></p><script>evil()</script><p>Second  para</p>"
    assert ingest.strip_html(html) == "Hello & welcome\nSecond para"


def test_content_prefers_full_content_over_summary():
    parsed = feedparser.parse(
        "<rss version='2.0' xmlns:content='http://purl.org/rss/1.0/modules/content/'>"
        "<channel><title>t</title><item><title>a</title>"
        "<description>short teaser</description>"
        "<content:encoded><![CDATA[<p>the full text</p>]]></content:encoded>"
        "</item></channel></rss>"
    )
    assert ingest.entry_content(parsed.entries[0]) == "the full text"


def test_content_falls_back_to_summary_then_none():
    with_summary = feedparser.parse(
        "<rss version='2.0'><channel><title>t</title><item><title>a</title>"
        "<description>&lt;p&gt;teaser&lt;/p&gt;</description></item></channel></rss>"
    )
    assert ingest.entry_content(with_summary.entries[0]) == "teaser"
    assert ingest.entry_content(one_item(None).entries[0]) is None


# --- feeds.toml -------------------------------------------------------------


def test_repo_feeds_file_loads():
    feeds = ingest.load_feeds("feeds.toml")
    names = [feed.name for feed in feeds]
    assert feeds and len(names) == len(set(names))
    assert all(feed.url.startswith("https://") for feed in feeds)


def test_feed_missing_a_field_is_reported(tmp_path):
    path = tmp_path / "feeds.toml"
    path.write_text('[[feeds]]\nname = "no url here"\n')
    with pytest.raises(ValueError, match="feed #1 is missing 'url'"):
        ingest.load_feeds(path)
