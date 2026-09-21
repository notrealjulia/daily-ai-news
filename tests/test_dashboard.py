"""Tests for the read-only dashboard: ainews/dashboard.py and app.py."""

import ast
import hashlib
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ainews import dashboard, db, defaults

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 15, 0, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
MODEL, PROMPT = defaults.DEFAULT_MODEL, defaults.DIGEST_PROMPT_VERSION


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def add_article(conn, title, *, source="The Decoder", url=None, published=NOW - timedelta(hours=2),
                english_title=None, enriched_with=("e", "p")) -> int:
    """`enriched_with` is (model, prompt version) of an enrichment holding `english_title`
    (None for an English title); the runs made by add_run were built from ("e", "p"). Pass
    enriched_with=None for an article that was never enriched."""
    url = url or f"https://example.test/{title.replace(' ', '-')}"
    db.insert_article(
        conn, source=source, url=url, title=title,
        published_at=published, fetched_at=published, feed_text=None,
    )  # fmt: skip
    article_id = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()["id"]
    if enriched_with is not None:
        db.insert_enrichment(
            conn, article_id=article_id, category="Other", summary="s", english_title=english_title,
            model=enriched_with[0], prompt_version=enriched_with[1], created_at=NOW,
        )  # fmt: skip
    return article_id


def add_run(conn, stories, *, created=NOW, digest_prompt=PROMPT, extra_digests=(), key="h") -> int:
    """A story run. Each story is (category, summary, [article ids]) or (..., status).

    Every category with a ready story gets a digest written with `digest_prompt` five
    minutes after the run. `extra_digests` are (category, summary, prompt_version, created_at).
    """
    run = db.create_story_run(
        conn, model="story-model", prompt_version="story-prompt", enrichment_model="e",
        enrichment_prompt_version="p", window_start=created - timedelta(hours=24),
        window_end=created, input_hash=key, created_at=created,
    )  # fmt: skip
    digested = set()
    for category, summary, article_ids, *rest in stories:
        status = rest[0] if rest else "ready"
        ready = status == "ready"
        # "unfinished" is a pending story that already has a category, so the run has a
        # digest for it: only the explicit "every story is ready" rule can reject that run.
        story = db.create_story(
            conn, run, article_ids=article_ids,
            category=category if status in ("ready", "unfinished") else None,
            summary=summary if ready else None, grouping_reason=None,
        )  # fmt: skip
        if status == "failed":
            db.mark_story_failed(conn, story, error="boom")
        if ready and category not in digested:
            digested.add(category)
            db.insert_digest(
                conn, run_id=run, category=category, story_count=1, total_story_count=1,
                headline=f"Headline for {category}", summary=f"Digest of {category}.",
                model=MODEL, prompt_version=digest_prompt,
                created_at=created + timedelta(minutes=5),
            )  # fmt: skip
    for category, summary, prompt_version, when in extra_digests:
        db.insert_digest(
            conn, run_id=run, category=category, story_count=1, total_story_count=1,
            headline="Old headline", summary=summary, model=MODEL, prompt_version=prompt_version,
            created_at=when,
        )  # fmt: skip
    conn.commit()
    return run


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- which run is shown ------------------------------------------------------


@pytest.mark.parametrize(
    ("newer_run", "shown"),
    [
        ("is-also-complete", "Newer run story."),  # the newest eligible run wins
        ("has-an-unfinished-story", "Older run story."),  # never a partly processed run
        ("has-only-an-old-prompt-digest", "Older run story."),  # never an old experimental prompt
        ("nothing-is-complete", None),
    ],
)
def test_only_the_latest_fully_processed_and_digested_run_is_shown(conn, newer_run, shown):
    article = add_article(conn, "An article")
    later = NOW + timedelta(hours=1)
    if newer_run != "nothing-is-complete":
        add_run(conn, [("Research", "Older run story.", [article])], key="older")
    if newer_run == "is-also-complete":
        add_run(conn, [("Research", "Newer run story.", [article])], created=later, key="newer")
    elif newer_run == "has-an-unfinished-story":
        # Research has a digest (from the ready story), yet the run is not fully processed.
        other = add_article(conn, "Another article")  # an article is in one story per run
        add_run(conn, [("Research", "Newer run story.", [article]),
                       ("Research", "Still being summarized.", [other], "unfinished")],
                created=later, key="newer")  # fmt: skip
    elif newer_run == "has-only-an-old-prompt-digest":
        add_run(conn, [("Research", "Newer run story.", [article])], created=later, digest_prompt="v-old", key="newer")
    else:  # the only run is still waiting for a story summary
        add_run(conn, [("Research", "Unfinished story.", [article], "pending")], key="only")

    data = dashboard.load_dashboard(conn)

    if shown is None:
        assert data is None
    else:
        assert [s.summary for s in data.categories["Research"].stories] == [shown]


# --- what is shown -----------------------------------------------------------


def test_categories_appear_in_grid_order_with_counts_digests_and_newest_first_stories(conn):
    r1 = add_article(conn, "Research one", published=NOW - timedelta(hours=5))
    r2 = add_article(conn, "Research two", published=NOW - timedelta(hours=1))
    i1 = add_article(conn, "Industry one", published=NOW - timedelta(hours=3))
    junk = add_article(conn, "Buy now", published=NOW - timedelta(hours=2))
    old_prompt_later = NOW + timedelta(hours=1)
    add_run(
        conn,
        [
            ("Research", "Older research story.", [r1]),
            ("Research", "Newer research story.", [r2]),
            ("Industry News", "An industry story.", [i1]),
            ("Spam", "Spam must never appear.", [junk]),  # can't happen in a real run; must still be hidden
        ],
        extra_digests=[("Research", "Old experimental digest.", "v-old", old_prompt_later)],
    )

    data = dashboard.load_dashboard(conn)

    assert list(data.categories) == [name for row in dashboard.CATEGORY_GRID for name in row]
    assert "Spam" not in data.categories and "Spam must never appear." not in repr(data)
    research = data.categories["Research"]
    assert [s.summary for s in research.stories] == ["Newer research story.", "Older research story."]
    assert research.digest == "Digest of Research."  # the current prompt's, not the later old one
    assert research.headline == "Headline for Research"  # stored with that digest, so from the same prompt
    for empty in ("Product Release", "Business", "Regulation & Policy", "Other"):
        assert data.categories[empty].story_count == 0 and data.categories[empty].digest is None
    assert (data.total_stories, data.total_articles, data.window_hours) == (3, 3, 24)
    # Last updated: the run's digests (15:05), not the old-prompt digest written later.
    assert data.last_updated == NOW + timedelta(minutes=5)
    assert dashboard.format_header(data, tz=UTC) == (
        "Last updated: Sep 20, 15:05 · Last 24 hours · 3 stories from 3 articles"
    )
    assert dashboard.expander_label(2) == "View 2 stories" and dashboard.expander_label(1) == "View 1 story"
    assert dashboard.empty_text(24) == "No stories in the last 24 hours."


def test_story_titles_are_english_and_the_source_titles_are_kept(conn):
    danish = add_article(conn, "Danske startups rejser kapital", english_title="Danish startups raise capital")
    english = add_article(conn, "OpenAI ships a model")  # already English: nothing to translate
    later = add_article(conn, "Anden dansk overskrift", english_title="Another Danish headline",
                        published=NOW - timedelta(hours=1))
    never_enriched = add_article(conn, "Uden berigelse", enriched_with=None)
    other_version = add_article(conn, "Kun en anden version", english_title="Translation from another prompt",
                                enriched_with=("e", "another-prompt"))
    add_run(conn, [
        ("Research", "Story one.", [danish]),
        ("Research", "Story two.", [english]),
        ("Research", "Story three, two articles.", [later, never_enriched]),  # earliest article is `never_enriched`
        ("Research", "Story four.", [other_version]),
    ])

    titles = {s.summary: s.title for s in dashboard.load_dashboard(conn).categories["Research"].stories}

    assert titles["Story one."] == "Danish startups raise capital"  # translated
    assert titles["Story two."] == "OpenAI ships a model"  # English titles are unchanged
    assert titles["Story three, two articles."] == "Uden berigelse"  # earliest article; no translation exists
    assert titles["Story four."] == "Kun en anden version"  # only the run's own enrichment counts
    stored = {r["title"] for r in conn.execute("SELECT title FROM articles")}
    assert "Danske startups rejser kapital" in stored and "Danish startups raise capital" not in stored


def test_a_multi_source_story_appears_once_with_all_its_sources_linked(conn):
    later = add_article(conn, "Later headline", source="TechCrunch AI",
                        url="https://techcrunch.test/b", published=NOW - timedelta(hours=1))  # fmt: skip
    earliest = add_article(conn, "Earliest headline", source="The Decoder",
                           url="https://the-decoder.test/a", published=NOW - timedelta(hours=3))  # fmt: skip
    odd = add_article(conn, "Odd link", source="Blog", url="javascript:alert(1)", published=NOW - timedelta(hours=2))
    paren = add_article(conn, "Paren link", source="Wiki", url="https://x.test/a_(b)", published=NOW - timedelta(minutes=30))
    add_run(conn, [("Research", "One combined summary.", [later, earliest, odd, paren])])

    data = dashboard.load_dashboard(conn)
    (story,) = data.categories["Research"].stories

    assert data.total_stories == 1 and data.total_articles == 4  # once, with all four articles
    assert story.title == "Earliest headline"  # the earliest article's title
    assert [s.name for s in story.sources] == ["The Decoder", "Blog", "TechCrunch AI", "Wiki"]
    markdown = dashboard.story_markdown(story)
    assert "[The Decoder](https://the-decoder.test/a)" in markdown
    assert "[TechCrunch AI](https://techcrunch.test/b)" in markdown
    assert "javascript" not in markdown and "Blog" in markdown  # not a link, still named
    assert "%28b%29" in markdown  # parentheses can't end the link early


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Anthropic may raise $100 billion at a $2 trillion valuation",
         r"Anthropic may raise \$100 billion at a \$2 trillion valuation"),
        ("*bold* and _italic_ [link](http://evil.test) `code`",
         r"\*bold\* and \_italic\_ \[link\](http://evil.test) \`code\`"),
        ("- <b>x</b> & y", r"\- \<b\>x\</b\> \& y"),
    ],
    ids=["dollar-amounts", "markdown-and-links", "html-and-list-marker"],
)  # fmt: skip
def test_text_from_the_web_is_escaped_before_it_is_shown_as_markdown(raw, expected):
    assert dashboard.escape_markdown(raw) == expected


FEEDS_TOML = """\
# a comment
[[feeds]]
name = "OpenAI"
url = "https://openai.com/news/rss.xml"
strategy = "fulltext"

# [[feeds]]
# name = "Disabled Source"

[[feeds]]
name = "Simon Willison"
url = "https://simonwillison.net/atom/everything/"
strategy = "feed_content"
"""


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (FEEDS_TOML, "Sources monitored: OpenAI · Simon Willison"),  # file order; names only
        ("# no feeds configured\n", None),
        ("[[feeds]\nbroken", None),  # not valid TOML: no footer, no crash
        (None, None),  # no such file
    ],
    ids=["names-only-in-file-order", "no-feeds", "invalid-toml", "missing-file"],
)
def test_the_footer_lists_the_source_names_from_feeds_toml(tmp_path, content, expected):
    path = tmp_path / "feeds.toml"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    assert dashboard.sources_caption(path) == expected


# --- read-only, and no way to reach the pipeline -----------------------------


def test_the_dashboard_can_only_read(tmp_path):
    path = tmp_path / "ainews.db"
    writer = db.connect(path)
    add_run(writer, [("Research", "A story.", [add_article(writer, "An article")])])
    writer.close()
    before = file_hash(path)

    readonly = db.connect_readonly(path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        readonly.execute("DELETE FROM stories")
    readonly.close()
    assert dashboard.open_dashboard(path) is not None
    assert file_hash(path) == before  # the database file was not touched

    missing = tmp_path / "missing.db"
    assert dashboard.open_dashboard(missing) is None and not missing.exists()  # nothing created

    old = tmp_path / "old.db"  # from before clustering existed: not migrated, just "nothing to show"
    legacy = sqlite3.connect(old)
    legacy.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY)")
    legacy.commit()
    legacy.close()
    old_before = file_hash(old)
    assert dashboard.open_dashboard(old) is None and file_hash(old) == old_before


def test_the_dashboard_code_cannot_reach_openai_or_run_the_pipeline():
    forbidden = ("openai", "trafilatura", "feedparser", "ainews.llm", "ainews.enrich", "ainews.stories",
                 "ainews.digest", "ainews.extract", "ainews.ingest", "ainews.inspect_feed")  # fmt: skip
    probe = f"import sys, ainews.dashboard; print(sorted(m for m in {forbidden!r} if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT)
    assert result.stdout.strip() == "[]", result.stderr  # importing it loads none of them

    source = APP.read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert imported == {"streamlit", "ainews.dashboard"}  # app.py is a thin renderer
    assert "sqlite3" not in source and "SELECT" not in source


# --- the Streamlit app itself ------------------------------------------------


@pytest.mark.parametrize("with_data", [True, False], ids=["with-a-completed-run", "before-anything-has-run"])
def test_the_streamlit_app_renders(tmp_path, monkeypatch, with_data):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)  # the app reads ./ainews.db and ./feeds.toml, like the command line does
    if with_data:
        (tmp_path / "feeds.toml").write_text(FEEDS_TOML, encoding="utf-8")
        conn = db.connect(tmp_path / "ainews.db")
        add_run(conn, [("Research", "A research story.", [add_article(conn, "A research article")])])
        conn.close()

    app = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not app.exception
    if not with_data:
        assert "No completed run yet" in app.info[0].value
        return
    everything = " ".join(
        [m.value for m in app.markdown] + [c.value for c in app.caption] + [e.label for e in app.expander]
        + [s.value for s in app.subheader]  # the category headings are native subheaders
    )
    for name in ("Product Release", "Industry News", "Research", "Business", "Regulation", "Other"):
        assert name in everything
    assert "#### Headline for Research" in everything  # the headline is a heading above the digest
    assert everything.index("Headline for Research") < everything.index("Digest of Research.")
    assert "Last 24 hours · 1 story from 1 article" in everything
    assert "View 1 story" in everything and "No stories in the last 24 hours." in everything
    assert "Spam" not in everything
    assert "Sources monitored: OpenAI · Simon Willison" in everything
    assert "openai.com" not in everything and "fulltext" not in everything  # names only
