"""Tests for the V2 (React) dashboard's read-only server: v2/server.py."""

import json
import subprocess
import sys
from pathlib import Path

from ainews import db, defaults
from test_dashboard import FEEDS_TOML, add_article, add_run
from v2 import server

ROOT = Path(__file__).resolve().parent.parent


def test_the_v2_server_cannot_reach_a_provider_or_run_the_pipeline():
    # The same rule as ainews.dashboard itself (see test_dashboard.py), for the same reason.
    forbidden = ("openai", "elevenlabs", "trafilatura", "feedparser", "streamlit", "ainews.llm",
                 "ainews.enrich", "ainews.stories", "ainews.digest", "ainews.narrate",
                 "ainews.extract", "ainews.ingest", "ainews.inspect_feed")  # fmt: skip
    probe = f"import sys, v2.server; print(sorted(m for m in {forbidden!r} if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT)
    assert result.stdout.strip() == "[]", result.stderr


def test_payload_before_anything_has_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no ./ainews.db here
    assert server.payload() == {"data": None, "database": "the local SQLite file"}


def test_payload_with_a_completed_run(tmp_path, monkeypatch):
    """What the local page and the deployed snapshot both show, end to end."""
    monkeypatch.chdir(tmp_path)  # payload() reads ./ainews.db, ./feeds.toml and ./audio/, like the command line does
    (tmp_path / "feeds.toml").write_text(FEEDS_TOML, encoding="utf-8")
    conn = db.connect(tmp_path / "ainews.db")
    add_run(conn, [("Research", "A research story.", [add_article(conn, "A research article")])])
    conn.close()
    (tmp_path / "audio").mkdir()
    for name in ("research.mp3", "briefing.mp3"):
        (tmp_path / "audio" / name).write_bytes(b"fake-mp3-bytes")

    data = server.payload()["data"]

    assert data["header"].endswith("Last 24 hours · 1 story from 1 article")
    assert data["sources"] == "Sources monitored: OpenAI · Simon Willison"  # file order; names only
    assert data["briefing_audio"].startswith("/audio/briefing.mp3?v=")
    assert data["empty_text"] == "No stories in the last 24 hours."
    names = [category["name"] for category in data["categories"]]
    assert names == list(defaults.CATEGORY_ORDER)  # all six in reading order, never Spam
    research = data["categories"][names.index("Research")]
    assert (research["headline"], research["digest"]) == ("Headline for Research", "Digest of Research.")
    assert research["expander_label"] == "View 1 story"
    assert research["audio"].startswith("/audio/research.mp3?v=")
    assert [story["title"] for story in research["stories"]] == ["A research article"]
    business = data["categories"][names.index("Business")]
    assert business["stories"] == [] and business["audio"] is None  # no stories, no narration file


def test_snapshot_writes_the_same_json_the_server_serves(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "payload", lambda: {"data": {"header": "Last updated: today"}})
    target = tmp_path / "public" / "dashboard.json"
    assert server.write_snapshot(target) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == {"data": {"header": "Last updated: today"}}


def test_snapshot_refuses_when_there_is_no_completed_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no ./ainews.db here
    target = tmp_path / "dashboard.json"
    assert server.write_snapshot(target) == 1
    assert not target.exists()  # so the deploy fails instead of publishing an empty page


def test_byte_range():
    assert server.byte_range(None, 10) == (0, 9)
    assert server.byte_range("bytes=2-", 10) == (2, 9)
    assert server.byte_range("bytes=2-5", 10) == (2, 5)
    assert server.byte_range("bytes=2-99", 10) == (2, 9)
    assert server.byte_range("bytes=-3", 10) == (7, 9)
    assert server.byte_range("bytes=20-", 10) is None  # past the end: 416
