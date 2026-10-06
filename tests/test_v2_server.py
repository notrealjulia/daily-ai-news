"""Tests for the V2 (React) dashboard's read-only server: v2/server.py."""

import json
import subprocess
import sys
from pathlib import Path

from v2 import server

ROOT = Path(__file__).resolve().parent.parent


def test_the_v2_server_cannot_reach_a_provider_or_run_the_pipeline():
    # The same rule as the Streamlit dashboard (see test_dashboard.py), for the same reason.
    forbidden = ("openai", "elevenlabs", "trafilatura", "feedparser", "streamlit", "ainews.llm",
                 "ainews.enrich", "ainews.stories", "ainews.digest", "ainews.narrate",
                 "ainews.extract", "ainews.ingest", "ainews.inspect_feed")  # fmt: skip
    probe = f"import sys, v2.server; print(sorted(m for m in {forbidden!r} if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT)
    assert result.stdout.strip() == "[]", result.stderr


def test_payload_before_anything_has_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no ./ainews.db here
    assert server.payload() == {"data": None, "database": "the local SQLite file"}


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
