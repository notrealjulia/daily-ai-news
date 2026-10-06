"""Read-only JSON + audio server for the V2 (React) dashboard.

Run from the project root:  python -m v2.server

Like app.py, this is a thin layer over ainews.dashboard: no SQL, no pipeline code and no
provider code, so it can't run a stage or call an LLM/TTS provider either (a test
enforces it). It serves exactly two things, on localhost only: what the Streamlit page
shows, as JSON, and the narration MP3s `narrate` wrote to audio/.
"""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ainews import dashboard

PORT = 8000  # v2/vite.config.ts proxies /api and /audio here


def _audio_url(path: Path | None) -> str | None:
    # The file is overwritten in place by every `narrate` run; its mtime in the URL keeps
    # the browser from playing yesterday's cached take.
    return f"/{path.as_posix()}?v={int(path.stat().st_mtime)}" if path else None


def payload() -> dict:
    """Everything app.py renders, in the same order, with the same text."""
    data = dashboard.open_dashboard()
    if data is None:
        return {"data": None, "database": dashboard.database_label()}
    sources = dashboard.sources_caption()
    return {
        "data": {
            "header": dashboard.format_header(data),
            "briefing_audio": _audio_url(dashboard.briefing_audio_path()),
            "empty_text": dashboard.empty_text(data.window_hours),
            # sources_caption() is escaped for Streamlit's markdown; React shows plain text.
            "sources": re.sub(r"\\(.)", r"\1", sources) if sources else None,
            "categories": [
                {
                    "name": category.name,
                    "headline": category.headline,
                    "digest": category.digest,
                    "audio": _audio_url(dashboard.category_audio_path(name)),
                    "expander_label": dashboard.expander_label(category.story_count),
                    "stories": [
                        {
                            "title": story.title,
                            "summary": story.summary,
                            "sources": [{"name": s.name, "url": s.url} for s in story.sources],
                        }
                        for story in category.stories
                    ],
                }
                for row in dashboard.CATEGORY_GRID
                for name in row
                for category in [data.categories[name]]
            ],
        }
    }


def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """(start, end) inclusive for a Range header, the whole file when there is none (or
    it isn't one we understand), or None when it can't be satisfied. Browsers need range
    requests to seek within audio."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header or "")
    if not match or not (match[1] or match[2]):
        return (0, size - 1)
    if not match[1]:  # "bytes=-500": the last 500 bytes
        start, end = max(size - int(match[2]), 0), size - 1
    else:
        start, end = int(match[1]), min(int(match[2]), size - 1) if match[2] else size - 1
    return (start, end) if start <= end else None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/dashboard":
                self._send(200, "application/json", json.dumps(payload()).encode())
            # Only a plain file name directly inside audio/: no "..", no subfolders.
            elif re.fullmatch(r"/audio/[\w-]+\.mp3", path) and Path(path[1:]).is_file():
                self._send_audio(Path(path[1:]))
            else:
                self._send(404, "text/plain", b"Not found")
        except ConnectionError:
            pass  # the browser dropped the request, as it does when seeking audio

    def _send(self, status: int, content_type: str, body: bytes, headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_audio(self, file: Path):
        content = file.read_bytes()
        span = byte_range(self.headers.get("Range"), len(content))
        if span is None:
            return self._send(416, "text/plain", b"", {"Content-Range": f"bytes */{len(content)}"})
        start, end = span
        headers = {"Accept-Ranges": "bytes"}
        if "Range" in self.headers:
            headers["Content-Range"] = f"bytes {start}-{end}/{len(content)}"
        self._send(206 if "Range" in self.headers else 200, "audio/mpeg", content[start : end + 1], headers)


if __name__ == "__main__":
    print(f"V2 API on http://localhost:{PORT} (read-only). Ctrl+C to stop.")
    ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()
