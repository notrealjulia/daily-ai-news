"""Shared fixtures."""

import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture(autouse=True)
def no_real_openai_key(monkeypatch, tmp_path_factory):
    """No test may ever see the real API key, whatever is in .env or the environment.

    A dummy key is set, and the .env location is pointed at a file that doesn't exist.
    Even a test that accidentally reached the network would fail authentication rather
    than spend money.
    """
    from ainews import llm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(llm, "ENV_PATH", tmp_path_factory.getbasetemp() / "no-such-dir" / ".env")


class Routes(dict):
    """path -> (status, headers, body bytes). `hits` counts requests served per path."""

    def __init__(self):
        super().__init__()
        self.hits: Counter[str] = Counter()


@pytest.fixture
def server():
    """A local HTTP server. Yields (base_url, routes); unknown paths return 404."""
    routes = Routes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            routes.hits[self.path] += 1
            status, headers, body = routes.get(self.path, (404, {}, b"not found"))
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep test output quiet
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", routes
    httpd.shutdown()
    httpd.server_close()
