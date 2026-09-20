"""Shared fixtures."""

import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


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
