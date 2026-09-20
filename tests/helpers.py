"""Small builders shared by several test files."""


def ok(body: str | bytes, content_type: str = "text/html; charset=utf-8"):
    """A 200 response tuple for the `server` fixture's routes."""
    data = body.encode() if isinstance(body, str) else body
    return 200, {"Content-Type": content_type}, data


def article_page(title: str, marker: str) -> str:
    """A page whose article is long enough for Trafilatura to extract."""
    paragraphs = "".join(
        f"<p>{marker} paragraph {n}: the quick brown fox jumps over the lazy dog while "
        "a group of researchers measure how consistently agents repeat their results "
        "across many independent runs of the same task.</p>"
        for n in range(1, 7)
    )
    return (
        f"<html><head><title>{title}</title></head><body><nav><a href='/'>Home</a></nav>"
        f"<article><h1>{title}</h1>{paragraphs}</article><footer>© Example</footer></body></html>"
    )
