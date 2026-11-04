import pytest

from app.ingestion import extract
from app.ingestion.extract import MIN_CHARS, extract_text

PARAGRAPHS = [
    (
        "At 14:02 UTC a configuration change was rolled out to the primary database cluster. "
        "Within minutes, connection pools were exhausted and API error rates climbed sharply."
    ),
    (
        "The on-call engineer paged the database team, who identified the faulty change and "
        "began a rollback while traffic was shifted to a secondary region."
    ),
    (
        "Error rates returned to normal at 14:47 UTC. Some customers saw delayed webhooks for "
        "a further twenty minutes while queued events were replayed."
    ),
    (
        "The root cause was a missing validation step in the deploy pipeline, which allowed a "
        "pool size of zero to reach production without review."
    ),
    (
        "Action items: add schema validation for database config, require a canary stage for "
        "all cluster changes, and alert on connection pool saturation."
    ),
]
LONG_BODY = "".join(f"<p>{p}</p>" for p in PARAGRAPHS)


def article_html(body: str) -> str:
    return f"""
    <html>
      <head><title>Postmortem</title></head>
      <body>
        <nav><a href="/">Home</a> <a href="/blog">Blog</a></nav>
        <article>
          <h1>Database outage postmortem</h1>
          {body}
        </article>
        <footer>Copyright 2026 Example Corp</footer>
      </body>
    </html>
    """


# --- real trafilatura ---


def test_extracts_long_article():
    text = extract_text(article_html(LONG_BODY))

    assert text is not None
    assert len(text) >= MIN_CHARS
    assert "connection pools were exhausted" in text


def test_strips_page_boilerplate():
    text = extract_text(article_html(LONG_BODY))

    assert text is not None
    assert "Copyright 2026 Example Corp" not in text


def test_short_article_returns_none():
    text = extract_text(article_html("<p>Short note: we had an outage.</p>"))

    assert text is None


def test_empty_html_returns_none():
    assert extract_text("") is None


def test_html_without_content_returns_none():
    assert extract_text("<html><body></body></html>") is None


# --- trafilatura mocked, to pin down the edge cases exactly ---


@pytest.fixture
def fake_extract(monkeypatch):
    def set_result(result):
        monkeypatch.setattr(extract.trafilatura, "extract", lambda html: result)

    return set_result


def test_none_from_trafilatura_returns_none(fake_extract):
    fake_extract(None)

    assert extract_text("<html></html>") is None


def test_empty_string_from_trafilatura_returns_none(fake_extract):
    fake_extract("")

    assert extract_text("<html></html>") is None


def test_text_exactly_min_chars_is_kept(fake_extract):
    fake_extract("a" * MIN_CHARS)

    assert extract_text("<html></html>") == "a" * MIN_CHARS


def test_text_one_below_min_chars_returns_none(fake_extract):
    fake_extract("a" * (MIN_CHARS - 1))

    assert extract_text("<html></html>") is None


def test_whitespace_is_stripped(fake_extract):
    fake_extract("  \n" + "a" * MIN_CHARS + "\n  ")

    assert extract_text("<html></html>") == "a" * MIN_CHARS


def test_length_is_checked_after_stripping(fake_extract):
    # Long enough only if the padding is counted.
    fake_extract(" " * 50 + "a" * (MIN_CHARS - 1) + " " * 50)

    assert extract_text("<html></html>") is None
