import httpx
import pytest

from app.ingestion import fetch
from app.ingestion.fetch import FetchError, fetch_html

URL = "https://example.com/postmortem"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda _seconds: None)


def make_test_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_returns_html_on_success():
    client = make_test_client(lambda request: httpx.Response(200, text="<html>ok</html>"))

    assert fetch_html(client, URL) == "<html>ok</html>"


def test_retries_on_503_then_succeeds():
    responses = [httpx.Response(503), httpx.Response(200, text="<html>ok</html>")]
    client = make_test_client(lambda request: responses.pop(0))

    assert fetch_html(client, URL) == "<html>ok</html>"
    assert responses == []


def test_404_fails_immediately_without_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    client = make_test_client(handler)

    with pytest.raises(FetchError, match="404"):
        fetch_html(client, URL)
    assert len(calls) == 1


def test_gives_up_after_retries_on_500():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    client = make_test_client(handler)

    with pytest.raises(FetchError, match="gave up after 4 attempts"):
        fetch_html(client, URL, retries=3)
    assert len(calls) == 4


def test_network_error_is_wrapped_in_fetch_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = make_test_client(handler)

    with pytest.raises(FetchError):
        fetch_html(client, URL, retries=1)