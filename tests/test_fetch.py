import httpx
import pytest

from app.ingestion import fetch
from app.ingestion.fetch import FetchError, fetch_html

URL = "https://example.com/postmortem"


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Replace time.sleep so tests run instantly, and record the delays that were requested."""
    recorded = []
    monkeypatch.setattr(fetch.time, "sleep", recorded.append)
    return recorded


def make_test_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- success ---------------------------------------------------------------


def test_returns_html_on_success(sleeps):
    client = make_test_client(lambda request: httpx.Response(200, text="<html>ok</html>"))

    assert fetch_html(client, URL) == "<html>ok</html>"
    assert sleeps == []  # no waiting when the first attempt works


# --- HTTP status codes -----------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_status_is_retried_then_succeeds(status):
    calls = []
    responses = [httpx.Response(status), httpx.Response(200, text="<html>ok</html>")]

    def handler(request):
        calls.append(request)
        return responses.pop(0)

    client = make_test_client(handler)

    assert fetch_html(client, URL) == "<html>ok</html>"
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_non_retryable_status_fails_immediately(status, sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    client = make_test_client(handler)

    with pytest.raises(FetchError, match=str(status)):
        fetch_html(client, URL)
    assert len(calls) == 1
    assert sleeps == []


def test_gives_up_after_retries_on_500():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    client = make_test_client(handler)

    with pytest.raises(FetchError, match="gave up after 4 attempts") as exc_info:
        fetch_html(client, URL, retries=3)
    assert len(calls) == 4
    assert exc_info.value.url == URL


# --- backoff ---------------------------------------------------------------


def test_backoff_delay_grows_with_each_retry(sleeps):
    client = make_test_client(lambda request: httpx.Response(500))

    with pytest.raises(FetchError):
        fetch_html(client, URL, retries=3, backoff=2)

    assert sleeps == [2, 4, 8]  # backoff ** attempt, no sleep after the last failure


# --- network errors --------------------------------------------------------


def test_network_error_is_retried_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, text="<html>ok</html>")

    client = make_test_client(handler)

    assert fetch_html(client, URL) == "<html>ok</html>"
    assert len(calls) == 2


def test_network_error_every_time_is_wrapped_in_fetch_error():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    client = make_test_client(handler)

    with pytest.raises(FetchError, match="gave up after 2 attempts"):
        fetch_html(client, URL, retries=1)
    assert len(calls) == 2


# --- errors that retrying cannot fix ---------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        httpx.InvalidURL("bad url"),
        httpx.UnsupportedProtocol("missing scheme"),
        httpx.TooManyRedirects("redirect loop"),
    ],
    ids=["invalid-url", "unsupported-protocol", "too-many-redirects"],
)
def test_permanent_errors_fail_immediately(error, sleeps):
    calls = []

    def handler(request):
        calls.append(request)
        raise error

    client = make_test_client(handler)

    with pytest.raises(FetchError):
        fetch_html(client, URL)
    assert len(calls) == 1
    assert sleeps == []