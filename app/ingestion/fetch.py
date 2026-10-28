import httpx
import time

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

class FetchError(Exception):
    """Raised when downloading a source URL fails."""

    def __init__(self, url: str, message: str):
        self.url = url
        super().__init__(f"Failed to fetch {url}: {message}")


def make_client(timeout: float = 20.0) -> httpx.Client:
    headers = {
        "User-Agent": "incident-rag/0.1 (Contact: seef87as@gmail.com)",
    }
    return httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)


def fetch_html(client: httpx.Client, url: str, retries: int = 3, backoff: float = 1.5) -> str:
    last_exc: httpx.HTTPError | None = None
    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(backoff**attempt)
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in RETRYABLE_STATUS:
                raise FetchError(url, str(e)) from e
            last_exc = e
        except httpx.HTTPError as e:
            last_exc = e
    raise FetchError(url, f"gave up after {retries + 1} attempts: {last_exc}") from last_exc
