import logging
import time

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "incident-rag/0.1 (+https://github.com/<your-username>/incident-rag)"


class FetchError(Exception):
    """Raised when downloading a source URL fails."""

    def __init__(self, url: str, message: str):
        self.url = url
        super().__init__(f"Failed to fetch {url}: {message}")


def make_client(timeout: float = 20.0) -> httpx.Client:
    headers = {"User-Agent": USER_AGENT}
    return httpx.Client(headers=headers, timeout=timeout, follow_redirects=True)


def fetch_html(client: httpx.Client, url: str, retries: int = 3, backoff: float = 1.5) -> str:
    """Download a page. `retries` is the number of extra attempts after the first one."""
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
        except (httpx.InvalidURL, httpx.UnsupportedProtocol) as e:
            raise FetchError(url, f"bad URL: {e}") from e
        except httpx.TransportError as e:
            last_exc = e
        except httpx.HTTPError as e:
            raise FetchError(url, str(e)) from e
        log.warning("attempt %s/%s failed for %s: %s", attempt + 1, retries + 1, url, last_exc)
    raise FetchError(url, f"gave up after {retries + 1} attempts: {last_exc}") from last_exc