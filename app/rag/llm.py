import logging
from functools import cache

import anthropic

from app.config import settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """The model returned no usable answer (refused, or was cut off)."""


class LLMUnavailable(LLMError):
    """The model could not be reached after the SDK's retries (outage, overload, timeout).

    A subclass of LLMError, so `except LLMError` handles both; catch LLMUnavailable first
    when the two need different handling (this one is temporary, worth trying again later).
    """


@cache
def _client() -> anthropic.Anthropic:
    """One client for the whole process; it reuses HTTP connections between calls.

    Creating the client makes no network request, so nothing here can time out.
    """
    return anthropic.Anthropic(
        api_key=settings.llm_api_key.get_secret_value() or None,
        timeout=30.0,  # per attempt; the SDK retries twice, so the worst case is about 90 s
    )


def call_claude(system: str, user: str) -> str:
    """Send one question to Claude and return the answer text.

    Raises LLMUnavailable when the model can't be reached (timeout, connection error, 429,
    5xx) after the SDK's own retries, and LLMError when it declines or the answer is cut off.
    Other errors (bad key 401, unknown model 404, bad request 400) are setup bugs and are
    raised unchanged so they are not hidden behind "unavailable".
    """
    try:
        response = _client().messages.create(
            model=settings.llm_model,
            max_tokens=16000,  # thinking counts toward this; only tokens actually used are billed
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    # APITimeoutError is a subclass of APIConnectionError, so it must come first.
    except anthropic.APITimeoutError as e:
        raise LLMUnavailable("the model did not answer in time") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable("could not reach the model") from e
    except anthropic.RateLimitError as e:
        raise LLMUnavailable("rate limited by the API") from e
    except anthropic.APIStatusError as e:
        if e.status_code >= 500:  # includes 529 "overloaded"
            raise LLMUnavailable(f"the API returned {e.status_code}") from e
        raise

    # Logged before the stop_reason checks: refused and cut-off calls are billed too.
    # Thinking tokens are included in output_tokens. Prices: see the Usage page in the Console.
    log.info(
        "claude usage: model=%s input_tokens=%s output_tokens=%s stop_reason=%s",
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.stop_reason,
    )

    if response.stop_reason == "refusal":
        raise LLMError("the model declined to answer")
    if response.stop_reason == "max_tokens":
        raise LLMError("the answer was cut off (max_tokens reached)")
    # content is a list of blocks; skip thinking blocks and keep only the text.
    return "".join(block.text for block in response.content if block.type == "text")
