from types import SimpleNamespace

import anthropic
import httpx2  # the anthropic SDK is built on httpx2 (not the httpx package the fetcher uses)
import pytest

from app.config import settings
from app.rag import llm
from app.rag.llm import LLMError, LLMUnavailable, call_claude

# These tests replace the Anthropic client with a fake, so they make no API calls and cost
# nothing. The fake response has the same shape the SDK returns: .stop_reason and .content,
# a list of blocks that each have a .type.


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def thinking_block():
    return SimpleNamespace(type="thinking", thinking="")


class FakeClient:
    def __init__(self, content, stop_reason="end_turn", input_tokens=1200, output_tokens=300):
        self.response = SimpleNamespace(
            content=content,
            stop_reason=stop_reason,
            model="claude-test-model",
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        )
        self.calls: list[dict] = []
        self.messages = self  # so client.messages.create(...) works

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.fixture
def fake(monkeypatch):
    """Install a fake client; set fake.client.response before calling call_claude."""
    holder = SimpleNamespace(client=FakeClient([text_block("ok")]))
    monkeypatch.setattr(llm, "_client", lambda: holder.client)
    return holder


def test_returns_the_text(fake):
    fake.client = FakeClient([text_block("The disk filled up [1].")])

    assert call_claude("system", "user") == "The disk filled up [1]."


def test_thinking_blocks_are_skipped(fake):
    fake.client = FakeClient([thinking_block(), text_block("Answer [1].")])

    assert call_claude("system", "user") == "Answer [1]."


def test_several_text_blocks_are_joined(fake):
    fake.client = FakeClient([text_block("First part "), thinking_block(), text_block("[2].")])

    assert call_claude("system", "user") == "First part [2]."


def test_sends_system_user_and_settings(fake):
    call_claude("the system prompt", "the user message")

    (kwargs,) = fake.client.calls
    assert kwargs["system"] == "the system prompt"
    assert kwargs["messages"] == [{"role": "user", "content": "the user message"}]
    assert kwargs["model"] == settings.llm_model
    assert kwargs["max_tokens"] == 16000


def test_refusal_raises(fake):
    fake.client = FakeClient([], stop_reason="refusal")

    with pytest.raises(LLMError, match="declined"):
        call_claude("system", "user")


def test_cut_off_answer_raises(fake):
    fake.client = FakeClient([text_block("The root cause was")], stop_reason="max_tokens")

    with pytest.raises(LLMError, match="cut off"):
        call_claude("system", "user")


def test_client_is_created_once():
    llm._client.cache_clear()
    try:
        assert llm._client() is llm._client()
    finally:
        llm._client.cache_clear()


def test_client_has_a_short_timeout():
    llm._client.cache_clear()
    try:
        assert llm._client().timeout == 30.0
    finally:
        llm._client.cache_clear()


# --- the API is unavailable or rejects the request ---

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(code: int) -> anthropic.APIStatusError:
    """The error the SDK raises for an HTTP error status (after its own retries)."""
    response = httpx2.Response(code, request=REQUEST)
    classes = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        404: anthropic.NotFoundError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
    }
    return classes.get(code, anthropic.APIStatusError)(f"HTTP {code}", response=response, body=None)


class FailingClient:
    """A client whose create() raises the given SDK error."""

    def __init__(self, error: Exception):
        self.error = error
        self.messages = self

    def create(self, **kwargs):
        raise self.error


@pytest.mark.parametrize(
    "error, message",
    [
        (anthropic.APITimeoutError(request=REQUEST), "did not answer in time"),
        (anthropic.APIConnectionError(request=REQUEST), "could not reach"),
        (status_error(429), "rate limited"),
        (status_error(500), "returned 500"),
        (status_error(503), "returned 503"),
        (status_error(529), "returned 529"),  # overloaded
    ],
    ids=["timeout", "connection", "429", "500", "503", "529"],
)
def test_temporary_failures_raise_llm_unavailable(fake, error, message):
    fake.client = FailingClient(error)

    with pytest.raises(LLMUnavailable, match=message) as caught:
        call_claude("system", "user")

    assert caught.value.__cause__ is error  # the original SDK error is kept for logging


@pytest.mark.parametrize("code", [400, 401, 404])
def test_setup_errors_are_not_hidden(fake, code):
    # A bad request, wrong key or unknown model is a bug to fix, not an outage.
    error = status_error(code)
    fake.client = FailingClient(error)

    with pytest.raises(anthropic.APIStatusError) as caught:
        call_claude("system", "user")

    assert caught.value is error
    assert not isinstance(caught.value, LLMError)


def test_unavailable_is_also_an_llm_error():
    # So `except LLMError` in answer_question catches both kinds of failure.
    assert issubclass(LLMUnavailable, LLMError)


def test_refusal_is_not_unavailable(fake):
    fake.client = FakeClient([], stop_reason="refusal")

    with pytest.raises(LLMError) as caught:
        call_claude("system", "user")

    assert not isinstance(caught.value, LLMUnavailable)


# --- usage logging ---


def test_usage_is_logged(fake, caplog):
    fake.client = FakeClient([text_block("ok")], input_tokens=1834, output_tokens=412)

    with caplog.at_level("INFO", logger="app.rag.llm"):
        call_claude("system", "user")

    assert "input_tokens=1834" in caplog.text
    assert "output_tokens=412" in caplog.text
    assert "model=claude-test-model" in caplog.text
    assert "stop_reason=end_turn" in caplog.text


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_usage_is_logged_even_when_the_answer_is_unusable(fake, caplog, stop_reason):
    # These calls are still billed, so they should still show up in the log.
    fake.client = FakeClient([], stop_reason=stop_reason, input_tokens=900, output_tokens=16000)

    with caplog.at_level("INFO", logger="app.rag.llm"), pytest.raises(LLMError):
        call_claude("system", "user")

    assert "output_tokens=16000" in caplog.text
    assert f"stop_reason={stop_reason}" in caplog.text


def test_nothing_is_logged_when_the_call_fails(fake, caplog):
    fake.client = FailingClient(anthropic.APIConnectionError(request=REQUEST))

    with caplog.at_level("INFO", logger="app.rag.llm"), pytest.raises(LLMUnavailable):
        call_claude("system", "user")

    assert "claude usage" not in caplog.text
