import pytest
from fastapi.testclient import TestClient

import app.main
from app.main import app as api
from app.main import get_conn
from app.rag.answer import Answer
from app.retrieval.search import Hit

# These tests need no database and no API key: get_conn is overridden and answer_question
# is replaced with a fake. TestClient is not used as a context manager, so the lifespan
# (which opens the real connection pool) never runs.


def hit(n: int, url: str) -> Hit:
    return Hit(
        content=f"chunk {n}",
        title=f"Title {url}",
        company="Acme",
        url=url,
        chunk_index=n,
        score=0.9,
    )


def source_json(numbers: list[int], url: str) -> dict:
    return {"numbers": numbers, "title": f"Title {url}", "company": "Acme", "url": url}


class FakeAnswer:
    """Stands in for answer_question: returns a fixed Answer and records the call."""

    def __init__(self, result: Answer):
        self.result = result
        self.calls = []

    def __call__(self, conn, question, k):
        self.calls.append((question, k))
        return self.result


@pytest.fixture
def client():
    api.dependency_overrides[get_conn] = lambda: None  # no real database connection
    yield TestClient(api)
    api.dependency_overrides.clear()  # the override lives on the app; don't leak it


@pytest.fixture
def fake(monkeypatch):
    """Call fake.set(Answer(...)) to choose what answer_question returns."""
    holder = FakeAnswer(Answer(text="ok", sources=[], cited=[]))

    def set_result(result: Answer) -> FakeAnswer:
        holder.result = result
        return holder

    holder.set = set_result
    # main.py did `from app.rag.answer import answer_question`, so patch main's copy.
    monkeypatch.setattr(app.main, "answer_question", holder)
    return holder


def test_answer_with_grouped_sources(client, fake):
    fake.set(
        Answer(
            text="Cause [2][1].",
            sources=[hit(1, "https://a.test"), hit(2, "https://a.test"), hit(3, "https://b.test")],
            cited=[2, 1],
        )
    )

    response = client.post("/ask", json={"question": "Why?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Cause [2][1].",
        "generated": True,
        "sources": [source_json([1, 2], "https://a.test")],  # b.test wasn't cited
    }


def test_several_documents_are_listed_in_order(client, fake):
    fake.set(
        Answer(
            text="[3] and [1].",
            sources=[hit(1, "https://a.test"), hit(2, "https://b.test"), hit(3, "https://c.test")],
            cited=[3, 1],
        )
    )

    sources = client.post("/ask", json={"question": "Why?"}).json()["sources"]

    assert sources == [source_json([1], "https://a.test"), source_json([3], "https://c.test")]


def test_k_is_passed_to_answer_question(client, fake):
    client.post("/ask", json={"question": "Why?", "k": 3})

    assert fake.calls == [("Why?", 3)]


def test_k_defaults_to_5(client, fake):
    client.post("/ask", json={"question": "Why?"})

    assert fake.calls == [("Why?", 5)]


def test_fallback_lists_every_source(client, fake):
    # The model failed: nothing is cited, but the user should still see what was found.
    fake.set(
        Answer(
            text="The answer service is unavailable right now.",
            sources=[hit(1, "https://a.test"), hit(2, "https://a.test"), hit(3, "https://b.test")],
            cited=[],
            generated=False,
        )
    )

    body = client.post("/ask", json={"question": "Why?"}).json()

    assert body["generated"] is False
    assert body["sources"] == [
        source_json([1, 2], "https://a.test"),
        source_json([3], "https://b.test"),
    ]


def test_nothing_found(client, fake):
    fake.set(Answer(text="No postmortems found.", sources=[], cited=[], generated=False))

    response = client.post("/ask", json={"question": "Why?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "No postmortems found.",
        "generated": False,
        "sources": [],
    }


def test_missing_company_is_null(client, fake):
    source = hit(1, "https://a.test")
    source.company = None
    fake.set(Answer(text="[1]", sources=[source], cited=[1]))

    (only,) = client.post("/ask", json={"question": "Why?"}).json()["sources"]

    assert only["company"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "x", "k": 0},
        {"question": "x", "k": 21},
        {"question": "x", "k": "five"},
        {},
    ],
    ids=["empty", "too-long", "k-0", "k-21", "k-not-a-number", "no-question"],
)
def test_bad_input_is_rejected(client, fake, body):
    response = client.post("/ask", json=body)

    assert response.status_code == 422
    assert fake.calls == []  # rejected before answer_question runs


def test_question_of_only_spaces_is_rejected(client, monkeypatch):
    def raises_value_error(conn, question, k):
        raise ValueError("empty question")

    monkeypatch.setattr(app.main, "answer_question", raises_value_error)

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
    assert response.json() == {"detail": "empty question"}


def test_get_is_not_allowed(client):
    assert client.get("/ask").status_code == 405


def test_health_still_works(client):
    assert client.get("/health").json() == {"status": "ok"}
