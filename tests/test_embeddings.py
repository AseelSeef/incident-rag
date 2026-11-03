import math

import pytest

from app.embeddings import CACHE_DIR, EMBEDDING_DIM, embed_query, embed_texts


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_empty_list_returns_empty_list():
    assert embed_texts([]) == []


def test_one_vector_per_text_with_right_size():
    vectors = embed_texts(["database outage", "disk full", "DNS failure"])

    assert len(vectors) == 3
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_returns_plain_python_floats():
    (vector,) = embed_texts(["database outage"])

    assert isinstance(vector, list)
    assert all(isinstance(x, float) for x in vector)


def test_vectors_are_unit_length():
    for vector in embed_texts(["database outage", "a much longer sentence about a network partition"]):
        assert math.isclose(math.sqrt(dot(vector, vector)), 1.0, abs_tol=1e-3)


def test_same_text_gives_same_vector():
    a, b = embed_texts(["database outage", "database outage"])

    assert all(math.isclose(x, y, abs_tol=1e-5) for x, y in zip(a, b))


def test_order_matches_input():
    texts = ["database outage", "chocolate cake recipe"]

    together = embed_texts(texts)
    separately = [embed_texts([t])[0] for t in texts]

    for a, b in zip(together, separately):
        assert dot(a, b) > 0.999


def test_related_texts_are_closer_than_unrelated():
    outage, db_down, cake = embed_texts(
        ["the database went offline", "our DB cluster was down", "chocolate cake recipe"]
    )

    assert dot(outage, db_down) > dot(outage, cake)


def test_batch_size_does_not_change_results():
    texts = [f"incident number {i}" for i in range(5)]

    small = embed_texts(texts, batch_size=2)
    large = embed_texts(texts, batch_size=32)

    for a, b in zip(small, large):
        assert dot(a, b) > 0.999


# --- embed_query ---


def test_query_vector_has_right_size_and_unit_length():
    vector = embed_query("why did the database go down?")

    assert len(vector) == EMBEDDING_DIM
    assert math.isclose(math.sqrt(dot(vector, vector)), 1.0, abs_tol=1e-3)


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_empty_question_raises(question):
    with pytest.raises(ValueError):
        embed_query(question)


def test_surrounding_whitespace_is_ignored():
    plain = embed_query("database outage")
    padded = embed_query("  \n database outage \t ")

    assert dot(plain, padded) > 0.999


def test_query_uses_instruction_prefix():
    # The query instruction should make the vector differ from embedding the bare text.
    (as_document,) = embed_texts(["database outage"])
    as_query = embed_query("database outage")

    assert dot(as_document, as_query) < 0.999


def test_query_finds_relevant_chunk():
    chunks = [
        "The primary database ran out of connections and the API returned errors.",
        "We updated the marketing site with a new pricing page.",
        "Our office will be closed for the public holiday.",
    ]
    chunk_vectors = embed_texts(chunks)
    query = embed_query("why did the API fail?")

    scores = [dot(query, v) for v in chunk_vectors]
    assert scores.index(max(scores)) == 0


def test_cache_dir_is_inside_project():
    project_root = CACHE_DIR.parents[1]

    assert (project_root / "pyproject.toml").exists()
