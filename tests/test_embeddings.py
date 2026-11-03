import math

from app.embeddings import EMBEDDING_DIM, embed_texts


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
