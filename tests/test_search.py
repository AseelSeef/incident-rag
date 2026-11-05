import math
from itertools import pairwise

import pytest

from app.embeddings import EMBEDDING_DIM
from app.ingestion.db import replace_chunks, upsert_document
from app.ingestion.sources import Source
from app.retrieval.search import Hit, search

# The fake query vector points along the first axis. A chunk stored with vec_with_score(s)
# has cosine similarity exactly s with it. Real chunks in your dev database are spread over
# all 384 axes, so their similarity to this axis is small (well under 0.5); the test chunks
# use 0.6 and above, so they always rank first. Everything is rolled back after each test.


def query_vector(question: str) -> list[float]:
    return [1.0] + [0.0] * (EMBEDDING_DIM - 1)


def vec_with_score(s: float) -> list[float]:
    return [s, math.sqrt(1 - s * s)] + [0.0] * (EMBEDDING_DIM - 2)


def add_document(conn, url: str, chunks: list[tuple[str, float]], **source_fields) -> int:
    fields = {"title": "Database outage", "company": "Acme"} | source_fields
    doc_id = upsert_document(conn, Source(url=url, **fields), "content")
    replace_chunks(conn, doc_id, [text for text, _ in chunks], [vec_with_score(s) for _, s in chunks])
    return doc_id


@pytest.fixture
def docs(conn):
    # Use an exact scan instead of the approximate HNSW index. Rolled-back test runs leave dead
    # test vectors in the index until VACUUM; they crowd out the live rows, so an index scan can
    # return fewer than k hits. These tests check search(), not the index. (SET LOCAL ends with
    # the test's transaction.)
    conn.execute("SET LOCAL enable_indexscan = off")
    add_document(
        conn,
        "https://search.test/a",
        [("search-test exact match", 1.0), ("search-test close", 0.8)],
    )
    add_document(
        conn,
        "https://search.test/b",
        [("search-test related", 0.6)],
        title="Network outage",
        company="Globex",
    )
    return conn


def test_best_match_first(docs):
    hits = search(docs, "anything", k=3, embed=query_vector)

    assert [h.content for h in hits] == [
        "search-test exact match",
        "search-test close",
        "search-test related",
    ]


def test_scores_are_cosine_similarity(docs):
    hits = search(docs, "anything", k=3, embed=query_vector)

    assert [h.score for h in hits] == pytest.approx([1.0, 0.8, 0.6], abs=1e-4)


def test_scores_never_increase(docs):
    hits = search(docs, "anything", k=10, embed=query_vector)

    assert all(a.score >= b.score for a, b in pairwise(hits))


def test_k_limits_results(docs):
    assert len(search(docs, "anything", k=1, embed=query_vector)) == 1
    assert len(search(docs, "anything", k=2, embed=query_vector)) == 2


def test_hit_fields_come_from_chunk_and_document(docs):
    hits = search(docs, "anything", k=3, embed=query_vector)

    assert hits[1] == Hit(
        content="search-test close",
        title="Database outage",
        company="Acme",
        url="https://search.test/a",
        chunk_index=1,
        score=pytest.approx(0.8, abs=1e-4),
    )
    assert (hits[2].title, hits[2].company, hits[2].url, hits[2].chunk_index) == (
        "Network outage",
        "Globex",
        "https://search.test/b",
        0,
    )


def test_score_is_a_plain_float(docs):
    (hit,) = search(docs, "anything", k=1, embed=query_vector)

    assert type(hit.score) is float


def test_question_is_passed_to_embed(docs):
    seen = []

    def recording_embed(question: str) -> list[float]:
        seen.append(question)
        return query_vector(question)

    search(docs, "why did the database go down?", embed=recording_embed)

    assert seen == ["why did the database go down?"]


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(conn, k):
    def must_not_embed(question: str) -> list[float]:
        raise AssertionError("embed should not be called for invalid k")

    with pytest.raises(ValueError, match="k must be greater than 0"):
        search(conn, "anything", k=k, embed=must_not_embed)
