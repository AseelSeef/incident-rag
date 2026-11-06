import math
from itertools import pairwise

import pytest

from app.embeddings import EMBEDDING_DIM
from app.ingestion.db import replace_chunks, upsert_document
from app.ingestion.sources import Source
from app.retrieval.search import Hit, _limit_per_document, search

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


@pytest.mark.parametrize("max_per_doc", [0, -1])
def test_max_per_doc_must_be_positive(conn, max_per_doc):
    with pytest.raises(ValueError, match="max_per_doc must be greater than 0"):
        search(conn, "anything", max_per_doc=max_per_doc, embed=query_vector)


# --- per-document cap ---


def test_search_caps_chunks_per_document(docs):
    # Document a has two chunks (1.0 and 0.8); with a cap of 1 its second chunk is dropped
    # and document b moves up.
    hits = search(docs, "anything", k=2, max_per_doc=1, embed=query_vector)

    assert [h.content for h in hits] == ["search-test exact match", "search-test related"]


def test_default_cap_keeps_two_chunks_per_document(docs):
    hits = search(docs, "anything", k=3, embed=query_vector)

    assert [h.url for h in hits[:2]] == ["https://search.test/a"] * 2


@pytest.mark.parametrize("k, expected", [(5, "40"), (10, "40"), (20, "80")])
def test_hnsw_candidate_limit_is_raised_for_large_k(docs, k, expected):
    # The index returns at most hnsw.ef_search rows; k * 4 candidates must fit.
    search(docs, "anything", k=k, embed=query_vector)

    assert docs.execute("SHOW hnsw.ef_search").fetchone()[0] == expected


def test_hnsw_setting_does_not_outlive_the_transaction(conn):
    search(conn, "anything", k=20, embed=query_vector)
    conn.rollback()

    assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == "40"


def chunk(url: str, n: int) -> Hit:
    return Hit(content=f"{url}-{n}", title="t", company="c", url=url, chunk_index=n, score=0.5)


def contents(hits: list[Hit]) -> list[str]:
    return [h.content for h in hits]


def test_limit_skips_extra_chunks_of_a_document():
    hits = [chunk("a", 0), chunk("a", 1), chunk("a", 2), chunk("b", 0), chunk("a", 3)]

    assert contents(_limit_per_document(hits, k=5, max_per_doc=2)) == ["a-0", "a-1", "b-0"]


def test_limit_keeps_the_original_order():
    hits = [chunk("b", 0), chunk("a", 0), chunk("b", 1), chunk("c", 0)]

    assert contents(_limit_per_document(hits, k=4, max_per_doc=2)) == ["b-0", "a-0", "b-1", "c-0"]


def test_limit_stops_at_k():
    hits = [chunk("a", 0), chunk("b", 0), chunk("c", 0), chunk("d", 0)]

    assert contents(_limit_per_document(hits, k=2, max_per_doc=2)) == ["a-0", "b-0"]


def test_limit_of_one_gives_one_chunk_per_document():
    hits = [chunk("a", 0), chunk("a", 1), chunk("b", 0), chunk("b", 1), chunk("c", 0)]

    assert contents(_limit_per_document(hits, k=5, max_per_doc=1)) == ["a-0", "b-0", "c-0"]


def test_limit_returns_fewer_than_k_when_documents_run_out():
    hits = [chunk("a", 0), chunk("a", 1), chunk("a", 2)]

    assert contents(_limit_per_document(hits, k=5, max_per_doc=2)) == ["a-0", "a-1"]


def test_limit_of_nothing_is_nothing():
    assert _limit_per_document([], k=5, max_per_doc=2) == []
