from datetime import date

import pytest

from app.embeddings import EMBEDDING_DIM
from app.ingestion.db import existing_urls, replace_chunks, update_metadata, upsert_document
from app.ingestion.sources import Source


def make_source(**overrides) -> Source:
    fields = {
        "title": "Database outage",
        "company": "Acme",
        "url": "https://example.com/test-postmortem",
        "date": date(2024, 1, 2),
    }
    fields.update(overrides)
    return Source(**fields)


def fetch_row(conn, doc_id: int):
    return conn.execute(
        "SELECT title, source_url, company, published, content FROM documents WHERE id = %s",
        (doc_id,),
    ).fetchone()


def test_insert_stores_all_fields(conn):
    doc_id = upsert_document(conn, make_source(), "the content")

    assert fetch_row(conn, doc_id) == (
        "Database outage",
        "https://example.com/test-postmortem",
        "Acme",
        date(2024, 1, 2),
        "the content",
    )


def test_same_url_returns_same_id(conn):
    first = upsert_document(conn, make_source(), "v1")
    second = upsert_document(conn, make_source(), "v2")

    assert first == second


def test_same_url_updates_fields(conn):
    doc_id = upsert_document(conn, make_source(), "old content")
    upsert_document(
        conn,
        make_source(title="Updated title", company="Acme Inc", date=date(2024, 2, 3)),
        "new content",
    )

    title, _, company, published, content = fetch_row(conn, doc_id)
    assert (title, company, published, content) == (
        "Updated title",
        "Acme Inc",
        date(2024, 2, 3),
        "new content",
    )


def test_same_url_does_not_duplicate_rows(conn):
    url = "https://example.com/test-postmortem"
    upsert_document(conn, make_source(url=url), "v1")
    upsert_document(conn, make_source(url=url), "v2")

    (count,) = conn.execute(
        "SELECT count(*) FROM documents WHERE source_url = %s", (url,)
    ).fetchone()
    assert count == 1


def test_update_bumps_updated_at(conn):
    doc_id = upsert_document(conn, make_source(), "v1")
    # now() is fixed for the whole transaction, so backdate the row to see the change.
    conn.execute(
        "UPDATE documents SET updated_at = now() - interval '1 day' WHERE id = %s", (doc_id,)
    )
    upsert_document(conn, make_source(), "v2")

    (age_ok,) = conn.execute(
        "SELECT updated_at >= now() - interval '1 minute' FROM documents WHERE id = %s",
        (doc_id,),
    ).fetchone()
    assert age_ok


def backdate(conn, doc_id: int) -> None:
    """now() is fixed for the whole transaction, so move updated_at back to see any change."""
    conn.execute(
        "UPDATE documents SET updated_at = now() - interval '1 day' WHERE id = %s", (doc_id,)
    )


def updated_at(conn, doc_id: int):
    (value,) = conn.execute(
        "SELECT updated_at FROM documents WHERE id = %s", (doc_id,)
    ).fetchone()
    return value


def test_same_content_keeps_updated_at(conn):
    doc_id = upsert_document(conn, make_source(), "same text")
    backdate(conn, doc_id)
    before = updated_at(conn, doc_id)

    upsert_document(conn, make_source(), "same text")

    assert updated_at(conn, doc_id) == before


def test_company_or_date_change_keeps_updated_at(conn):
    # Saved, but not an edit: company and date aren't embedded, so no re-index is needed.
    doc_id = upsert_document(conn, make_source(), "same text")
    backdate(conn, doc_id)
    before = updated_at(conn, doc_id)

    upsert_document(conn, make_source(company="Acme Inc", date=date(2020, 5, 6)), "same text")

    _, _, company, published, _ = fetch_row(conn, doc_id)
    assert (company, published) == ("Acme Inc", date(2020, 5, 6))
    assert updated_at(conn, doc_id) == before


def test_title_change_bumps_updated_at(conn):
    # Each chunk is embedded with its title, so a new title makes the embeddings stale.
    doc_id = upsert_document(conn, make_source(), "same text")
    backdate(conn, doc_id)
    before = updated_at(conn, doc_id)

    upsert_document(conn, make_source(title="Renamed"), "same text")

    assert fetch_row(conn, doc_id)[0] == "Renamed"
    assert updated_at(conn, doc_id) > before


# --- update_metadata: details from the CSV, without downloading again ---


def test_update_metadata_sets_a_new_date(conn):
    doc_id = upsert_document(conn, make_source(date=None), "the content")

    changed = update_metadata(conn, make_source(date=date(2021, 10, 4)))

    assert changed is True
    assert fetch_row(conn, doc_id)[3] == date(2021, 10, 4)


def test_update_metadata_leaves_the_content_alone(conn):
    doc_id = upsert_document(conn, make_source(), "the content")

    update_metadata(conn, make_source(title="Renamed", company="Other"))

    assert fetch_row(conn, doc_id)[1:] == (make_source().url, "Other", date(2024, 1, 2), "the content")


def test_update_metadata_without_changes_returns_false(conn):
    doc_id = upsert_document(conn, make_source(), "the content")
    backdate(conn, doc_id)
    before = updated_at(conn, doc_id)

    assert update_metadata(conn, make_source()) is False
    assert updated_at(conn, doc_id) == before


def test_update_metadata_for_unknown_url_returns_false(conn):
    assert update_metadata(conn, make_source(url="https://example.com/not-stored")) is False


def test_update_metadata_can_clear_a_date(conn):
    doc_id = upsert_document(conn, make_source(), "the content")

    assert update_metadata(conn, make_source(date=None)) is True
    assert fetch_row(conn, doc_id)[3] is None


def test_update_metadata_date_change_keeps_updated_at(conn):
    doc_id = upsert_document(conn, make_source(), "the content")
    backdate(conn, doc_id)
    before = updated_at(conn, doc_id)

    update_metadata(conn, make_source(date=date(2020, 1, 1)))

    assert updated_at(conn, doc_id) == before


def test_update_metadata_title_change_bumps_updated_at(conn):
    doc_id = upsert_document(conn, make_source(), "the content")
    backdate(conn, doc_id)
    before = updated_at(conn, doc_id)

    update_metadata(conn, make_source(title="Renamed"))

    assert updated_at(conn, doc_id) > before


def test_different_urls_get_different_ids(conn):
    a = upsert_document(conn, make_source(url="https://example.com/a"), "a")
    b = upsert_document(conn, make_source(url="https://example.com/b"), "b")

    assert a != b


def test_missing_date_stored_as_null(conn):
    doc_id = upsert_document(conn, make_source(date=None), "content")

    assert fetch_row(conn, doc_id)[3] is None


def test_does_not_commit(conn):
    url = "https://example.com/uncommitted"
    upsert_document(conn, make_source(url=url), "content")
    conn.rollback()

    assert conn.execute(
        "SELECT 1 FROM documents WHERE source_url = %s", (url,)
    ).fetchone() is None


def test_existing_urls_includes_stored_documents(conn):
    upsert_document(conn, make_source(url="https://example.com/a"), "a")
    upsert_document(conn, make_source(url="https://example.com/b"), "b")

    assert {"https://example.com/a", "https://example.com/b"} <= existing_urls(conn)


def test_existing_urls_excludes_unknown_url(conn):
    assert "https://example.com/never-stored" not in existing_urls(conn)


# --- replace_chunks ---


def vec(value: float) -> list[float]:
    """A 384-dim test vector whose first number identifies it."""
    return [float(value)] + [0.0] * (EMBEDDING_DIM - 1)


def stored_chunks(conn, doc_id: int):
    """(chunk_index, content, first number of the embedding) for each chunk, in index order."""
    return conn.execute(
        "SELECT chunk_index, content, embedding[1] FROM ("
        "  SELECT chunk_index, content, embedding::real[] AS embedding"
        "  FROM chunks WHERE document_id = %s"
        ") c ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()


def test_replace_chunks_stores_in_order(conn):
    doc_id = upsert_document(conn, make_source(), "content")

    replace_chunks(conn, doc_id, ["first", "second", "third"], [vec(1), vec(2), vec(3)])

    assert stored_chunks(conn, doc_id) == [(0, "first", 1.0), (1, "second", 2.0), (2, "third", 3.0)]


def test_replace_chunks_removes_old_chunks(conn):
    doc_id = upsert_document(conn, make_source(), "content")
    replace_chunks(conn, doc_id, ["old a", "old b", "old c"], [vec(1), vec(2), vec(3)])

    replace_chunks(conn, doc_id, ["new a"], [vec(9)])

    assert stored_chunks(conn, doc_id) == [(0, "new a", 9.0)]


def test_replace_chunks_with_empty_list_clears_document(conn):
    doc_id = upsert_document(conn, make_source(), "content")
    replace_chunks(conn, doc_id, ["a", "b"], [vec(1), vec(2)])

    replace_chunks(conn, doc_id, [], [])

    assert stored_chunks(conn, doc_id) == []


def test_replace_chunks_leaves_other_documents_alone(conn):
    doc_a = upsert_document(conn, make_source(url="https://example.com/a"), "a")
    doc_b = upsert_document(conn, make_source(url="https://example.com/b"), "b")
    replace_chunks(conn, doc_b, ["b chunk"], [vec(5)])

    replace_chunks(conn, doc_a, ["a chunk"], [vec(1)])
    replace_chunks(conn, doc_a, ["a chunk v2"], [vec(2)])

    assert stored_chunks(conn, doc_b) == [(0, "b chunk", 5.0)]


@pytest.mark.parametrize("n_chunks, n_embeddings", [(2, 1), (1, 2), (0, 1)])
def test_replace_chunks_mismatched_lengths_raise(conn, n_chunks, n_embeddings):
    doc_id = upsert_document(conn, make_source(), "content")
    replace_chunks(conn, doc_id, ["keep me"], [vec(1)])

    with pytest.raises(ValueError, match="chunks but"):
        replace_chunks(conn, doc_id, ["x"] * n_chunks, [vec(0)] * n_embeddings)

    # Checked before deleting, so the existing chunks survive.
    assert stored_chunks(conn, doc_id) == [(0, "keep me", 1.0)]


def test_replace_chunks_does_not_commit(conn):
    doc_id = upsert_document(conn, make_source(), "content")
    replace_chunks(conn, doc_id, ["a"], [vec(1)])
    conn.rollback()

    (count,) = conn.execute(
        "SELECT count(*) FROM chunks WHERE document_id = %s", (doc_id,)
    ).fetchone()
    assert count == 0


def test_deleting_document_deletes_its_chunks(conn):
    doc_id = upsert_document(conn, make_source(), "content")
    replace_chunks(conn, doc_id, ["a", "b"], [vec(1), vec(2)])

    conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))

    assert stored_chunks(conn, doc_id) == []


def test_chunks_embedding_size_matches_model(conn):
    # vector(n) in db/init.sql must match the embedding model; for vector columns atttypmod is n.
    (dim,) = conn.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()
    assert dim == EMBEDDING_DIM
