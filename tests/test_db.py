from datetime import date

from app.ingestion.db import existing_urls, upsert_document
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
