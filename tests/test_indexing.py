import pytest

from app.embeddings import EMBEDDING_DIM
from app.indexing.run import embedding_text, index_documents
from app.ingestion.db import replace_chunks, upsert_document
from app.ingestion.sources import Source

# Your dev database may already hold real documents. Every test runs inside the rolled-back
# `conn` transaction with commit disabled, so they are never changed; assertions only look at
# the documents each test creates, or compare counts before and after.

LONG_TEXT = "\n".join(
    f"Paragraph {i}: the database connection pool was exhausted and requests failed. " * 3
    for i in range(10)
)  # several chunks with the default chunk size


class FakeEmbedder:
    """Stands in for embed_texts: records what it was asked to embed, returns float vectors."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(self.calls))] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]

    def embedded(self) -> list[str]:
        return [t for call in self.calls for t in call]


@pytest.fixture
def db(conn, monkeypatch):
    """The rollback connection, with commit counted instead of done.

    Documents already in the database are marked as indexed so tests only index the
    documents they create; otherwise every test would index all your real documents.
    """
    conn.execute("UPDATE documents SET indexed_at = updated_at")
    commits = []
    monkeypatch.setattr(conn, "commit", lambda: commits.append(1))
    conn.commits = commits
    return conn


@pytest.fixture
def embed():
    return FakeEmbedder()


def add_document(conn, url: str, content: str = LONG_TEXT) -> int:
    return upsert_document(conn, Source(title="Outage", company="Acme", url=url), content)


def chunk_rows(conn, doc_id: int):
    return conn.execute(
        "SELECT chunk_index, content FROM chunks WHERE document_id = %s ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()


def unindexed_count(conn) -> int:
    (n,) = conn.execute(
        "SELECT count(*) FROM documents WHERE indexed_at IS NULL OR updated_at > indexed_at"
    ).fetchone()
    return n


def timestamps(conn, doc_id: int):
    return conn.execute(
        "SELECT updated_at, indexed_at FROM documents WHERE id = %s", (doc_id,)
    ).fetchone()


def simulate_later_update(conn, doc_id: int, new_content: str) -> None:
    """What re-ingesting changed text does. now() is fixed inside one transaction,
    so move updated_at forward by hand to stand in for 'later'."""
    conn.execute(
        "UPDATE documents SET content = %s, updated_at = updated_at + interval '1 minute' "
        "WHERE id = %s",
        (new_content, doc_id),
    )


def test_indexes_new_document(db, embed):
    doc_id = add_document(db, "https://index.test/a")

    index_documents(db, embed=embed)

    rows = chunk_rows(db, doc_id)
    assert len(rows) > 1
    assert [i for i, _ in rows] == list(range(len(rows)))
    assert all(text in LONG_TEXT for _, text in rows)


def test_each_chunk_is_embedded_with_the_title(db, embed):
    doc_id = add_document(db, "https://index.test/a")

    index_documents(db, embed=embed)

    stored = [text for _, text in chunk_rows(db, doc_id)]
    assert embed.calls[-1] == [f"Outage\n\n{text}" for text in stored]


def test_stored_content_has_no_title(db, embed):
    doc_id = add_document(db, "https://index.test/a")

    index_documents(db, embed=embed)

    assert not any(text.startswith("Outage") for _, text in chunk_rows(db, doc_id))


def test_embedding_text_puts_the_title_first():
    assert embedding_text("AWS S3 outage", "The command removed servers.") == (
        "AWS S3 outage\n\nThe command removed servers."
    )


@pytest.mark.parametrize("title", [None, ""])
def test_embedding_text_without_a_title_is_the_chunk(title):
    assert embedding_text(title, "The command removed servers.") == "The command removed servers."


def test_returns_number_of_documents_indexed(db, embed):
    add_document(db, "https://index.test/a")
    add_document(db, "https://index.test/b")
    expected = unindexed_count(db)

    assert index_documents(db, embed=embed) == expected
    assert unindexed_count(db) == 0


def test_commits_once_per_document(db, embed):
    add_document(db, "https://index.test/a")

    count = index_documents(db, embed=embed)

    assert len(db.commits) == count


def test_already_indexed_document_is_skipped(db, embed):
    doc_id = add_document(db, "https://index.test/a")
    index_documents(db, embed=embed)
    embed.calls.clear()

    count = index_documents(db, embed=embed)

    assert count == 0
    assert embed.calls == []
    assert len(chunk_rows(db, doc_id)) > 1


def test_force_reindexes_indexed_document(db, embed):
    doc_id = add_document(db, "https://index.test/a")
    replace_chunks(db, doc_id, ["stale chunk"], [[0.0] * EMBEDDING_DIM])

    index_documents(db, embed=embed, force=True)

    texts = [text for _, text in chunk_rows(db, doc_id)]
    assert "stale chunk" not in texts
    assert len(texts) > 1


def test_indexing_sets_indexed_at_to_updated_at(db, embed):
    doc_id = add_document(db, "https://index.test/a")
    assert timestamps(db, doc_id)[1] is None

    index_documents(db, embed=embed)

    updated_at, indexed_at = timestamps(db, doc_id)
    assert indexed_at == updated_at


def test_changed_document_is_reindexed(db, embed):
    doc_id = add_document(db, "https://index.test/a")
    index_documents(db, embed=embed)

    simulate_later_update(db, doc_id, "The new root cause was an expired TLS certificate.")
    count = index_documents(db, embed=embed)

    assert count == 1
    assert chunk_rows(db, doc_id) == [(0, "The new root cause was an expired TLS certificate.")]
    updated_at, indexed_at = timestamps(db, doc_id)
    assert indexed_at == updated_at


def test_reingest_with_same_timestamp_is_not_reindexed(db, embed):
    # upsert_document sets updated_at = now(); inside one transaction that equals indexed_at,
    # so only an update that happens later counts as a change.
    doc_id = add_document(db, "https://index.test/a")
    index_documents(db, embed=embed)
    embed.calls.clear()

    add_document(db, "https://index.test/a")

    assert index_documents(db, embed=embed) == 0
    assert len(chunk_rows(db, doc_id)) > 1


def test_never_indexed_document_with_old_chunks_is_reindexed(db, embed):
    # Chunks without indexed_at (e.g. stored before this column existed) are replaced.
    doc_id = add_document(db, "https://index.test/a")
    replace_chunks(db, doc_id, ["stale chunk"], [[0.0] * EMBEDDING_DIM])

    index_documents(db, embed=embed)

    texts = [text for _, text in chunk_rows(db, doc_id)]
    assert "stale chunk" not in texts
    assert len(texts) > 1


def test_short_document_gets_one_chunk(db, embed):
    doc_id = add_document(db, "https://index.test/short", "A short postmortem.")

    index_documents(db, embed=embed)

    assert chunk_rows(db, doc_id) == [(0, "A short postmortem.")]


def test_blank_document_gets_no_chunks_and_no_embed_call(db, embed):
    doc_id = add_document(db, "https://index.test/blank", "   \n  ")

    index_documents(db, embed=embed)

    assert chunk_rows(db, doc_id) == []
    assert "" not in embed.embedded()


def test_nothing_to_do_returns_zero(db, embed):
    index_documents(db, embed=embed)  # index whatever is there
    embed.calls.clear()

    assert index_documents(db, embed=embed) == 0
    assert embed.calls == []
