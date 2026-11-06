import psycopg

from app.ingestion.sources import Source

UPSERT_SQL = """
    INSERT INTO documents (title, source_url, company, published, content)
    VALUES (%(title)s, %(url)s, %(company)s, %(published)s, %(content)s)
    ON CONFLICT (source_url) DO UPDATE SET
        title      = EXCLUDED.title,
        company    = EXCLUDED.company,
        published  = EXCLUDED.published,
        content    = EXCLUDED.content,
        -- Only a change to what gets embedded counts as an edit, so unchanged documents aren't
        -- re-indexed: the text, and the title (each chunk is embedded with it).
        updated_at = CASE
            WHEN documents.content IS DISTINCT FROM EXCLUDED.content
              OR documents.title IS DISTINCT FROM EXCLUDED.title THEN now()
            ELSE documents.updated_at
        END
    RETURNING id
"""

# Updates the CSV details of a stored document without downloading it again. Only rows whose
# details differ are touched. In SET, `title` still means the old value, so a new title marks
# the document for re-indexing; company and date aren't embedded, so they don't.
UPDATE_METADATA_SQL = """
    UPDATE documents SET
        title      = %(title)s,
        company    = %(company)s,
        published  = %(published)s,
        updated_at = CASE WHEN title IS DISTINCT FROM %(title)s THEN now() ELSE updated_at END
    WHERE source_url = %(url)s
      AND (title, company, published)
          IS DISTINCT FROM (%(title)s::text, %(company)s::text, %(published)s::date)
"""


def upsert_document(conn: psycopg.Connection, source: Source, content: str) -> int:
    """Insert or update a document by URL. Returns the document id.

    Does not commit; the caller owns the transaction.
    """
    row = conn.execute(
        UPSERT_SQL,
        {
            "title": source.title,
            "url": source.url,
            "company": source.company,
            "published": source.date,
            "content": content,
        },
    ).fetchone()
    return row[0]


def update_metadata(conn: psycopg.Connection, source: Source) -> bool:
    """Copy title, company and date from the source to its stored document, if they changed.

    Returns True if the document was updated. Does not commit.
    """
    cursor = conn.execute(
        UPDATE_METADATA_SQL,
        {
            "title": source.title,
            "url": source.url,
            "company": source.company,
            "published": source.date,
        },
    )
    return cursor.rowcount == 1


def existing_urls(conn: psycopg.Connection) -> set[str]:
    """Return the source URLs of all documents already stored."""
    rows = conn.execute("SELECT source_url FROM documents").fetchall()
    return {url for (url,) in rows}


def replace_chunks(conn: psycopg.Connection, document_id: int, chunks: list[str],
                   embeddings: list[list[float]], ) -> None:
    """Replace all chunks of a document. Does not commit; the caller owns the transaction."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"got {len(chunks)} chunks but {len(embeddings)} embeddings")
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding)"
            " VALUES (%s, %s, %s, %s)",
            [(document_id, i, text, vec) for i, (text, vec) in enumerate(zip(chunks, embeddings))],
        )
