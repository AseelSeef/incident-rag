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
        updated_at = now()
    RETURNING id
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
