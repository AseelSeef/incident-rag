import argparse
import logging
from collections.abc import Callable

import psycopg

from app.config import settings
from app.embeddings import embed_texts
from app.ingestion.chunk import chunk_text
from app.ingestion.db import replace_chunks

log = logging.getLogger(__name__)

Embedder = Callable[[list[str]], list[list[float]]]

STALE_SQL = """
    SELECT id, content, updated_at FROM documents
    WHERE indexed_at IS NULL OR updated_at > indexed_at
    ORDER BY id
"""
ALL_SQL = "SELECT id, content, updated_at FROM documents ORDER BY id"

# Records which version was indexed. If the document was updated again since we read it,
# updated_at is now later than this value, so the next run picks it up again.
MARK_INDEXED_SQL = "UPDATE documents SET indexed_at = %s WHERE id = %s"


def index_documents(
    conn: psycopg.Connection,
    embed: Embedder = embed_texts,
    force: bool = False,
) -> int:
    """Chunk and embed documents, storing the chunks. Commits after every document.

    Indexes documents never indexed or changed since they were last indexed
    (updated_at > indexed_at), or every document when `force` is set.
    `embed` turns a list of texts into vectors (swap it out in tests).
    Returns the number of documents indexed.
    """
    documents = conn.execute(ALL_SQL if force else STALE_SQL).fetchall()

    for doc_id, content, version in documents:
        chunks = chunk_text(content)
        vectors = embed(chunks) if chunks else []
        replace_chunks(conn, doc_id, chunks, vectors)
        conn.execute(MARK_INDEXED_SQL, (version, doc_id))
        conn.commit()
        log.info("indexed document %s (%s chunks)", doc_id, len(chunks))

    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk and embed stored documents.")
    parser.add_argument("--force", action="store_true", help="re-index documents already indexed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
        count = index_documents(conn, force=args.force)

    print(f"\nIndexed {count} document(s).")


if __name__ == "__main__":
    main()
