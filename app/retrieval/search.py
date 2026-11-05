from collections.abc import Callable
from dataclasses import dataclass

import psycopg

from app.config import settings
from app.embeddings import embed_query


@dataclass
class Hit:
    content: str
    title: str
    company: str | None
    url: str
    chunk_index: int
    score: float  # cosine similarity: 1 = identical, higher is better


# The query vector is sent as a plain list (a float array) and cast with ::vector, so the
# connection doesn't need pgvector's register_vector().
QUERY = """
    SELECT c.content, c.chunk_index, d.title, d.company, d.source_url,
           c.embedding <=> %(q)s::vector AS distance
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    ORDER BY c.embedding <=> %(q)s::vector
    LIMIT %(k)s
"""


def search(
    conn: psycopg.Connection,
    question: str,
    k: int = 5,
    embed: Callable[[str], list[float]] = embed_query,
) -> list[Hit]:
    """Return the k chunks most similar to the question, best first."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    rows = conn.execute(QUERY, {"q": embed(question), "k": k}).fetchall()
    return [
        Hit(
            content=content,
            title=title,
            company=company,
            url=url,
            chunk_index=chunk_index,
            score=1 - distance,
        )
        for content, chunk_index, title, company, url, distance in rows
    ]

if __name__ == "__main__":
    questions = [
        "why did the database failover take so long?",
        "outages caused by DNS",
        "what happens when a config change is rolled out to every server at once?",
        "how did Cloudflare handle the power failure?",
        "deployment that deleted production data",
    ]
    with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
        for q in questions:
            print(f"\n=== {q}")
            for hit in search(conn, q, k=3):
                preview = " ".join(hit.content.split())[:150]
                print(f"{hit.score:.3f}  [{hit.company or '?'}] {hit.title}")
                print(f"       {preview}...")

