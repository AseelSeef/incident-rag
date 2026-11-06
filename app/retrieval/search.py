from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

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
    published: date | None = None  # the document's date, if data/sources.csv has one


# The query vector is sent as a plain list (a float array) and cast with ::vector, so the
# connection doesn't need pgvector's register_vector().
QUERY = """
    SELECT c.content, c.chunk_index, d.title, d.company, d.source_url, d.published,
           c.embedding <=> %(q)s::vector AS distance
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    ORDER BY c.embedding <=> %(q)s::vector
    LIMIT %(limit)s
"""

# Fetch k * 4 chunks, so enough are left after the per-document cap drops some.
CANDIDATES_PER_RESULT = 4


def search(
    conn: psycopg.Connection,
    question: str,
    k: int = 5,
    max_per_doc: int = 2,
    embed: Callable[[str], list[float]] = embed_query,
) -> list[Hit]:
    """Return up to k chunks most similar to the question, best first.

    At most max_per_doc chunks come from any one document while other documents can fill the
    slots, so one long postmortem can't crowd out the rest. Slots no other document can fill
    go to more chunks of the same documents. Fewer than k are returned only if there are
    fewer than k candidates in total.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if max_per_doc <= 0:
        raise ValueError("max_per_doc must be greater than 0")
    limit = k * CANDIDATES_PER_RESULT
    # The HNSW index returns at most hnsw.ef_search rows (default 40), whatever LIMIT says.
    # set_config(..., true) applies to this transaction only, like SET LOCAL; plain SET
    # can't take a query parameter.
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(limit, 40)),))
    rows = conn.execute(QUERY, {"q": embed(question), "limit": limit}).fetchall()
    hits = [
        Hit(
            content=content,
            title=title,
            company=company,
            url=url,
            chunk_index=chunk_index,
            score=1 - distance,
            published=published,
        )
        for content, chunk_index, title, company, url, published, distance in rows
    ]
    return _limit_per_document(hits, k, max_per_doc)


def _limit_per_document(hits: list[Hit], k: int, max_per_doc: int) -> list[Hit]:
    """Keep the best hits, at most max_per_doc from each document, up to k in total.

    If that leaves fewer than k, the free slots go to the best skipped hits. The result is
    sorted best first.
    """
    kept: list[Hit] = []
    counts: dict[str, int] = {}
    skipped: list[Hit] = []
    for hit in hits:
        if counts.get(hit.url, 0) >= max_per_doc:
            skipped.append(hit)
            continue
        counts[hit.url] = counts.get(hit.url, 0) + 1
        kept.append(hit)
        if len(kept) == k:
            break
    if len(kept) < k:
        kept.extend(skipped[:k - len(kept)])
    return sorted(kept, key=lambda hit: hit.score, reverse=True)


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

