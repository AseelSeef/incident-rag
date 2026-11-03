from functools import cache
from pathlib import Path

from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # must match vector(384) in db/init.sql
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Inside the project folder, wherever Python is run from (.cache/ is in .gitignore).
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "fastembed"


@cache
def _model() -> TextEmbedding:
    """Load the model once (the first call downloads it, ~130 MB) and reuse it."""
    return TextEmbedding(EMBEDDING_MODEL, cache_dir=str(CACHE_DIR))


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed document texts (e.g. chunks). Returns one 384-dim, unit-length vector per text,
    in the same order as `texts`."""
    if not texts:
        return []
    return [vector.tolist() for vector in _model().embed(texts, batch_size=batch_size)]


def embed_query(question: str) -> list[float]:
    """Embed a search question (with the bge query instruction). Returns one 384-dim vector."""
    question = question.strip()
    if not question:
        raise ValueError("empty question")
    return embed_texts([QUERY_PREFIX + question])[0]