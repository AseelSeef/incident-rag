from functools import cache

from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # must match vector(384) in db/init.sql


@cache
def _model() -> TextEmbedding:
    """Load the model once (the first call downloads it, ~130 MB) and reuse it."""
    return TextEmbedding(EMBEDDING_MODEL,cache_dir=".cache/fastembed")


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed document texts (e.g. chunks). Returns one 384-dim, unit-length vector per text,
    in the same order as `texts`."""
    if not texts:
        return []
    return [vector.tolist() for vector in _model().embed(texts, batch_size=batch_size)]
