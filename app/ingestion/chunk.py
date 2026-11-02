import re


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """(start, end) of each non-blank line in `text`, with surrounding spaces trimmed."""
    spans = []
    for m in re.finditer(r"[^\n]+", text):
        line = m.group()
        stripped = line.strip()
        if not stripped:
            continue  # blank or whitespace-only line
        start = m.start() + (len(line) - len(line.lstrip()))
        spans.append((start, start + len(stripped)))
    return spans


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> list[str]:
    if max_chars <= 0:
        raise ValueError(f"max_chars must be > 0, got {max_chars}")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError(f"overlap must be >= 0 and < {max_chars}, got {overlap}")
    spans = _paragraphs(text)
    # Pieces leave `overlap` chars of room, so the next chunk can repeat the end of the previous one.
    piece_max = max_chars - overlap
    spans = [piece for s, e in spans for piece in _split_long(text, s, e, piece_max)]
    if not spans:
        return []
    chunks = []
    chunk_start, chunk_end = spans[0]
    for start, end in spans[1:]:
        if end - chunk_start <= max_chars:
            chunk_end = end
        else:
            chunks.append(text[chunk_start:chunk_end])
            chunk_start = _overlap_start(text, chunk_start, chunk_end, start, end, max_chars, overlap)
            chunk_end = end

    chunks.append(text[chunk_start:chunk_end])
    return chunks


def _split_long(text: str, s: int, e: int, max_chars: int) -> list[tuple[int, int]]:
    pieces = []
    while e - s > max_chars:
        cut = text.rfind(" ", s, s + max_chars + 1)
        if cut <= s:
            cut = s + max_chars
            pieces.append((s, cut))
            s = cut
        else:
            pieces.append((s, cut))
            s = cut + 1
    pieces.append((s, e))
    return pieces


def _overlap_start(
    text: str,
    prev_start: int,
    prev_end: int,
    start: int,
    end: int,
    max_chars: int,
    overlap: int,
) -> int:
    """Where the next chunk should begin so it repeats up to `overlap` chars of the previous one.

    `prev_start:prev_end` is the chunk just saved; `start:end` is the piece starting the next one.
    Always returns a word start between `prev_start + 1` and `start`.
    """
    if overlap == 0:
        return start

    # 1. Go back `overlap` chars from the previous chunk's end,
    # 2. but keep the new chunk within max_chars and don't repeat the whole previous chunk.
    new_start = max(prev_end - overlap, end - max_chars, prev_start + 1)
    if new_start >= start:
        return start

    # 3. Move forward to the start of a word.
    if text[new_start - 1].isspace():
        return new_start
    m = re.compile(r"\s+").search(text, new_start, start)
    if m is None:
        return start  # no word boundary in range (e.g. after a hard cut): no overlap
    return m.end()
