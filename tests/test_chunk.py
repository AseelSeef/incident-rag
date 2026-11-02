import pytest

from app.ingestion.chunk import chunk_text


def numbered_words(n: int) -> str:
    """Text of n unique words ("w0 w1 w2 ..."), so each word can be traced to a chunk."""
    return " ".join(f"w{i}" for i in range(n))


# --- small inputs ---


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_whitespace_only_returns_no_chunks():
    assert chunk_text("  \n\n \t ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Database outage at 14:02 UTC.", max_chars=100, overlap=0) == [
        "Database outage at 14:02 UTC."
    ]


def test_short_text_is_stripped():
    assert chunk_text("\n  Database outage.  \n", max_chars=100, overlap=0) == [
        "Database outage."
    ]


def test_text_exactly_max_chars_is_one_chunk():
    text = "a" * 100

    assert chunk_text(text, max_chars=100, overlap=0) == [text]


# --- long inputs ---


def test_long_text_gives_several_chunks():
    chunks = chunk_text(numbered_words(500), max_chars=200, overlap=20)

    assert len(chunks) > 1


def test_no_chunk_exceeds_max_chars():
    chunks = chunk_text(numbered_words(500), max_chars=200, overlap=20)

    assert all(len(c) <= 200 for c in chunks)


def test_no_empty_chunks():
    text = "\n\n".join(numbered_words(60) for _ in range(5))

    chunks = chunk_text(text, max_chars=200, overlap=20)

    assert all(c.strip() for c in chunks)


def test_every_word_appears_in_some_chunk():
    text = numbered_words(500)

    chunks = chunk_text(text, max_chars=200, overlap=20)

    found = {word for c in chunks for word in c.split()}
    assert found == set(text.split())


def test_chunks_are_in_order():
    chunks = chunk_text(numbered_words(500), max_chars=200, overlap=0)

    first_numbers = [int(c.split()[0][1:]) for c in chunks]
    assert first_numbers == sorted(first_numbers)


def test_chunks_are_verbatim_from_text():
    text = numbered_words(500)

    chunks = chunk_text(text, max_chars=200, overlap=20)

    assert all(c in text for c in chunks)


def test_words_are_not_cut_in_half():
    text = numbered_words(500)
    vocabulary = set(text.split())

    chunks = chunk_text(text, max_chars=200, overlap=20)

    assert all(word in vocabulary for c in chunks for word in c.split())


# --- overlap ---


def test_consecutive_chunks_overlap():
    chunks = chunk_text(numbered_words(500), max_chars=200, overlap=50)

    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(prev.split()) & set(nxt.split()), f"no overlap between {prev!r} and {nxt!r}"


def test_zero_overlap_shares_no_words():
    chunks = chunk_text(numbered_words(500), max_chars=200, overlap=0)

    for prev, nxt in zip(chunks, chunks[1:]):
        assert not set(prev.split()) & set(nxt.split())


# --- boundaries ---


# trafilatura separates paragraphs with a single "\n"; "\n\n" is checked too for other sources.
PARAGRAPH_SEPARATORS = pytest.mark.parametrize("sep", ["\n", "\n\n"], ids=["single", "double"])


@PARAGRAPH_SEPARATORS
def test_splits_on_paragraph_boundary_when_possible(sep):
    para_a = ("Alpha " * 50).strip()  # 299 chars
    para_b = ("Bravo " * 50).strip()
    text = para_a + sep + para_b

    chunks = chunk_text(text, max_chars=400, overlap=0)

    assert chunks == [para_a, para_b]


@PARAGRAPH_SEPARATORS
def test_short_paragraphs_are_packed_together(sep):
    p1 = "First paragraph here."  # 21 chars
    p2 = "Second paragraph here."  # 22 chars
    p3 = "Third paragraph here."  # 21 chars
    text = f"{p1}{sep}{p2}{sep}{p3}"  # 66 or 68 chars

    chunks = chunk_text(text, max_chars=50, overlap=0)

    assert chunks == [f"{p1}{sep}{p2}", p3]


def test_word_longer_than_max_chars_is_split():
    text = "x" * 250

    chunks = chunk_text(text, max_chars=100, overlap=0)

    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


# --- invalid arguments ---


@pytest.mark.parametrize(
    "max_chars, overlap",
    [
        (0, 0),  # max_chars must be positive
        (-10, 0),
        (100, -1),  # overlap can't be negative
        (100, 100),  # overlap must be smaller than max_chars
        (100, 150),
    ],
)
def test_invalid_arguments_raise(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", max_chars=max_chars, overlap=overlap)
