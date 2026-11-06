import pytest

from app.retrieval.search import Hit
from evals.search_eval import (
    QUESTIONS,
    Question,
    Result,
    load_questions,
    run,
    summarize,
    unknown_urls,
)


def hit(url: str, score: float = 0.8) -> Hit:
    return Hit(content="...", title="t", company="c", url=url, chunk_index=0, score=score)


def result(expected: list[str], found_urls: list[str]) -> Result:
    return Result(Question("q?", expected), [hit(url) for url in found_urls])


# --- one result ---


def test_documents_are_counted_once_in_order():
    r = result(["a"], ["b", "a", "b", "c", "a"])

    assert r.urls == ["b", "a", "c"]


def test_rank_is_by_document_not_by_chunk():
    # Two chunks of "b" come first, but "a" is still the second document.
    r = result(["a"], ["b", "b", "a"])

    assert r.first_rank == 2


def test_missing_document_has_no_rank_and_zero_recall():
    r = result(["a"], ["b", "c"])

    assert r.first_rank is None
    assert r.recall == 0
    assert r.found == []


def test_recall_is_the_share_of_expected_documents_found():
    r = result(["a", "b", "c"], ["a", "x", "c"])

    assert r.recall == pytest.approx(2 / 3)
    assert r.found == ["a", "c"]


def test_top_score_is_the_best_hit():
    r = Result(Question("q?", []), [hit("a", 0.71), hit("b", 0.65)])

    assert r.top_score == 0.71
    assert Result(Question("q?", []), []).top_score == 0.0


# --- summary ---


def test_summary_ignores_unanswerable_questions():
    results = [
        result(["a"], ["a"]),  # rank 1
        result(["a"], ["b", "a"]),  # rank 2
        result(["a", "b"], ["a", "x"]),  # rank 1, half found
        result(["a"], ["x"]),  # missed
        result([], ["x"]),  # unanswerable: not scored
    ]

    s = summarize(results)

    assert s["hit_rate"] == 3 / 4
    assert s["recall"] == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 1 + 0) / 4)


def test_summary_of_no_answerable_questions_is_zero():
    assert summarize([result([], ["x"])]) == {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0}


# --- loading ---


def test_load_questions(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"question": "Why?", "expected_urls": ["u1"]}\n'
        "\n"  # blank lines are skipped
        '{"question": "Unrelated?", "expected_urls": []}\n',
        encoding="utf-8",
    )

    questions = load_questions(path)

    assert questions == [Question("Why?", ["u1"]), Question("Unrelated?", [])]
    assert [q.answerable for q in questions] == [True, False]


def test_object_split_over_lines_names_the_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "Why?", "expected_urls":\n  ["u1"]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_questions(path)


def test_the_real_question_file_loads():
    questions = load_questions(QUESTIONS)

    assert len(questions) >= 10
    assert any(q.answerable for q in questions)
    assert any(not q.answerable for q in questions)


def test_unknown_urls():
    questions = [Question("a?", ["known", "typo"]), Question("b?", [])]

    assert unknown_urls(questions, {"known"}) == ["typo"]


# --- running ---


def test_run_searches_each_question_with_k():
    calls = []

    def fake_search(conn, question, k):
        calls.append((question, k))
        return [hit("a")]

    results = run(None, [Question("one?", ["a"]), Question("two?", [])], k=7, search_fn=fake_search)

    assert calls == [("one?", 7), ("two?", 7)]
    assert [r.urls for r in results] == [["a"], ["a"]]
