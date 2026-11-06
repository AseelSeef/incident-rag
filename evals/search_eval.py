"""Search eval: does the right postmortem show up in the top k chunks?

Makes no Claude calls, so it is free; run it after every change to chunking, embedding or search:

    python -m evals.search_eval          # k=5, like /ask
    python -m evals.search_eval --k 10
    python -m evals.search_eval --max-per-doc 1

Questions live in evals/questions.jsonl, one JSON object per line:
    {"question": "...", "expected_urls": ["https://..."]}
An empty expected_urls list means the postmortems can't answer the question; those are not
scored, but their best similarity score is shown (search always returns k chunks, even for
unrelated questions, so this shows how "close" an unrelated question looks).
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psycopg

from app.config import settings
from app.ingestion.db import existing_urls
from app.retrieval.search import Hit, search

QUESTIONS = Path(__file__).with_name("questions.jsonl")


@dataclass
class Question:
    question: str
    expected_urls: list[str]

    @property
    def answerable(self) -> bool:
        return bool(self.expected_urls)


@dataclass
class Result:
    question: Question
    hits: list[Hit]

    @property
    def urls(self) -> list[str]:
        """Documents found, best first, each once (several chunks can come from one document)."""
        return list(dict.fromkeys(hit.url for hit in self.hits))

    @property
    def found(self) -> list[str]:
        return [url for url in self.question.expected_urls if url in self.urls]

    @property
    def recall(self) -> float:
        """Share of the expected documents that were found (1.0 = all of them)."""
        return len(self.found) / len(self.question.expected_urls)

    @property
    def first_rank(self) -> int | None:
        """Position (1 = best) of the first expected document among the documents found."""
        for rank, url in enumerate(self.urls, start=1):
            if url in self.question.expected_urls:
                return rank
        return None

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    questions = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name} line {line_no}: {e.msg} (one object per line)") from None
        questions.append(Question(data["question"], data["expected_urls"]))
    return questions


def unknown_urls(questions: list[Question], known: set[str]) -> list[str]:
    """Expected URLs that are not in the database (usually a typo; they could never be found)."""
    return [url for q in questions for url in q.expected_urls if url not in known]


def summarize(results: list[Result]) -> dict[str, float]:
    scored = [r for r in results if r.question.answerable]
    if not scored:
        return {"hit_rate": 0.0, "recall": 0.0, "mrr": 0.0}
    n = len(scored)
    return {
        # at least one expected document was found
        "hit_rate": sum(r.first_rank is not None for r in scored) / n,
        "recall": sum(r.recall for r in scored) / n,
        # mean reciprocal rank: 1.0 if the right document is always first, 0.5 if always second
        "mrr": sum(1 / r.first_rank for r in scored if r.first_rank) / n,
    }


def run(
    conn: psycopg.Connection,
    questions: list[Question],
    k: int = 5,
    max_per_doc: int = 2,
    search_fn: Callable[..., list[Hit]] = search,
) -> list[Result]:
    return [
        Result(q, search_fn(conn, q.question, k=k, max_per_doc=max_per_doc)) for q in questions
    ]


def print_report(results: list[Result], k: int, max_per_doc: int) -> None:
    titles = {hit.url: f"{hit.company}: {hit.title}" for r in results for hit in r.hits}

    print(f"Answerable questions (k={k} chunks, at most {max_per_doc} per document)\n")
    for r in results:
        if not r.question.answerable:
            continue
        status = "OK  " if r.recall == 1 else ("PART" if r.found else "MISS")
        rank = f"rank {r.first_rank}" if r.first_rank else "  -   "
        found = f"{len(r.found)}/{len(r.question.expected_urls)}"
        print(f"{status} {found}  {rank}  {r.top_score:.3f}  {r.question.question}")
        if r.recall < 1:
            for url in r.urls:
                print(f"            got:  {titles[url]}")

    print("\nUnanswerable questions (best score; lower is better)\n")
    for r in results:
        if not r.question.answerable:
            print(f"     {r.top_score:.3f}  {r.question.question}")

    s = summarize(results)
    print(f"\nhit rate {s['hit_rate']:.0%}   recall {s['recall']:.0%}   MRR {s['mrr']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score search against evals/questions.jsonl")
    parser.add_argument("--k", type=int, default=5, help="chunks to retrieve (default 5)")
    parser.add_argument(
        "--max-per-doc", type=int, default=2, help="chunks per document (default 2)"
    )
    args = parser.parse_args()

    questions = load_questions()
    with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
        for url in unknown_urls(questions, existing_urls(conn)):
            print(f"WARNING: expected URL is not in the database: {url}")
        results = run(conn, questions, k=args.k, max_per_doc=args.max_per_doc)
    print_report(results, args.k, args.max_per_doc)


if __name__ == "__main__":
    main()
