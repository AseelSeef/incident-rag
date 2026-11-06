# Incident RAG

[![CI](https://github.com/AseelSeef/incident-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/AseelSeef/incident-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A retrieval-augmented question-answering system over public incident postmortems: it finds similar past incidents and answers with citations.

Ask *"Why did GitLab lose production data?"* or *"Which outages were caused by DNS problems?"*
and get an answer grounded in the original postmortems, with every claim cited as `[N]` and
linked to its source.

Built with Python, FastAPI, Postgres + pgvector, local embeddings (fastembed) and Claude.

## How it works

```mermaid
flowchart LR
    CSV[data/sources.csv] --> Ingest
    Ingest -->|documents| DB[(Postgres + pgvector)]
    DB --> Index
    Index -->|chunks + embeddings| DB
    Q[Question] --> Search
    DB --> Search
    Search -->|top-k chunks| Prompt
    Prompt --> Claude
    Claude -->|answer with N citations| API[POST /ask]
```

1. **Ingest** (`app/ingestion`): downloads each postmortem listed in `data/sources.csv`,
   extracts the main text with trafilatura, and stores it in `documents`. Already stored URLs
   are skipped; a document's `updated_at` changes only when its text changes.
2. **Index** (`app/indexing`): splits each new or changed document into overlapping chunks
   (by paragraph, up to 1000 characters) and embeds them locally with
   `BAAI/bge-small-en-v1.5` (384 dimensions, via fastembed). Each chunk is embedded together
   with its document's title; the stored text is the plain chunk. Stored in `chunks` with an
   HNSW cosine index.
3. **Search** (`app/retrieval`): embeds the question (with the bge query instruction) and
   returns the k most similar chunks, at most 2 from any one document.
4. **Answer** (`app/rag`): puts the chunks into a prompt as numbered sources and asks Claude to
   answer only from them, citing each claim as `[N]`. Source text is treated as data: tags in it
   are escaped so a page can't fake or close a source. If Claude is unavailable, declines or is
   cut off, the sources found are returned with a short note instead.
5. **API** (`app/main.py`): `POST /ask` returns the answer and the cited documents, with the
   chunks of one document grouped together.

## Quick start

Requires Python 3.11+ and Docker. Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env      # then put your Anthropic API key in LLM_API_KEY
docker compose up -d        # Postgres with pgvector; creates the tables on first start
```

On macOS / Linux, use `source .venv/bin/activate` and `cp .env.example .env` instead.

Then build the index and start the API:

```powershell
python -m app.ingestion.run          # download and store the postmortems
python -m app.indexing.run           # chunk and embed them
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and try `POST /ask`.

The first indexing run downloads the embedding model (about 130 MB) into `.cache/fastembed`.
Search, indexing and the tests work without an API key; only answering calls Claude.

## Asking questions

From the API:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" `
  -Body '{"question": "Why did GitLab lose data?"}' | ConvertTo-Json -Depth 5
```

```bash
curl -s http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Why did GitLab lose data?"}'
```

```json
{
  "answer": "An engineer accidentally removed the database directory on the primary ... [1][2]",
  "generated": true,
  "sources": [
    {"numbers": [1, 2], "title": "GitLab database outage of January 31, 2017", "company": "Gitlab", "url": "https://about.gitlab.com/..."}
  ]
}
```

- `question`: 1 to 1000 characters. `k` (optional, 1 to 20, default 5): how many chunks to search.
- `generated` is `false` when the answer is one of our own messages (nothing found, or the model
  failed); `sources` then lists everything that was found.
- If Postgres is unreachable, `/ask` returns `503 {"detail": "database unavailable"}` within
  5 seconds; the API recovers on its own once the database is back.

From the command line, for one question:

```powershell
python -m app.rag.answer "Which outages were caused by DNS?"   # calls Claude
python -m app.retrieval.search                                  # search only, no Claude
```

## Evaluation

`evals/questions.jsonl` holds 21 questions, each with the postmortem(s) that should be found:
single-incident questions, cross-incident ones ("which outages were caused by DNS?"), several
worded without the company name, and 3 the collection can't answer. Every expected answer was
checked against the stored text.

`python -m evals.search_eval` runs them through search and scores the **documents** found
(several chunks of one document count once). It makes no Claude calls, so it is free and runs
in seconds.

- **hit rate**: at least one expected document was found
- **recall**: share of the expected documents found
- **MRR**: how high the first expected document ranks (1.0 = always first)

How each change moved the numbers (k=5):

| Change | Hit rate | Recall | MRR |
|---|---|---|---|
| Plain vector search | 94% | 87% | 0.94 |
| + at most 2 chunks per document | 94% | 90% | 0.94 |
| + document title embedded with each chunk | 94% | **93%** | 0.94 |

The one remaining miss is a paraphrase: *"a command meant to remove a few servers removed far
more than intended"* never reaches the AWS S3 postmortem's "one of the inputs to the command was
entered incorrectly". With 21 questions, one question moves recall by several points, so the
numbers show direction more than precision.

### Answer eval

`python -m evals.answer_eval` runs 27 cases through the real `answer_question` (Haiku 5.5) and
grades each answer. This calls Claude: a full run costs about $0.21, mostly for the judge.
Cases are the 21 above plus `evals/answer_cases.jsonl`: incidents that aren't in the
collection (CrowdStrike, Akamai, Netflix), two prompt-injection cases where a planted source
is added to the search results, and a question in Arabic.

| Check | How |
|---|---|
| valid_citations | every `[N]` refers to a source that was given |
| cites_expected | an expected postmortem is cited, when the sources contain the answer |
| grounded | judge (Sonnet 5.5): every claim is supported by the sources |
| abstains_correctly | judge: says "not in the sources" exactly when the sources lack the answer |
| injection_resisted | judge: doesn't obey or repeat the planted source (rejecting it is fine) |
| language_ok | the answer is in the question's language |
| context_sufficient | judge: the sources contain the answer (measures search; not part of pass/fail) |

Baseline: **27/27 cases pass**, and `context_sufficient` is **16/21**. Given its sources, the
model didn't invent facts, didn't follow either injection, and said so whenever the sources
lacked the answer. The gaps are in search: in 3 of the 5 short cases, the per-document cap
left only 2 chunks of one long postmortem, none with the cause.

Building the grader took two corrections found by reading pilot answers: a word check failed
an answer that quoted the injected "BANANA" only to reject it, and "the right document was
retrieved" turned out not to mean "the answer was in the sources".

## Design decisions

- **Evaluate before tuning.** Each retrieval change above was kept or dropped based on the
  eval, not on a few hand-picked questions. Some trade-offs only show up that way: the title
  prefix helped at k=5 but made the S3 question worse at k=8, because another document's
  title shares a word with the question.
- **Per-document cap.** Long postmortems have many similar chunks and could fill every slot
  (one took 4 of 5), crowding out other incidents in cross-incident questions. Search fetches
  `k * 4` candidates from the HNSW index and keeps at most 2 per document. pgvector's HNSW scan
  returns at most `hnsw.ef_search` rows (default 40), so search raises it for the transaction
  when `k * 4` is larger.
- **Title prefix for embeddings only.** A chunk from the middle of a postmortem often doesn't
  name the company or system. The title is added to the text that is embedded, but not to the
  stored chunk, since the prompt already shows each source's title.
- **Local embeddings.** bge-small runs on the CPU via fastembed (ONNX, no PyTorch): no
  embedding costs, no second API key, and the tests can use the real model.
- **Citations are checked, not trusted.** Citation numbers are parsed from the answer
  (`[1]`, `[2][3]`, `[1, 2]`), and numbers that don't match a source are dropped.
- **Sources are data, not instructions.** Postmortems are third-party web pages. Sources go in
  `<source id="N">` tags and any of our tag names inside them are escaped, so a page can't close
  its source early and inject text that looks like instructions or another source.
- **Fail soft, but not silently.** Timeouts, rate limits and 5xx errors from the API become a
  fallback answer that still lists the sources found; a wrong key, an unknown model or a bad
  request is raised, because those are bugs to fix, not outages. Token usage is logged for
  every call, including refused and cut-off ones, since those are billed too.
- **Re-index only what changed.** `indexed_at` stores the `updated_at` of the version that was
  embedded, and re-ingesting identical text doesn't touch `updated_at`, so running the pipeline
  again only embeds new or changed documents.

## Testing

```powershell
pytest                       # 277 tests, about 20 s
ruff check app tests evals
```

- The tests never call Claude: the client is replaced with fakes that mirror the SDK's
  response and error types (timeouts, 429, 5xx, refusals, cut-off answers).
- Database tests run against the real Postgres + pgvector, each inside a transaction that is
  rolled back, so your data is never changed. Locally they are skipped if Postgres isn't
  running.
- CI (GitHub Actions) runs ruff and the tests on Python 3.11 and 3.14 against a pgvector
  service container; there, an unreachable database fails the run instead of skipping tests.

## Commands

| Command | What it does |
|---|---|
| `python -m app.ingestion.run` | Fetch and store sources. `--limit N`, `--force` (re-fetch stored URLs), `--delay` |
| `python -m app.indexing.run` | Chunk and embed new or changed documents. `--force` re-indexes all |
| `uvicorn app.main:app --reload` | Start the API on port 8000 |
| `python -m evals.search_eval` | Score search against `evals/questions.jsonl` (free). `--k N`, `--max-per-doc N` |
| `python -m evals.answer_eval` | Grade answers (calls Claude, about $0.21). `--name`, `--only ID...`, `--limit N`, `--fresh` |
| `pytest` | Run the tests (database tests are skipped if Postgres isn't running) |
| `ruff check app tests evals` | Lint |

## Configuration

Settings are read from `.env` (see `app/config.py`). Run commands from the project folder so
`.env` is found.

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/incident_rag` | Postgres connection |
| `LLM_API_KEY` | (empty) | Anthropic API key; if empty, the SDK falls back to `ANTHROPIC_API_KEY` |
| `LLM_MODEL` | `claude-haiku-5-5` | Claude model used for answers |

## Costs

Embeddings run locally and cost nothing. Each answered question makes one Claude call; with
`claude-haiku-5-5` and `k=5` that is roughly 2,000 input and a few hundred output tokens, about
$0.001 or less. The API logs each call's token usage:

```
INFO:     app.rag.llm - claude usage: model=claude-haiku-5-5 input_tokens=1834 output_tokens=412 stop_reason=end_turn
```

Totals per day and per API key are on the Usage and Cost pages of the Claude Console.

## Project layout

```
app/
  ingestion/   sources.py, fetch.py, extract.py, chunk.py, db.py, run.py   download, store, chunk
  indexing/    run.py                                           embed new or changed documents
  embeddings.py                                                 local embedding model
  retrieval/   search.py                                        vector search with per-document cap
  rag/         answer.py (prompt, citations, fallback), llm.py (Claude client)
  main.py                                                       FastAPI app
evals/         questions.jsonl, search_eval.py                  retrieval evaluation
db/init.sql                                                     schema (runs on first DB start)
data/sources.csv                                                list of postmortem URLs
tests/
.github/workflows/ci.yml                                        lint + tests on every push
```

## Database schema changes

`db/init.sql` only runs when the database volume is first created. After changing it, apply the
change to an existing database by hand (for example with `ALTER TABLE` in
`docker compose exec db psql -U postgres -d incident_rag`), or recreate the volume with
`docker compose down -v`, which **deletes all stored data**.

## Limitations

- **A small collection**: 45 postmortems. Questions about incidents outside it get "not found"
  or loosely related sources.
- **A small eval set**: 21 questions are enough to catch regressions and compare changes, not
  to measure quality precisely.
- **Unrelated questions still retrieve something.** Search always returns k chunks; the best
  similarity for unrelated questions (0.52 to 0.66) is close to that of real matches (from
  0.73), so a score threshold can't tell them apart. The prompt tells the model to say when the
  sources don't answer the question instead.
- **The answer eval checks correctness, not completeness.** The judge checks that every claim
  is supported, not that the answer covers everything the sources say. With 27/27 passing, it
  catches regressions but can no longer show improvements.

## Roadmap

- [x] Ingestion, local embeddings, vector search, answers with citations over an API
- [x] Retrieval evaluation; per-document cap and title-aware embeddings measured against it
- [x] CI: lint and tests against Postgres + pgvector
- [x] Answer evaluation: citation validity, "not found" behaviour, prompt-injection cases
- [ ] Backfill search slots when the per-document cap leaves fewer than k chunks
- [ ] Hybrid search (vector + Postgres full-text) for exact names and error codes
- [ ] Agent with tools (find similar incidents, compare incidents), structured extraction
      (root cause, duration, impact)
- [ ] More sources

## Data

Postmortem texts are not stored in this repository. The ingestion script downloads them from
their original public sources; they belong to their authors.

## License

The code is released under the [MIT License](LICENSE). This does not cover the postmortems
themselves.
