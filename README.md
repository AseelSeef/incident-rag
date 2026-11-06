# Incident RAG

A retrieval-augmented question-answering system over public incident postmortems: it finds similar past incidents and answers with citations.

> Work in progress. Phase 1 (ingestion, embeddings, retrieval, cited answers over an API) is done.

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
   `BAAI/bge-small-en-v1.5` (384 dimensions, via fastembed). Stored in `chunks` with an
   HNSW cosine index.
3. **Search** (`app/retrieval`): embeds the question (with the bge query instruction) and
   returns the k most similar chunks.
4. **Answer** (`app/rag`): puts the chunks into a prompt as numbered sources and asks Claude to
   answer only from them, citing each claim as `[N]`. Source text is treated as data: tags in it
   are escaped so a page can't fake or close a source. If Claude is unavailable, declines or is
   cut off, the sources found are returned with a short note instead.
5. **API** (`app/main.py`): `POST /ask` returns the answer and the cited documents, with the
   chunks of one document grouped together.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env      # then put your Anthropic API key in LLM_API_KEY
docker compose up -d        # Postgres with pgvector; creates the tables on first start
```

Then build the index and start the API:

```powershell
python -m app.ingestion.run          # download and store the postmortems
python -m app.indexing.run           # chunk and embed them
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and try `POST /ask`.

The first indexing run downloads the embedding model (about 130 MB) into `.cache/fastembed`.

## Asking questions

From the API:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" `
  -Body '{"question": "Why did GitLab lose data?"}' | ConvertTo-Json -Depth 5
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

From the command line, for one question:

```powershell
python -m app.rag.answer "Which outages were caused by DNS?"   # calls Claude
python -m app.retrieval.search                                  # search only, no Claude
```

## Commands

| Command | What it does |
|---|---|
| `python -m app.ingestion.run` | Fetch and store sources. `--limit N`, `--force` (re-fetch stored URLs), `--delay` |
| `python -m app.indexing.run` | Chunk and embed new or changed documents. `--force` re-indexes all |
| `uvicorn app.main:app --reload` | Start the API on port 8000 |
| `pytest` | Run the tests (database tests are skipped if Postgres isn't running) |
| `ruff check app tests` | Lint |

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
The tests use fakes and never call the API.

## Project layout

```
app/
  ingestion/   sources.py, fetch.py, extract.py, db.py, run.py   download and store documents
  indexing/    run.py                                           chunk + embed documents
               (chunking is in ingestion/chunk.py)
  embeddings.py                                                 local embedding model
  retrieval/   search.py                                        vector search
  rag/         answer.py (prompt, citations, fallback), llm.py (Claude client)
  main.py                                                       FastAPI app
db/init.sql                                                     schema (runs on first DB start)
data/sources.csv                                                list of postmortem URLs
tests/
```

## Database schema changes

`db/init.sql` only runs when the database volume is first created. After changing it, apply the
change to an existing database by hand (for example with `ALTER TABLE` in
`docker compose exec db psql -U postgres -d incident_rag`), or recreate the volume with
`docker compose down -v`, which **deletes all stored data**.

## Roadmap

- [x] Phase 1: ingestion script, embeddings, retrieval, answers with citations
- [ ] Phase 2: agent with tools (similar incidents, compare incidents), structured extraction
- [ ] Phase 3: evaluation set, CI, logging, architecture write-up

## Data

Postmortem texts are not stored in this repository. The ingestion script downloads
them from their original public sources.
