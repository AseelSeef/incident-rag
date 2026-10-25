# Incident RAG

A RAG agent that searches public incident postmortems, finds similar past
incidents, and answers questions with citations.

> Work in progress. The architecture diagram and design decisions will be added as the project grows.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
docker compose up -d
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. Run the tests with `pytest`.

## Roadmap

- [ ] Phase 1: ingestion script, embeddings, retrieval, answers with citations
- [ ] Phase 2: agent with tools (similar incidents, compare incidents), structured extraction
- [ ] Phase 3: evaluation set, CI, logging, architecture write-up

## Data

Postmortem texts are not stored in this repository. The ingestion script downloads
them from their original public sources.
