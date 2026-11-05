from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.answer import answer_question, cited_documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=5,
        open=False,
        kwargs={"connect_timeout": 5},
    )
    pool.open()
    app.state.pool = pool
    yield
    pool.close()


app = FastAPI(title="Incident RAG", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)  # how many chunks to search; capped, since more = pricier


class SourceOut(BaseModel):
    numbers: list[int]  # the [N] that refer to this document
    title: str
    company: str | None
    url: str


class AskResponse(BaseModel):
    answer: str
    generated: bool
    sources: list[SourceOut]


def get_conn(request: Request) -> Iterator[psycopg.Connection]:
    with request.app.state.pool.connection() as conn:
        yield conn


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskRequest, conn: Annotated[psycopg.Connection, Depends(get_conn)]) -> AskResponse:
    try:
        answer = answer_question(conn, body.question, k=body.k)
    except ValueError:
        raise HTTPException(status_code=422, detail="empty question") from None
    if not answer.generated:
        # Fallback (model failed or nothing found): show every source, not just cited ones.
        answer = replace(answer, cited=list(range(1, len(answer.sources) + 1)))

    sources = [
        SourceOut(numbers=numbers, title=hit.title, company=hit.company, url=hit.url)
        for numbers, hit in cited_documents(answer)
    ]
    return AskResponse(answer=answer.text, generated=answer.generated, sources=sources)


