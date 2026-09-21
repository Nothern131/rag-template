"""
FastAPI 服务 — 流式问答 + /health + 指标。
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import Config, load as load_config
from rag import RAGService

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class QuestionReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class IngestReq(BaseModel):
    files: list[str] = Field(default_factory=list)
    reset_first: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.cfg = cfg
    app.state.rag = RAGService(cfg)
    app.state.metrics = {"requests_total": 0, "errors_total": 0,
                         "ingest_total": 0}
    logger.info("RAG service started: llm=%s embed=%s",
                cfg.llm_backend, cfg.embedding_backend)
    yield
    logger.info("RAG service stopped")


app = FastAPI(title="RAG Template API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return app.state.rag.health_check()


@app.post("/ingest")
def ingest(req: IngestReq) -> dict:
    if req.reset_first:
        app.state.rag.reset()
    if req.files:
        result = app.state.rag.ingest_files(req.files)
    else:
        result = app.state.rag.ingest_directory()
    app.state.metrics["ingest_total"] += 1
    logger.info("ingest: %s", result)
    return result


@app.post("/ask")
def ask(req: QuestionReq) -> dict:
    t0 = time.time()
    app.state.metrics["requests_total"] += 1
    try:
        ans = app.state.rag.answer(req.question, top_k=req.top_k)
        if ans.error:
            app.state.metrics["errors_total"] += 1
        logger.info(
            "ask ok: q=%r confidence=%.3f latency_ms=%d",
            req.question[:50], ans.confidence, ans.latency_ms,
        )
        return asdict(ans)
    except Exception as e:
        app.state.metrics["errors_total"] += 1
        logger.exception("ask failed")
        raise HTTPException(500, str(e)) from e


@app.post("/ask/stream")
async def ask_stream(req: QuestionReq) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        try:
            for token in app.state.rag.stream_answer(req.question, top_k=req.top_k):
                yield token
            yield "\n"
        except Exception as e:
            logger.exception("stream failed")
            yield f"\n[stream_error] {e}"

    return StreamingResponse(gen(), media_type="text/plain")


@app.get("/metrics")
def metrics() -> dict:
    return app.state.metrics


def main() -> None:
    cfg = load_config()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
