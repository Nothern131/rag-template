"""
RAG 核心 — 检索 + 生成 + 溯源 + 置信度。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from chunker import Chunker
from config import Config
from embedder import Embedder
from llm import LLMResult, build_backend
from loader import Doc, load_directory
from store import ChromaStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    question: str
    answer: str
    citations: List[SearchResult]
    confidence: float
    model: str
    backend: str
    latency_ms: int
    error: str = ""


class RAGService:
    def __init__(self, cfg: Config, store: Optional[ChromaStore] = None) -> None:
        self.cfg = cfg
        self._store = store or ChromaStore(cfg, Embedder(cfg))
        self._chunker = Chunker(cfg)
        self._backend = build_backend(cfg)
        self._index_start: Optional[float] = None

    def health_check(self) -> dict:
        hc = self._store.health_check()
        hc["llm_backend"] = self.cfg.llm_backend
        hc["embedding_backend"] = self.cfg.embedding_backend
        hc["top_k"] = self.cfg.top_k
        return hc

    def ingest_directory(self, dir_path=None) -> dict:
        self._index_start = time.time()
        data_dir = dir_path or self.cfg.data_dir
        docs: List[Doc] = load_directory(data_dir)
        total_chunks = 0
        for doc in docs:
            chunks = self._chunker.split(doc)
            total_chunks += self._store.upsert(chunks)
        elapsed = int((time.time() - self._index_start) * 1000) if self._index_start else 0
        logger.info("ingest done: docs=%d chunks=%d elapsed_ms=%d",
                    len(docs), total_chunks, elapsed)
        return {
            "docs": len(docs),
            "chunks": total_chunks,
            "elapsed_ms": elapsed,
        }

    def ingest_files(self, paths: List[str]) -> dict:
        from loader import load_file
        total = 0
        for p in paths:
            doc = load_file(__import__("pathlib").Path(p))
            total += self._store.upsert(self._chunker.split(doc))
        return {"chunks": total}

    def answer(self, question: str, top_k: Optional[int] = None) -> Answer:
        start = time.time()
        question = (question or "").strip()
        if not question:
            return Answer(question="", answer="", citations=[],
                          confidence=0.0, model="", backend="",
                          latency_ms=0, error="empty_question")

        k = top_k or self.cfg.top_k
        hits = self._store.query(question, top_k=k)
        if not hits:
            return Answer(
                question=question,
                answer="根据现有资料无法回答此问题",
                citations=[],
                confidence=0.0,
                model=self.cfg.openai_model if self.cfg.llm_backend == "openai" else "mock",
                backend=self.cfg.llm_backend,
                latency_ms=int((time.time() - start) * 1000),
            )

        ctx_blocks = [h.text for h in hits]
        res: LLMResult = self._backend.chat(question, ctx_blocks)

        if res.error:
            return Answer(
                question=question,
                answer="",
                citations=hits,
                confidence=0.0,
                model=res.model,
                backend=res.backend,
                latency_ms=int((time.time() - start) * 1000),
                error=res.error,
            )

        confidence = min(h.score for h in hits)
        return Answer(
            question=question,
            answer=res.text,
            citations=hits,
            confidence=round(confidence, 4),
            model=res.model,
            backend=res.backend,
            latency_ms=int((time.time() - start) * 1000),
        )

    def stream_answer(
        self, question: str, top_k: Optional[int] = None
    ) -> Iterator[str]:
        hits = self._store.query(question, top_k=top_k or self.cfg.top_k)
        if not hits:
            yield "根据现有资料无法回答此问题"
            return
        ctx_blocks = [h.text for h in hits]
        yield from self._backend.stream(question, ctx_blocks)

    def reset(self) -> None:
        self._store.reset()
