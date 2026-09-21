"""
Chroma 向量存储 — 持久化 + 健康检查。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import EmbeddingFunction
from config import Config
from embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_path: str
    source_type: str
    index: int
    text: str
    score: float


class ChromaStore:
    def __init__(self, cfg: Config, embedder: Embedder) -> None:
        self.cfg = cfg
        self._client: ClientAPI = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name="rag_template",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder

    def health_check(self) -> dict:
        count = self.count()
        return {"status": "ok", "docs": count, "chroma_dir": str(self.cfg.chroma_dir)}

    def count(self) -> int:
        return int(self._collection.count())

    def upsert(self, chunks: List) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vecs = self._embedder.embed(texts)
        ids = [
            f"{c.doc_path}:{c.index}" for c in chunks
        ]
        metas = [
            {
                "doc_path": c.doc_path,
                "source_type": c.source_type,
                "chunk_index": c.index,
            }
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=[list(v) for v in vecs],
            documents=texts,
            metadatas=metas,
        )
        logger.info("upsert chunks: %d", len(chunks))
        return len(chunks)

    def query(self, question: str, top_k: Optional[int] = None) -> List[SearchResult]:
        k = top_k or self.cfg.top_k
        qvec = self._embedder.embed([question])
        if qvec.size == 0:
            return []
        resp = self._collection.query(
            query_embeddings=[list(qvec[0])],
            n_results=k,
        )
        out: List[SearchResult] = []
        ids = resp.get("ids", [[]])[0]
        docs = resp.get("documents", [[]])[0]
        metadatas = resp.get("metadatas", [[]])[0]
        distances = resp.get("distances", [[]])[0]
        for i, (did, doc, meta, dist) in enumerate(
            zip(ids, docs, metadatas, distances)
        ):
            out.append(
                SearchResult(
                    doc_path=meta["doc_path"] if meta else "",
                    source_type=meta.get("source_type", "") if meta else "",
                    index=meta.get("chunk_index", i) if meta else i,
                    text=doc or "",
                    score=float(1.0 - dist),
                )
            )
        out = [r for r in out if r.score >= self.cfg.min_score]
        return out

    def reset(self) -> None:
        self._client.delete_collection("rag_template")
        self._collection = self._client.get_or_create_collection(
            name="rag_template",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("chroma collection reset")
