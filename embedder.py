"""
嵌入后端 — 本地 / OpenAI 双路，失败自动降级到本地。
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
from config import Config

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._local_model = None
        self._local_ok = False
        if cfg.embedding_backend == "local":
            self._init_local()
        elif cfg.embedding_backend == "openai":
            if not cfg.openai_api_key:
                raise ValueError("openai embed 必须配置 openai_api_key")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        try:
            if self.cfg.embedding_backend == "local":
                self._init_local()
                vecs = self._local_model.encode(
                    list(texts), batch_size=32, show_progress_bar=False
                )
                return np.asarray(vecs, dtype=np.float32)
            return self._embed_openai(list(texts))
        except Exception as e:
            logger.warning("embed_failed backend=%s error=%s, fallback to local",
                           self.cfg.embedding_backend, e)
            self._init_local()
            vecs = self._local_model.encode(
                list(texts), batch_size=32, show_progress_bar=False
            )
            return np.asarray(vecs, dtype=np.float32)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        from openai import OpenAI  # 延迟导入，避免无 key 时启动崩溃

        client = OpenAI(
            api_key=self.cfg.openai_api_key,
            base_url=self.cfg.openai_base_url or None,
        )
        resp = client.embeddings.create(
            model=self.cfg.openai_embed_model,
            input=texts,
            timeout=self.cfg.llm_timeout_seconds,
        )
        return np.asarray(
            [d.embedding for d in resp.data], dtype=np.float32
        )

    def _init_local(self) -> None:
        if self._local_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self.cfg.local_embed_model)
            self._local_ok = True
        except Exception as e:
            raise RuntimeError(
                f"本地嵌入模型加载失败: {e}. "
                f"请 pip install sentence-transformers 或设置 EMBEDDING_BACKEND=openai"
            ) from e

    @property
    def vector_size(self) -> int:
        if self.cfg.embedding_backend == "local":
            self._init_local()
            return int(self._local_model.get_sentence_embedding_dimension())
        return 1536  # openai text-embedding-3-small
