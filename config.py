"""
通用 RAG 模板 — 配置加载

所有可调参数集中在此文件，禁止散落在各模块里。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).parent


@dataclass
class Config:
    # 数据源
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma"

    # 切分
    chunk_size: int = 400
    chunk_overlap: int = 80
    max_chunk_length: int = 800

    # 嵌入
    embedding_backend: Literal["local", "openai"] = "local"
    local_embed_model: str = "all-MiniLM-L6-v2"
    openai_embed_model: str = "text-embedding-3-small"
    openai_api_key: str = ""

    # LLM
    llm_backend: Literal["openai", "anthropic", "local", "mock"] = "mock"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-latest"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    # 检索
    top_k: int = 5
    min_score: float = 0.25

    # 服务
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_seconds: int = 90

    def validate(self) -> None:
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.chroma_dir.exists():
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
        if self.llm_backend == "openai" and not self.openai_api_key:
            raise ValueError("llm_backend=openai 必须设置 openai_api_key 或 OPENAI_API_KEY")
        if self.llm_backend == "anthropic" and not self.anthropic_api_key:
            raise ValueError("llm_backend=anthropic 必须设置 anthropic_api_key 或 ANTHROPIC_API_KEY")
        if self.embedding_backend == "openai" and not self.openai_api_key:
            raise ValueError("embedding_backend=openai 必须设置 openai_api_key 或 OPENAI_API_KEY")


def load() -> Config:
    cfg = Config(
        embedding_backend=_as_literal(os.getenv("EMBEDDING_BACKEND", "local")),
        llm_backend=_as_literal(os.getenv("LLM_BACKEND", "mock")),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        top_k=int(os.getenv("TOP_K", "5")),
    )
    cfg.validate()
    return cfg


def _as_literal(v: str):
    """把字符串安全地转成对应 Literal 类型，供 Config 使用。"""
    return v
