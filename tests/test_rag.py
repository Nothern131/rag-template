"""
核心测试 — 不依赖网络 / LLM API，用 Mock 后端验证全流程。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunker import Chunker
from config import Config
from loader import load_directory
from rag import RAGService
from store import ChromaStore
from embedder import Embedder


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    cfg = Config(
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "chroma",
        llm_backend="mock",
        embedding_backend="local",
        chunk_size=100,
        chunk_overlap=20,
        max_chunk_length=200,
        top_k=3,
        min_score=0.0,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "faq.txt").write_text(
        "问题1：如何退款？\n回答：7 天内无理由退款。\n\n"
        "问题2：账号被封怎么办？\n回答：联系客服申诉。\n\n"
        "问题3：如何兑换 CDK？\n回答：在商城页面输入 CDK。",
        encoding="utf-8",
    )
    return cfg


def test_loader(tmp_config: Config) -> None:
    docs = load_directory(tmp_config.data_dir)
    assert len(docs) == 1
    assert "退款" in docs[0].text


def test_chunker(tmp_config: Config) -> None:
    docs = load_directory(tmp_config.data_dir)
    chunker = Chunker(tmp_config)
    chunks = chunker.split(docs[0])
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= tmp_config.max_chunk_length + 50  # 允许少量重叠溢出


def test_end_to_end(tmp_config: Config) -> None:
    svc = RAGService(tmp_config)
    ing = svc.ingest_directory()
    assert ing["docs"] == 1
    assert ing["chunks"] >= 1

    ans = svc.answer("如何退款")
    assert ans.error == ""
    assert ans.citations, "应返回至少 1 条引用"
    assert "退款" in ans.citations[0].text or "7 天" in ans.answer

    hc = svc.health_check()
    assert hc["status"] == "ok"
    assert hc["llm_backend"] == "mock"


def test_empty_question(tmp_config: Config) -> None:
    svc = RAGService(tmp_config)
    svc.ingest_directory()
    ans = svc.answer("")
    assert ans.error == "empty_question"


def test_no_match(tmp_config: Config) -> None:
    svc = RAGService(tmp_config)
    svc.ingest_directory()
    ans = svc.answer("火星殖民计划", top_k=3)
    assert ans.error == ""
    # Mock 后端会返回"无法回答"
    assert "无法回答" in ans.answer or ans.citations


def test_chroma_reset(tmp_config: Config) -> None:
    svc = RAGService(tmp_config)
    svc.ingest_directory()
    assert svc._store.count() > 0
    svc.reset()
    assert svc._store.count() == 0
