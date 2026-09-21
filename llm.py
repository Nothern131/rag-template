"""
LLM 后端 — OpenAI / Anthropic / 本地 / Mock 四路，带重试。
"""
from __future__ import annotations

import logging
import time
from typing import Iterator, List, Optional, Protocol

from config import Config
from store import SearchResult

logger = logging.getLogger(__name__)


class LLMResult:
    __slots__ = ("text", "model", "backend", "latency_ms", "error")

    def __init__(
        self,
        text: str,
        model: str,
        backend: str,
        latency_ms: int,
        error: str = "",
    ) -> None:
        self.text = text
        self.model = model
        self.backend = backend
        self.latency_ms = latency_ms
        self.error = error


SYSTEM_PROMPT = """你是一个客服问答助手，基于下面给出的资料回答问题。
规则：
1. 只使用「参考资料」中的信息回答，不要编造。
2. 回答末尾用 [n] 标注引用了哪条资料，n 是 1-based 序号。
3. 如果资料里没有答案，直接回答"根据现有资料无法回答此问题"。
4. 回答要简洁，不超过 150 字。

参考资料：
{context}

问题：{question}

回答："""


class LLMBackend(Protocol):
    def chat(
        self, question: str, context_blocks: List[str]
    ) -> LLMResult:
        ...

    def stream(
        self, question: str, context_blocks: List[str]
    ) -> Iterator[str]:
        ...


class OpenAIBackend:
    def __init__(self, cfg: Config) -> None:
        from openai import OpenAI  # 延迟导入
        self._client = OpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url or None,
        )
        self._cfg = cfg
        self._model = cfg.openai_model
        self._backend = "openai"

    def _prompt(self, question: str, context_blocks: List[str]) -> str:
        ctx = "\n\n".join(
            f"[{i+1}] {b}" for i, b in enumerate(context_blocks)
        )
        return SYSTEM_PROMPT.format(context=ctx, question=question)

    def chat(self, question: str, context_blocks: List[str]) -> LLMResult:
        start = time.time()
        prompt = self._prompt(question, context_blocks)
        last_err = ""
        for attempt in range(self._cfg.llm_max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self._cfg.llm_max_tokens,
                    temperature=self._cfg.llm_temperature,
                    timeout=self._cfg.llm_timeout_seconds,
                )
                text = resp.choices[0].message.content or ""
                return LLMResult(
                    text=text,
                    model=self._model,
                    backend=self._backend,
                    latency_ms=int((time.time() - start) * 1000),
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(
                    "openai attempt=%d/%d failed: %s",
                    attempt + 1,
                    self._cfg.llm_max_retries + 1,
                    last_err,
                )
                if attempt < self._cfg.llm_max_retries:
                    time.sleep(2 ** attempt)
        return LLMResult(
            text="",
            model=self._model,
            backend=self._backend,
            latency_ms=int((time.time() - start) * 1000),
            error=last_err,
        )

    def stream(self, question: str, context_blocks: List[str]) -> Iterator[str]:
        prompt = self._prompt(question, context_blocks)
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._cfg.llm_max_tokens,
            temperature=self._cfg.llm_temperature,
            timeout=self._cfg.llm_timeout_seconds,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class AnthropicBackend:
    def __init__(self, cfg: Config) -> None:
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=cfg.anthropic_api_key,
            timeout=float(cfg.llm_timeout_seconds),
        )
        self._cfg = cfg
        self._model = cfg.anthropic_model
        self._backend = "anthropic"

    def _prompt(self, question: str, context_blocks: List[str]) -> str:
        ctx = "\n\n".join(
            f"[{i+1}] {b}" for i, b in enumerate(context_blocks)
        )
        return SYSTEM_PROMPT.format(context=ctx, question=question)

    def chat(self, question: str, context_blocks: List[str]) -> LLMResult:
        start = time.time()
        prompt = self._prompt(question, context_blocks)
        last_err = ""
        for attempt in range(self._cfg.llm_max_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._cfg.llm_max_tokens,
                    temperature=self._cfg.llm_temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text if resp.content else ""
                return LLMResult(
                    text=text,
                    model=self._model,
                    backend=self._backend,
                    latency_ms=int((time.time() - start) * 1000),
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(
                    "anthropic attempt=%d/%d failed: %s",
                    attempt + 1,
                    self._cfg.llm_max_retries + 1,
                    last_err,
                )
                if attempt < self._cfg.llm_max_retries:
                    time.sleep(2 ** attempt)
        return LLMResult(
            text="",
            model=self._model,
            backend=self._backend,
            latency_ms=int((time.time() - start) * 1000),
            error=last_err,
        )

    def stream(self, question: str, context_blocks: List[str]) -> Iterator[str]:
        prompt = self._prompt(question, context_blocks)
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._cfg.llm_max_tokens,
            temperature=self._cfg.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            for token in s.text_stream:
                yield token


class LocalBackend:
    """用本地 Ollama / vLLM 等，通过 OpenAI 兼容接口。"""

    def __init__(self, cfg: Config) -> None:
        from openai import OpenAI
        base_url = cfg.openai_base_url or "http://localhost:11434/v1"
        self._client = OpenAI(
            api_key=cfg.openai_api_key or "ollama",
            base_url=base_url,
        )
        self._model = cfg.openai_model or "llama3"
        self._backend = "local"
        self._cfg = cfg

    def chat(self, question: str, context_blocks: List[str]) -> LLMResult:
        start = time.time()
        prompt = OpenAIBackend(self._cfg)._prompt(question, context_blocks)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._cfg.llm_max_tokens,
                temperature=self._cfg.llm_temperature,
                timeout=self._cfg.llm_timeout_seconds,
            )
            text = resp.choices[0].message.content or ""
            return LLMResult(text, self._model, self._backend,
                             int((time.time() - start) * 1000))
        except Exception as e:
            return LLMResult("", self._model, self._backend,
                             int((time.time() - start) * 1000),
                             error=str(e))

    def stream(self, question: str, context_blocks: List[str]) -> Iterator[str]:
        prompt = OpenAIBackend(self._cfg)._prompt(question, context_blocks)
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._cfg.llm_max_tokens,
            temperature=self._cfg.llm_temperature,
            timeout=self._cfg.llm_timeout_seconds,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


class MockBackend:
    """离线测试用 — 直接返回资料摘要。"""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._model = "mock"
        self._backend = "mock"

    def chat(self, question: str, context_blocks: List[str]) -> LLMResult:
        start = time.time()
        if not context_blocks:
            text = "根据现有资料无法回答此问题"
        else:
            top = context_blocks[0][:200]
            text = f"摘要自资料[1]：{top} …（Mock 后端，未调用 LLM）"
        return LLMResult(text, self._model, self._backend,
                         int((time.time() - start) * 1000))

    def stream(self, question: str, context_blocks: List[str]) -> Iterator[str]:
        for tok in self.chat(question, context_blocks).text:
            yield tok


def build_backend(cfg: Config) -> LLMBackend:
    if cfg.llm_backend == "openai":
        return OpenAIBackend(cfg)
    if cfg.llm_backend == "anthropic":
        return AnthropicBackend(cfg)
    if cfg.llm_backend == "local":
        return LocalBackend(cfg)
    return MockBackend(cfg)
