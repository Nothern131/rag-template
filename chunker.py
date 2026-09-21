"""
语义感知切分 — 保留段落/标题边界，带重叠。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from config import Config
from loader import Doc


@dataclass
class Chunk:
    doc_path: str
    source_type: str
    index: int
    text: str


class Chunker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def split(self, doc: Doc) -> List[Chunk]:
        text = doc.text.strip()
        if not text:
            return []

        pieces = self._split_by_structure(text)
        chunks: List[Chunk] = []
        buf: List[str] = []
        buf_len = 0
        i = 0

        def flush() -> None:
            nonlocal i, buf, buf_len
            if not buf:
                return
            chunk_text = "\n".join(buf).strip()
            if chunk_text:
                chunks.append(
                    Chunk(doc.doc_path, doc.source_type, i, chunk_text)
                )
                i += 1
            buf, buf_len = [], 0

        overlap = self.cfg.chunk_overlap
        for piece in pieces:
            if buf_len + len(piece) > self.cfg.max_chunk_length and buf:
                flush()
            if len(piece) > self.cfg.max_chunk_length:
                for sub in self._hard_split(piece, self.cfg.max_chunk_length):
                    chunks.append(
                        Chunk(doc.doc_path, doc.source_type, i, sub)
                    )
                    i += 1
                continue
            buf.append(piece)
            buf_len += len(piece) + 1
        flush()

        if overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks, overlap)
            for idx, c in enumerate(chunks):
                c.index = idx
        return chunks

    def _split_by_structure(self, text: str) -> List[str]:
        parts = re.split(r"(\n#{1,6}\s|[。！？]\s|\n\s*\n)", text)
        out: List[str] = []
        buf = ""
        for p in parts:
            if p in ("\n", ""):
                continue
            if re.match(r"^\n?#{1,6}\s", p):
                if buf.strip():
                    out.append(buf.strip())
                buf = p
            else:
                buf += p
        if buf.strip():
            out.append(buf.strip())
        return out

    def _hard_split(self, text: str, max_len: int) -> List[str]:
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]

    def _add_overlap(
        self, chunks: List[Chunk], overlap: int
    ) -> List[Chunk]:
        if overlap <= 0 or len(chunks) < 2:
            return chunks
        result: List[Chunk] = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            prev_tail = prev.text[-overlap:]
            merged = cur.text
            if cur.index > 0:
                merged = prev_tail + "\n" + cur.text
            result.append(
                Chunk(cur.doc_path, cur.source_type, cur.index, merged)
            )
        return result
