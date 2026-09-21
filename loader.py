"""
文档加载 — 支持 txt / md / csv / json / jsonl / html。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)

SUPPORTED = {".txt", ".md", ".csv", ".json", ".jsonl", ".html", ".htm"}


@dataclass
class Doc:
    path: str
    text: str
    source_type: str
    metadata: dict


def load_file(path: Path) -> Doc:
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED)}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if ext in (".txt", ".md", ".html", ".htm"):
        text = _read_text(path)
    elif ext == ".csv":
        text = _csv_to_text(path)
    elif ext == ".json":
        text = _json_to_text(path)
    elif ext == ".jsonl":
        text = _jsonl_to_text(path)
    else:
        raise ValueError(f"Unknown extension: {ext}")

    return Doc(
        path=str(path),
        text=text,
        source_type=ext.lstrip("."),
        metadata={"size_bytes": path.stat().st_size},
    )


def load_directory(dir_path: Path) -> List[Doc]:
    docs: List[Doc] = []
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    for f in sorted(dir_path.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            try:
                docs.append(load_file(f))
            except Exception as e:
                logger.warning("skip_file_error: path=%s error=%s", f, e)
    return docs


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} with any known encoding")


def _csv_to_text(path: Path) -> str:
    import csv
    lines: List[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(" | ".join(row))
    return "\n".join(lines)


def _json_to_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _json_obj_to_text(data)


def _json_obj_to_text(obj: object) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts: List[str] = []
        for k, v in obj.items():
            if isinstance(v, str):
                parts.append(f"{k}: {v}")
            else:
                parts.append(f"{k}: {_json_obj_to_text(v)}")
        return "\n".join(parts)
    if isinstance(obj, list):
        return "\n".join(_json_obj_to_text(i) for i in obj)
    return str(obj)


def _jsonl_to_text(path: Path) -> str:
    lines: List[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines.append(_json_obj_to_text(obj))
            except json.JSONDecodeError:
                lines.append(line)
    return "\n".join(lines)
