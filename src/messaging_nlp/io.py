"""Strict JSON readers, canonical serialization and file hashing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

_HASH_CHUNK_SIZE = 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number: {text}")
    return value


def _reject_constant(text: str) -> None:
    raise ValueError(f"nonfinite JSON number: {text}")


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)


def loads_json(text: str) -> Any:
    """Parse strict JSON, rejecting duplicate keys and nonfinite numbers."""
    value = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_float=_finite_float,
        parse_constant=_reject_constant,
    )
    _reject_nonfinite(value)
    return value


def read_json(path: Path) -> Any:
    """Read and parse a UTF-8 JSON file strictly."""
    return loads_json(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file containing only nonblank object rows."""
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}: {path}")
        row = loads_json(line)
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL row at line {line_number} is not an object: {path}"
            )
        rows.append(row)
    return rows


def canonical_json(value: Any) -> bytes:
    """Encode deterministic, compact UTF-8 JSON."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest, reading in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
