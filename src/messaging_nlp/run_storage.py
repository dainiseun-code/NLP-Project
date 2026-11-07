"""Small fsynced execution evidence primitives; no inference or repair."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from messaging_nlp.io import canonical_json, sha256
from messaging_nlp.preparation import artifact_path
from messaging_nlp.run_contracts import ArtifactRef


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    write_new(temporary, canonical_json(value))
    os.replace(temporary, path)
    _sync_directory(path.parent)


def append_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def ref(root: Path, path: Path) -> ArtifactRef:
    path = Path(path)
    relative = path.relative_to(root).as_posix()
    return ArtifactRef(path=relative, sha256=sha256(path), bytes=path.stat().st_size)


def read_ref(root: Path, artifact: ArtifactRef) -> bytes:
    path = artifact_path(root, artifact.path)
    data = path.read_bytes()
    import hashlib

    if (
        len(data) != artifact.bytes
        or hashlib.sha256(data).hexdigest() != artifact.sha256
    ):
        raise ValueError(f"frozen artifact identity mismatch: {artifact.path}")
    return data
