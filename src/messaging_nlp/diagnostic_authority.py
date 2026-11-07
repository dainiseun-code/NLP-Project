"""One user-authorized, twenty-call diagnostic; never a generic retry switch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

AUTHORITY_NAME = "stage6-diagnostic-authorization-20261028.json"
AUTHORITY_SHA256 = "ad3e313b5874437c9e9659c5b2c116eca8644f7bc389e7fc50b93948a70a364e"
DIAGNOSTIC_ID = "contract-diagnostic-20261028"
DESTINATION = "runs/stage6-contract-diagnostic-20261028"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def load_authority(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("diagnostic authority cannot be a symlink")
    raw = path.read_bytes()
    if digest(raw) != AUTHORITY_SHA256:
        raise ValueError("diagnostic authority differs from its code-pinned approval")
    body = json.loads(raw)
    if "manifest" in body:
        root = (path.parent / body["project_root"]).resolve(strict=True)
        reference = body["manifest"]
        manifest_path = root / reference["path"]
        if manifest_path.is_symlink() or not manifest_path.resolve().is_relative_to(
            root
        ):
            raise ValueError("diagnostic manifest escapes its explicit root")
        manifest_raw = manifest_path.read_bytes()
        if (
            len(manifest_raw) != reference["bytes"]
            or digest(manifest_raw) != reference["sha256"]
        ):
            raise ValueError("diagnostic manifest differs from its approval")
        manifest = json.loads(manifest_raw)
        if set(manifest) & set(body):
            raise ValueError("diagnostic manifest cannot override authorization fields")
        body.update(manifest)
    if (
        body["diagnostic_id"] != DIAGNOSTIC_ID
        or body["destination"] != DESTINATION
        or body["max_calls"] != 20
        or body["restart_waived"] is not True
        or body["assistant_version_cap"] != {"previous": 4, "approved": 5}
        or len(body["slots"]) != 20
        or [s["slot"] for s in body["slots"]] != [f"{i:02d}" for i in range(20)]
    ):
        raise ValueError("diagnostic authority scope changed")
    return body


def check_version(body: dict, entries: list, version_id: str, refs: dict) -> None:
    if digest(canonical(entries[:4])) != body["prior_assistant_versions_sha256"]:
        raise ValueError("diagnostic must preserve all four earlier prompt versions")
    if {"version_id": version_id, "refs": refs} != body["fifth_version"]:
        raise ValueError(
            "only the explicitly authorized fifth prompt version is allowed"
        )


def check_slot(body: dict, refs: dict, seconds: int, generations: int) -> None:
    slot = next(
        (s for s in body["slots"] if s["slot"] == refs.get("diagnostic_slot")), None
    )
    if (
        refs.get("diagnostic_id") != DIAGNOSTIC_ID
        or refs.get("run_id") != DIAGNOSTIC_ID
        or slot is None
        or refs.get("messages_sha256") != slot["messages"]["sha256"]
        or not 0 < seconds <= 120
        or generations != 1
    ):
        raise ValueError("generation is outside the twenty frozen diagnostic slots")
