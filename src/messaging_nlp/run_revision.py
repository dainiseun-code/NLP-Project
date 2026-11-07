"""One explicitly authorized experiment, not a general retry/reset mechanism.

Standard-library only: usable by the budget ledger without any model dependency.
The primary records a new authority and code pin for each consequential decision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

AUTHORITY_PATH = "docs/stage6-revision-authorization-20261028.json"
AUTHORITY_SHA256 = "ea18e0799a660713c2a338ad25fecabd552934d0030f8f12a7d03058570a5963"
REVISION_RESET_ID = "stage6-full-rerun-reset-20261028"
EXPERIMENT_ID = "standalone-development-v2-20261028"
EPOCH_REVISION = "authorized_rerun_reset"


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def attempt_key(request, condition):
    return hashlib.sha256(
        canonical({"request": request, "condition": condition})
    ).hexdigest()


def read_bound(root, reference):
    name = reference["path"]
    parts = PurePosixPath(name)
    if (
        not name
        or parts.is_absolute()
        or ".." in parts.parts
        or "\\" in name
        or parts.as_posix() != name
    ):
        raise ValueError("revision artifact requires a project-relative safe path")
    path = root / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("revision artifact cannot traverse symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("revision artifact leaves project root")
    raw = path.read_bytes()
    if (
        len(raw) != reference["bytes"]
        or hashlib.sha256(raw).hexdigest() != reference["sha256"]
    ):
        raise ValueError("revision predecessor artifact identity changed")
    return raw


def load_authority(path, expected_sha256):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("revision authorization cannot traverse symlinks")
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    if (
        expected_sha256 != AUTHORITY_SHA256
        or hashlib.sha256(raw).hexdigest() != AUTHORITY_SHA256
    ):
        raise ValueError("untrusted revision authorization")
    body = json.loads(raw)
    if (
        body["schema_version"] != "nlp3-authorized-rerun-v1"
        or body["reset_id"] != REVISION_RESET_ID
        or body["experiment_id"] != EXPERIMENT_ID
        or body["run_id"] != EXPERIMENT_ID
        or body["scope"]["reset_storage_usage"] is not False
        or body["scope"]["reset_version_registrations"] is not False
        or body["scope"]["authorize_inference_now"] is not False
    ):
        raise ValueError("unsupported revision scope")
    root = (path.parent / body["project_root"]).resolve(strict=True)
    predecessor_raw = read_bound(root, body["predecessor_budget"])
    predecessor = json.loads(predecessor_raw)
    plan = json.loads(read_bound(root, body["predecessor_plan"]))
    status = json.loads(read_bound(root, body["predecessor_status"]))
    read_bound(root, body["protected_evidence"])
    if (
        predecessor["current_epoch"] != body["predecessor_epoch"]
        or any(r["status"] == "reserved" for r in predecessor["reservations"])
        or status["cleanup_verified"] is not True
        or status["pending_rows"]
        or status["persistence_errors"]
        or status["rows_saved"] != len(plan["rows"])
        or len(plan["rows"]) != body["scope"]["planned_rows"]
        or status["plan_sha256"] != body["predecessor_plan"]["sha256"]
    ):
        raise ValueError("revision requires a fully accounted stopped predecessor")
    keys = [attempt_key(row["request"], row["condition"]) for row in plan["rows"]]
    if len(keys) != len(set(keys)):
        raise ValueError("revision predecessor has duplicate request/conditions")
    return dict(
        body=body,
        root=root,
        predecessor=predecessor,
        predecessor_raw=predecessor_raw,
        predecessor_plan=plan,
        attempt_keys=frozenset(keys),
        sha256=expected_sha256,
        path=str(path),
    )


def validate_row_scope(info, refs):
    body = info["body"]
    if (
        refs.get("run_id") != body["run_id"]
        or refs.get("experiment_id") != body["experiment_id"]
        or refs.get("revision_authority_sha256") != info["sha256"]
        or refs.get("attempt_key") not in info["attempt_keys"]
    ):
        raise ValueError("row is outside the specifically authorized experiment")
