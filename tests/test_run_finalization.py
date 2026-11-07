"""Independent repair requirements: disk/receipt faults must not skip cleanup.

Only injected children and fictional worlds are used. This file is lead-owned,
independent of the coordinator repair implementation. No subprocess is launched.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import test_runner as support
from test_runner import deny_processes_and_optional_models, fixture  # noqa: F401

from messaging_nlp.run_contracts import PendingRowRecord

FIRST = "independent injected original persistence failure"


def _generates(f):
    return [
        request
        for child in f.children
        for request in child.requests
        if request.kind == "generate"
    ]


def _assert_mutex(verified):
    mutex = support.Mutex.instances[-1]
    assert mutex.releases, (
        "outermost disposition must be reached despite evidence faults"
    )
    assert mutex.releases[-1] is verified
    assert mutex.acquired is not verified


def _assert_failed(result):
    assert result["status"] == "failed"
    assert result["rows_saved"] <= result["planned_rows"]
    if result.get("pending_rows"):
        assert result["durability"] != "complete"
        for record in result["pending_rows"]:
            PendingRowRecord.model_validate(record)


@pytest.mark.parametrize("verified", [True, False])
@pytest.mark.parametrize(
    "fault", ["after_run_row", "receipt_validation", "ledger_charge"]
)
def test_fault_during_row_commit_preserves_first_error_and_owned_bytes(
    fixture,  # noqa: F811
    verified,
    fault,
    monkeypatch,
):
    f = fixture
    f.behavior.cleanup = verified
    original_write = f.runner._write_json
    original_row_bytes = {}

    if fault == "after_run_row":

        def write(path, value):
            original_write(path, value)
            if Path(path).name == "run-row.json" and Path(path).parent.name == "000000":
                original_row_bytes["bytes"] = Path(path).read_bytes()
                raise OSError(FIRST)

        monkeypatch.setattr(f.runner, "_write_json", write)
    elif fault == "receipt_validation":

        class InvalidReceipt:
            def __new__(cls, *args, **kwargs):
                raise ValueError(FIRST)

            @classmethod
            def model_validate(cls, *args, **kwargs):
                raise ValueError(FIRST)

        monkeypatch.setattr(f.runner, "RowReceipt", InvalidReceipt)
    else:

        def charge(*args, **kwargs):
            raise ValueError(FIRST)

        monkeypatch.setattr(support.Ledger, "charge", charge)

    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    _assert_failed(result)
    assert FIRST in result["stop_reason"], (
        "secondary exclusive-file collisions must not mask the first error"
    )
    assert len(_generates(f)) == 1, "persistence failure must stop before the next row"
    _assert_mutex(verified)
    run_row_path = f.root / "out/rows/000000/run-row.json"
    raw = run_row_path.read_bytes()
    assert json.loads(raw)["status"] == "ok", (
        "do not rewrite the genuine returned RunRow as unexecuted"
    )
    assert json.loads(raw)["raw_output"] == f.behavior.raw
    if original_row_bytes:
        assert raw == original_row_bytes["bytes"]
    if not (run_row_path.parent / "receipt.json").exists():
        pending = next(p for p in result["pending_rows"] if p["ordinal"] == 0)
        assert pending["provider_calls"] == 1
        assert pending["dispatched"] and pending["generation_started"]
        assert pending["generation_completed"]
        assert pending["raw_output"] == f.behavior.raw
        assert pending["run_row_bytes_utf8"].encode() == raw
        assert (f.root / "out/pending-rows.json").is_file()
        assert result["rows_saved"] < result["planned_rows"]


def _block_named_output(monkeypatch, root, names):
    original = Path.open

    def open_path(self, mode="r", *args, **kwargs):
        writing = any(flag in mode for flag in "wax+")
        matched = self.name in names or any(
            self.name.startswith(f".{name}.") for name in names
        )
        if writing and self.is_relative_to(root) and matched:
            raise OSError(FIRST + ": " + self.name)
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)


@pytest.mark.parametrize("verified", [True, False])
@pytest.mark.parametrize(
    "name", ["cleanup.json", "final-worker-logs.json", "completion.json", "status.json"]
)
def test_finalization_artifact_write_failures_always_dispose_mutex(
    fixture,  # noqa: F811
    verified,
    name,
    monkeypatch,
):
    f = fixture
    f.behavior.cleanup = verified
    factory = f.dependencies.child_factory

    def child(*args, **kwargs):
        value = factory(*args, **kwargs)
        value.output.mkdir(parents=True, exist_ok=True)
        (value.output / "stdout.log").write_bytes(b"private fictional log")
        return value

    dependencies = replace(f.dependencies, child_factory=child)
    _block_named_output(monkeypatch, f.root / "out", {name})
    result = f.runner._run_plan(f.path, dependencies=dependencies)
    _assert_failed(result)
    assert FIRST in json.dumps(result), (
        "unwritable artifacts must be exposed in returned evidence"
    )
    _assert_mutex(verified)
    if name == "status.json":
        assert len(_generates(f)) == 1
    if verified or name == "status.json":
        assert FIRST in result["stop_reason"]
    else:
        assert "cleanup" in result["stop_reason"].lower()


@pytest.mark.parametrize("verified", [True, False])
def test_storage_settlement_failure_cannot_bypass_mutex_or_mask_failure(
    fixture,  # noqa: F811
    verified,
    monkeypatch,
):
    f = fixture
    f.behavior.cleanup = verified

    def settle(*args, **kwargs):
        raise OSError(FIRST)

    monkeypatch.setattr(support.Ledger, "settle_storage", settle)
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    _assert_failed(result)
    assert FIRST in json.dumps(result)
    _assert_mutex(verified)
    assert len(_generates(f)) == 3


@pytest.mark.parametrize("verified", [True, False])
def test_output_becomes_unwritable_after_attempt_returns_truthful_memory_record(
    fixture,  # noqa: F811
    verified,
    monkeypatch,
):
    f = fixture
    f.behavior.cleanup = verified
    output = f.root / "out"
    original_open = Path.open
    original_write = f.runner._write_json
    blocked = {"value": False}
    preserved = {}

    def open_path(self, mode="r", *args, **kwargs):
        if (
            blocked["value"]
            and self.is_relative_to(output)
            and any(c in mode for c in "wax+")
        ):
            raise OSError("subsequent output filesystem unavailable")
        return original_open(self, mode, *args, **kwargs)

    def write(path, value):
        original_write(path, value)
        if Path(path).name == "run-row.json" and not blocked["value"]:
            preserved["bytes"] = Path(path).read_bytes()
            blocked["value"] = True
            raise OSError(FIRST)

    monkeypatch.setattr(Path, "open", open_path)
    monkeypatch.setattr(f.runner, "_write_json", write)
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    _assert_failed(result)
    _assert_mutex(verified)
    assert FIRST in result["stop_reason"]
    assert result["durability"] in ("partial", "unavailable")
    assert len(_generates(f)) == 1
    assert (output / "rows/000000/run-row.json").read_bytes() == preserved["bytes"]
    pending = next(p for p in result["pending_rows"] if p["ordinal"] == 0)
    assert pending["generation_started"] and pending["raw_output"] == f.behavior.raw
    assert result["persistence_errors"], (
        "never promise an unwritable failure index exists"
    )
