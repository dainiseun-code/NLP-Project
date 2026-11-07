"""Requirement-first coordinator checks: fictional data and pure injected children.

No test in this file launches a subprocess, imports a model framework/tokenizer,
or reads retained study data. Expected behavior is from stage4a-runner-contract.md.
"""

import builtins
import hashlib
import inspect
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import TZPATH

import pytest

from messaging_nlp.application_types import PreparationPacket
from messaging_nlp.context import PublicDataset
from messaging_nlp.contracts import Extraction, ExtractionResult, TechnicalFailure
from messaging_nlp.encoder_cache import (
    ExtractionArtifact,
    retained_identity,
    text_sha256,
    write_artifact,
)
from messaging_nlp.io import canonical_json
from messaging_nlp.preparation import prepare_inputs
from messaging_nlp.prompts import build_prompt
from messaging_nlp.run_contracts import (
    CleanupReceipt,
    DecodePolicy,
    ModelSpec,
    PlannedRow,
    ResourceLimits,
    RunPlan,
    SpendingPolicy,
    TokenReceipt,
    WorkerEvent,
    load_row_receipt,
)
from messaging_nlp.temporal import TimezoneBinding


@pytest.fixture(autouse=True)
def deny_processes_and_optional_models(monkeypatch):
    """Fail even if coordinator error handling swallows a forbidden attempt."""
    attempts = []
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "mlx",
            "mlx_lm",
            "transformers",
            "tokenizers",
            "safetensors",
        }:
            attempts.append(name)
            raise AssertionError(f"optional model import forbidden: {name}")
        return original_import(name, *args, **kwargs)

    def no_process(*args, **kwargs):
        attempts.append("process launch")
        raise AssertionError("coordinator leaf tests cannot launch processes")

    monkeypatch.setattr(builtins, "__import__", import_guard)
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, no_process)
    for name in (
        "system",
        "fork",
        "posix_spawn",
        "posix_spawnp",
        "execv",
        "execve",
        "execvp",
        "execvpe",
    ):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, no_process)
    yield
    assert not attempts, f"forbidden attempts: {attempts}"


def nondraft():
    return canonical_json(
        {
            "decision": {
                "action": "unsupported",
                "selection": None,
                "draft": None,
                "clarification": None,
                "reason": "Cannot send messages.",
            },
            "mentions": [],
        }
    ).decode()


class Ledger:
    """Independent conservative ledger double, no disk or worker process."""

    instances = []
    available = 10000

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.entries = {}
        self.history = []
        type(self).instances.append(self)

    def register_versions(self, policy):
        self.history.append(("versions", policy.assistant_version))

    def reserve(self, kind, seconds, generations=0, storage_bytes=0, refs=None):
        if sum(e[0] for e in self.entries.values()) + seconds > self.available:
            raise ValueError("budget exhausted")
        key = f"reservation-{len(self.entries)}"
        self.entries[key] = [seconds, generations, storage_bytes]
        self.history.append(("reserve", kind, key))
        return key

    def settle(
        self, key, status, seconds, generations=0, storage_bytes=0, never_launched=False
    ):
        self.history.append(("settle", key, status))
        actual = [seconds, generations, storage_bytes]
        if status == "finished":
            self.entries[key] = actual
        elif status == "cancelled" and never_launched:
            self.entries[key] = [0, 0, 0]
        else:
            self.entries[key] = [max(a, b) for a, b in zip(self.entries[key], actual)]

    def charge(self, key):
        return tuple(self.entries[key])

    def settle_storage(self, key, status, *, output_bytes, ledger_start_bytes):
        self.settle(key, status, 0, storage_bytes=output_bytes)

    def snapshot(self):
        return {
            "remaining_seconds": self.available
            - sum(e[0] for e in self.entries.values()),
            "remaining_generations": 3000 - sum(e[1] for e in self.entries.values()),
            "reservations": dict(self.entries),
        }


class Mutex:
    instances = []

    def __init__(self, *args):
        self.acquired = False
        self.releases = []
        type(self).instances.append(self)

    def acquire(self):
        self.acquired = True

    def release(self, cleanup_verified=True):
        self.releases.append(cleanup_verified)
        if cleanup_verified:
            self.acquired = False


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Import lazily: requirements/tests were written before the implementation.
    from messaging_nlp import runner
    from messaging_nlp.run_storage import ref

    Ledger.instances, Mutex.instances = [], []
    monkeypatch.setattr(Ledger, "available", 10000)
    monkeypatch.setattr(runner, "BudgetLedger", Ledger)
    monkeypatch.setattr(runner, "ModelMutex", Mutex)
    request = {
        "message_id": "m",
        "owner_id": "o",
        "text": "Send a message now.",
        "request_time": "2026-10-26T10:00:00+00:00",
        "time_zone": "UTC",
        "context_snapshot_id": "s",
    }
    dataset = PublicDataset.model_validate(
        {
            "owner_id": "o",
            "requests": [request],
            "contacts": [],
            "calendar": [],
            "history": [],
            "snapshots": [
                {
                    "context_snapshot_id": "s",
                    "timestamp": request["request_time"],
                    "record_ids": [],
                }
            ],
        }
    )
    utc = next(Path(p) / "UTC" for p in TZPATH if (Path(p) / "UTC").is_file())
    zone = tmp_path / "utc.tzif"
    zone.write_bytes(utc.read_bytes())
    binding = TimezoneBinding(
        "UTC", zone, hashlib.sha256(zone.read_bytes()).hexdigest()
    )
    identity = retained_identity()  # Identity constants only, no asset opens.
    result = ExtractionResult(
        message_id="m",
        status="ok",
        extraction=Extraction(entities=[]),
        failure=None,
        raw_output=None,
        input_sha256=text_sha256(request["text"]),
        extractor_sha256=identity.digest(),
    )
    write_artifact(
        tmp_path / "extraction.json",
        ExtractionArtifact(
            text=request["text"], identity=identity, result=result, diagnostics={}
        ),
    )
    prepared = prepare_inputs(
        dataset,
        "m",
        root=tmp_path,
        baseline_path="base.json",
        assisted_path="shared.json",
        extraction_path="extraction.json",
        identity=identity,
        binding=binding,
    )
    prompts, tokens, rows = {}, {}, []
    for ordinal, condition in enumerate(("B", "baseline", "A+B")):
        row_id = f"row-{ordinal}"
        sealed = prepared.baseline if condition == "baseline" else prepared.assisted
        prompt = build_prompt(
            PreparationPacket.model_validate_json(sealed.data), condition
        )
        prompts[row_id] = prompt
        prompt_path = tmp_path / f"{row_id}-prompt.json"
        prompt_path.write_bytes(
            canonical_json(
                {
                    "system": prompt.system,
                    "common": prompt.common.decode(),
                    "entities": None
                    if prompt.entities is None
                    else prompt.entities.decode(),
                }
            )
        )
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user.decode()},
        ]
        token = TokenReceipt(
            messages_sha256=hashlib.sha256(canonical_json(messages)).hexdigest(),
            template_sha256="a" * 64,
            model_receipt_sha256="b" * 64,
            token_ids_sha256=hashlib.sha256(
                canonical_json(list(range(20)))
            ).hexdigest(),
            token_count=20,
            token_ids=list(range(20)),
        )
        tokens[row_id] = token
        token_path = tmp_path / f"{row_id}-tokens.json"
        token_path.write_bytes(canonical_json(token.model_dump()))
        rows.append(
            PlannedRow(
                row_id=row_id,
                ordinal=ordinal,
                condition=condition,
                request=dataset.requests[0],
                preparation=ref(tmp_path, tmp_path / sealed.path),
                preparation_failure=None,
                prompt=ref(tmp_path, prompt_path),
                token_receipt=ref(tmp_path, token_path),
            )
        )
    artifact = ref(tmp_path, zone)
    plan = SimpleNamespace(
        run_id="fictional-run",
        output_path="out",
        ledger_path="ledger.json",
        mutex_path="model.lock",
        readiness_path="ready.json",
        approval_path="approval.json",
        opening=artifact,
        usage_reset=artifact,
        revision_authority=None,
        model=ModelSpec(
            path="fictional-model", receipt=artifact, template_sha256="a" * 64
        ),
        decoding=DecodePolicy(),
        resources=ResourceLimits(),
        rows=rows,
        spending=SpendingPolicy(
            startup_seconds=10,
            overhead_seconds=30,
            max_run_seconds=1000,
            storage_bytes=10000000,
            assistant_version="v1",
            assistant_version_refs={"policy": "frozen"},
        ),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b'{"fictional":true}')
    frozen = {
        p.relative_to(tmp_path).as_posix(): p.read_bytes()
        for p in tmp_path.iterdir()
        if p.is_file()
    }
    validated = SimpleNamespace(
        root=tmp_path,
        plan=plan,
        plan_sha256="d" * 64,
        dataset=dataset,
        binding=binding,
        prepared={"m": prepared},
        prompts=prompts,
        token_receipts=tokens,
        frozen_bytes=frozen,
    )
    monkeypatch.setattr(runner, "load_and_validate", lambda *a, **kw: validated)
    monkeypatch.setattr(
        runner, "authorization", lambda *a, **kw: {"status": "authorized"}
    )

    def create_output(validated):
        output = tmp_path / plan.output_path
        output.mkdir(exist_ok=False)
        persisted = RunPlan(
            run_id=plan.run_id,
            output_path=plan.output_path,
            readiness_path=plan.readiness_path,
            approval_path=plan.approval_path,
            opening=artifact,
            usage_reset=artifact,
            public_root="fictional-public",
            owner_id="o",
            timezone_name="UTC",
            timezone=artifact,
            roster=artifact,
            frozen_files=[dict(**artifact.model_dump(), role="other")],
            model=plan.model,
            decoding=plan.decoding,
            resources=plan.resources,
            spending=plan.spending,
            rows=plan.rows,
        )
        plan_bytes = canonical_json(persisted.model_dump())
        (output / "plan.json").write_bytes(plan_bytes)
        validated.plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
        (output / "frozen").mkdir()
        for name, data in validated.frozen_bytes.items():
            (output / "frozen" / name).write_bytes(data)
        return output

    monkeypatch.setattr(runner, "create_output", create_output)
    clock = SimpleNamespace(now=100.0)
    children = []
    behavior = SimpleNamespace(
        raw=nondraft(),
        finish="stop",
        fail=None,
        start_error=None,
        close_error=None,
        cleanup=True,
    )

    class Child:
        def __init__(self, command, output, sensors, limits, **kwargs):
            self.command, self.output = command, output
            self.requests, self.events = [], []
            self.deadlines = []
            self.returncode = 0
            children.append(self)

        def start(self):
            assert (self.output.parent / "planned-rows.jsonl").exists()
            assert (self.output.parent / "run-start.json").exists()
            assert (self.output.parent / "startup-reservations.json").exists()
            assert {h[1] for h in Ledger.instances[0].history if h[0] == "reserve"} == {
                "startup",
                "overhead",
                "storage",
            }
            if behavior.start_error:
                raise behavior.start_error

        def send(self, request, *, deadline=None):
            self.requests.append(request)
            self.deadlines.append((request.kind, deadline, clock.now))
            if request.kind == "load":
                self.events = [
                    WorkerEvent(kind="ready", payload={"runtime": "fictional"})
                ]
            elif request.kind == "generate":
                ordinal = next(r.ordinal for r in rows if r.row_id == request.row_id)
                row_dir = self.output.parent / "rows" / f"{ordinal:06d}"
                intent = json.loads((row_dir / "dispatch-intent.json").read_bytes())
                assert intent["reservation_id"] in Ledger.instances[0].entries
                for previous in rows[:ordinal]:
                    assert (
                        self.output.parent
                        / "rows"
                        / f"{previous.ordinal:06d}"
                        / "receipt.json"
                    ).exists(), "outcome must precede next send"
                self.events = [
                    WorkerEvent(kind="generation_started", row_id=request.row_id),
                    WorkerEvent(
                        kind="fragment",
                        row_id=request.row_id,
                        payload={"text": behavior.raw},
                    ),
                ]
                if not behavior.fail:
                    self.events.append(
                        WorkerEvent(
                            kind="terminal",
                            row_id=request.row_id,
                            payload={
                                "finish_reason": behavior.finish,
                                "generation_tokens": 12,
                                "input_tokens": 20,
                                "elapsed_seconds": 2.0,
                            },
                        )
                    )

        def wait_event(self, deadline):
            if not self.events:
                raise behavior.fail or RuntimeError("unexpected extra wait")
            event = self.events.pop(0)
            clock.now += 1
            return event

        def close(self):
            if behavior.close_error:
                raise behavior.close_error
            return CleanupReceipt(verified=behavior.cleanup, returncode=0)

    dependencies = runner.Dependencies(
        boot=lambda: "darwin-kern-boottime:200:0",
        sensors_factory=lambda root: object(),
        child_factory=Child,
        model_checker=lambda *a: {},
        clock=lambda: clock.now,
    )
    return SimpleNamespace(
        runner=runner,
        path=plan_path,
        root=tmp_path,
        validated=validated,
        dependencies=dependencies,
        children=children,
        behavior=behavior,
        clock=clock,
        monkeypatch=monkeypatch,
    )


def receipts(f):
    output = f.root / "out"
    return [
        load_row_receipt(output, p)[0]
        for p in sorted((output / "rows").glob("*/receipt.json"))
    ]


def test_dry_run_has_no_output_ledger_mutex_or_child(fixture):
    f = fixture
    result = f.runner._run_plan(f.path, dry_run=True, dependencies=f.dependencies)
    assert result["status"] == "authorized"
    assert not f.children and not Ledger.instances and not Mutex.instances
    assert not (f.root / "out").exists()


def test_queued_but_undelivered_fragments_recovered_from_durable_journal(fixture):
    f = fixture
    factory = f.dependencies.child_factory

    def child(*args, **kwargs):
        value = factory(*args, **kwargs)
        original = value.wait_event

        def wait(deadline):
            if (
                value.requests[-1].kind == "generate"
                and value.events[0].kind == "fragment"
            ):
                value.output.mkdir(parents=True, exist_ok=True)
                event = value.events.pop(0)
                (value.output / "raw-fragments.jsonl").write_bytes(
                    canonical_json(event.model_dump()) + b"\n"
                )
                raise RuntimeError("deadline after journal fsync before event delivery")
            return original(deadline)

        value.wait_event = wait
        return value

    result = f.runner._run_plan(
        f.path, dependencies=replace(f.dependencies, child_factory=child)
    )
    assert result["status"] == "failed"
    output = f.root / "out"
    path = sorted((output / "rows").glob("*/receipt.json"))[0]
    receipt, row, _ = load_row_receipt(output, path)
    assert receipt.generation_started and not receipt.generation_completed
    assert row.raw_output == f.behavior.raw
    assert receipt.charged_seconds >= 120


@pytest.mark.parametrize(
    "message",
    [
        "missing approval",
        "unchanged boot",
        "missing boot",
        "mismatched approval",
        "changed boot not consent",
    ],
)
def test_gate_error_retains_all_rows_without_charges_or_child(fixture, message):
    f = fixture

    def reject(*a, **kw):
        raise ValueError(message)

    f.monkeypatch.setattr(f.runner, "authorization", reject)
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "failed"
    assert len(receipts(f)) == 3
    assert all(
        r.disposition == "unattempted_stop"
        and r.provider_calls == 0
        and r.reservation_id is None
        for r in receipts(f)
    )
    assert not Ledger.instances and not f.children
    assert message in (f.root / "out" / "failure.json").read_text()


def test_one_load_frozen_order_durable_rows_and_shared_inputs(fixture):
    f = fixture
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "completed"
    assert len(f.children) == 1
    child = f.children[0]
    assert child.command == [
        str(f.root / "environments/qwen/.venv/bin/python"),
        "-I",
        "-m",
        "messaging_nlp.qwen_worker",
    ]
    assert [r.kind for r in child.requests].count("load") == 1
    assert [r.row_id for r in child.requests if r.kind == "generate"] == [
        "row-0",
        "row-1",
        "row-2",
    ]
    rows = receipts(f)
    assert [r.condition for r in rows] == ["B", "baseline", "A+B"]
    assert all(r.provider_calls == 1 and r.generation_completed for r in rows)
    assert rows[0].bindings["preparation"] == rows[2].bindings["preparation"]
    assert (
        Ledger.instances[0].kwargs["reset_sha256"]
        == f.validated.plan.usage_reset.sha256
    )
    assert not Mutex.instances[0].acquired


def test_completed_invalid_json_settles_actual_not_planned(fixture):
    f = fixture
    f.behavior.raw = "{broken"
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    rows = receipts(f)
    assert len(f.children[0].requests) >= 4
    assert all(
        r.disposition == "attempted_failure"
        and r.generation_completed
        and r.charged_seconds < 120
        and r.charged_generations == 1
        for r in rows
    )


def test_parent_absolute_deadlines_bound_load_and_full_row_dispatch(fixture):
    f = fixture
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    deadlines = f.children[0].deadlines
    assert deadlines[0][0] == "load" and deadlines[0][1] == 110
    assert all(deadline == started + 120 for kind, deadline, started in deadlines[1:])


def test_length_valid_json_is_failed_without_retry(fixture):
    f = fixture
    f.behavior.finish = "length"
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    rows = receipts(f)
    assert all(
        r.disposition == "attempted_failure"
        and r.finish_reason == "length"
        and r.generation_completed
        for r in rows
    )
    for row in rows:
        raw = json.loads((f.root / "out" / row.run_row.path).read_bytes())
        assert raw["status"] == "failed" and raw["decision"] is None


@pytest.mark.parametrize(
    "kind", ["timeout", "sensor_error", "resource_stop", "runtime_error"]
)
def test_partial_infrastructure_failure_halts_and_preserves_reservation(fixture, kind):
    f = fixture
    from messaging_nlp.run_supervision import SupervisionError

    f.behavior.raw = "partial text"
    f.behavior.fail = SupervisionError(kind, "fictional halt")
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    first, *rest = receipts(f)
    assert (
        first.provider_calls == 1
        and first.generation_started
        and not first.generation_completed
    )
    assert first.charged_seconds >= 120 and first.charged_generations >= 1
    assert (f.root / "out" / first.raw_output.path).read_bytes() == b"partial text"
    assert all(
        r.disposition == "unattempted_stop" and r.provider_calls == 0 for r in rest
    )
    assert len([r for r in f.children[0].requests if r.kind == "generate"]) == 1


def test_budget_stop_has_all_rows_and_no_shortened_call(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(Ledger, "available", 150)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    rows = receipts(f)
    assert all(
        r.disposition == "unattempted_stop" and r.provider_calls == 0 for r in rows
    )
    assert not [r for r in f.children[0].requests if r.kind == "generate"]


def test_insufficient_overall_runway_never_dispatches(fixture):
    f = fixture
    f.validated.plan.spending = f.validated.plan.spending.model_copy(
        update={"max_run_seconds": 130}
    )
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert all(r.provider_calls == 0 for r in receipts(f))


@pytest.mark.parametrize("cleanup", [False, "error"])
def test_unverified_cleanup_retains_mutex_and_failure(fixture, cleanup):
    f = fixture
    if cleanup == "error":
        f.behavior.close_error = RuntimeError("fictional close error")
    else:
        f.behavior.cleanup = False
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "failed"
    assert Mutex.instances[0].acquired
    assert (
        json.loads((f.root / "out" / "cleanup.json").read_bytes())["verified"] is False
    )


def test_start_failure_retains_zero_row_calls_and_cleanup(fixture):
    f = fixture
    f.behavior.start_error = RuntimeError("fictional launch error")
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "failed"
    assert all(r.provider_calls == 0 for r in receipts(f))
    assert (f.root / "out" / "cleanup.json").exists()


def test_sensor_failure_discovered_during_verified_close_fails_run_not_rows(fixture):
    f = fixture
    original = f.dependencies.child_factory.close

    def close(self):
        self.failure = {
            "code": "sensor_error",
            "message": "fictional late sample failure",
        }
        return original(self)

    f.monkeypatch.setattr(f.dependencies.child_factory, "close", close)
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "failed" and result["cleanup_verified"]
    assert all(r.disposition == "completed" for r in receipts(f))
    assert not Mutex.instances[0].acquired
    assert (f.root / "out" / "supervisor-failure.json").exists()


def test_existing_destination_is_never_overwritten(fixture):
    f = fixture
    (f.root / "out").mkdir()
    sentinel = f.root / "out" / "sentinel"
    sentinel.write_bytes(b"previous evidence")
    with pytest.raises(FileExistsError):
        f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert sentinel.read_bytes() == b"previous evidence"
    assert not f.children and not Ledger.instances


def test_immutable_outcome_precedes_next_reservation(fixture, monkeypatch):
    f = fixture
    original = Ledger.reserve
    reserved = []

    def reserve(self, kind, seconds, **kwargs):
        if kind == "row":
            if reserved:
                previous = (
                    f.root / "out" / "rows" / f"{reserved[-1]:06d}" / "receipt.json"
                )
                assert previous.exists(), "row must fsync before next reservation"
            reserved.append(len(reserved))
        return original(self, kind, seconds, **kwargs)

    monkeypatch.setattr(Ledger, "reserve", reserve)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert reserved == [0, 1, 2]


def test_blocked_shared_preparations_get_wrappers_without_calls(fixture):
    f = fixture
    failure = TechnicalFailure(
        stage="extraction", code="missing", message="fictional missing extraction"
    )
    for index in (0, 2):
        row = f.validated.plan.rows[index]
        f.validated.plan.rows[index] = row.model_copy(
            update={
                "preparation": None,
                "preparation_failure": failure,
                "prompt": None,
                "token_receipt": None,
            }
        )
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    rows = receipts(f)
    assert [r.disposition for r in rows] == [
        "preparation_blocked",
        "completed",
        "preparation_blocked",
    ]
    assert [r.provider_calls for r in rows] == [0, 1, 0]
    assert [r.row_id for r in f.children[0].requests if r.kind == "generate"] == [
        "row-1"
    ]


def test_exact_canonical_rows_and_resolved_envelopes_retained(fixture):
    f = fixture
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    from messaging_nlp.application_types import OutputEnvelope
    from messaging_nlp.contracts import RunRow

    for receipt in receipts(f):
        raw = (f.root / "out" / receipt.run_row.path).read_bytes()
        row = RunRow.model_validate_json(raw)
        assert canonical_json(row.model_dump()) == raw
        envelope = (f.root / "out" / receipt.resolved_envelope.path).read_bytes()
        assert (
            canonical_json(OutputEnvelope.model_validate_json(envelope).model_dump())
            == envelope
        )
        assert row.raw_output == nondraft()


def test_generation_and_startup_are_not_doublecharged_as_overhead(fixture):
    f = fixture
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    ledger = Ledger.instances[0]
    overhead = next(h[2] for h in ledger.history if h[:2] == ("reserve", "overhead"))
    startup = next(h[2] for h in ledger.history if h[:2] == ("reserve", "startup"))
    assert ledger.charge(startup)[0] == 1
    assert ledger.charge(overhead)[0] == 0
    assert sum(r.charged_seconds for r in receipts(f)) == 9
    for path in (f.root / "out" / "rows").glob("*/timing.json"):
        timing = json.loads(path.read_bytes())
        assert timing["preprocessing_verification_seconds"] == 0
        assert timing["generation_seconds"] == 3


def test_cleanup_time_charged_only_as_overhead(fixture):
    f = fixture
    original = f.dependencies.child_factory.close

    def close(self):
        f.clock.now += 5
        return original(self)

    f.monkeypatch.setattr(f.dependencies.child_factory, "close", close)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    ledger = Ledger.instances[0]
    overhead = next(h[2] for h in ledger.history if h[:2] == ("reserve", "overhead"))
    assert ledger.charge(overhead)[0] == 5


def test_sensor_construction_failure_has_no_child_or_row_charges(fixture):
    f = fixture

    def sensors(root):
        raise RuntimeError("fictional initial sensor error")

    dependencies = replace(f.dependencies, sensors_factory=sensors)
    f.runner._run_plan(f.path, dependencies=dependencies)
    assert not f.children
    assert all(r.reservation_id is None and r.provider_calls == 0 for r in receipts(f))
    assert not Mutex.instances[0].acquired


def test_boot_sensor_failure_is_retained_before_any_ledger(fixture):
    f = fixture

    def boot():
        raise RuntimeError("fictional boot sensor error")

    f.runner._run_plan(f.path, dependencies=replace(f.dependencies, boot=boot))
    assert not f.children and not Ledger.instances
    assert len(receipts(f)) == 3


@pytest.mark.parametrize(
    "bad",
    [
        WorkerEvent(kind="terminal", row_id="row-0", payload={"finish_reason": "stop"}),
        WorkerEvent(
            kind="fragment", row_id="row-0", payload={"text": "unacknowledged"}
        ),
        WorkerEvent(kind="generation_started", row_id="wrong-row"),
        WorkerEvent(kind="ready"),
        WorkerEvent(kind="error", row_id="row-0", payload={"message": "fictional"}),
    ],
)
def test_unexpected_protocol_halts_without_retry(fixture, bad):
    f = fixture
    original = f.dependencies.child_factory.send

    def send(self, request, *, deadline=None):
        original(self, request)
        if request.kind == "generate":
            self.events = [bad]

    f.monkeypatch.setattr(f.dependencies.child_factory, "send", send)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert receipts(f)[0].charged_seconds >= 120
    assert all(r.provider_calls == 0 for r in receipts(f)[1:])
    assert len([r for r in f.children[0].requests if r.kind == "generate"]) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("generation_tokens", 0),
        ("generation_tokens", 1025),
        ("input_tokens", 21),
        ("elapsed_seconds", -1),
        ("finish_reason", "unknown"),
    ],
)
def test_invalid_terminal_metadata_is_infrastructure_failure(fixture, key, value):
    f = fixture
    original = f.dependencies.child_factory.send

    def send(self, request, *, deadline=None):
        original(self, request)
        if request.kind == "generate":
            terminal = self.events[-1]
            self.events[-1] = terminal.model_copy(
                update={"payload": {**terminal.payload, key: value}}
            )

    f.monkeypatch.setattr(f.dependencies.child_factory, "send", send)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert not receipts(f)[0].generation_completed
    assert receipts(f)[0].charged_seconds >= 120
    assert all(r.provider_calls == 0 for r in receipts(f)[1:])


def test_send_failure_keeps_conservative_reservation_and_intent(fixture):
    f = fixture
    original = f.dependencies.child_factory.send

    def send(self, request, *, deadline=None):
        if request.kind == "generate":
            raise RuntimeError("fictional broken pipe")
        return original(self, request)

    f.monkeypatch.setattr(f.dependencies.child_factory, "send", send)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    first = receipts(f)[0]
    assert (
        first.dispatch_intent and not first.dispatched and not first.generation_started
    )
    assert first.provider_calls == 1 and first.charged_seconds >= 120


def test_stalled_prefill_times_out_without_claiming_model_generation_started(fixture):
    f = fixture
    from messaging_nlp.run_supervision import SupervisionError

    original = f.dependencies.child_factory.send

    def send(self, request, *, deadline=None):
        original(self, request)
        if request.kind == "generate":
            self.events = []

    f.behavior.fail = SupervisionError("timeout", "fictional stalled prefill")
    f.monkeypatch.setattr(f.dependencies.child_factory, "send", send)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    first, *rest = receipts(f)
    assert first.provider_calls == 1 and first.dispatched
    assert not first.generation_started and not first.generation_completed
    assert first.charged_seconds >= 120 and first.charged_generations == 1
    assert all(r.provider_calls == 0 for r in rest)


def test_interrupt_retains_partial_and_unexecuted_rows(fixture):
    f = fixture
    f.behavior.raw, f.behavior.fail = "interrupted partial", KeyboardInterrupt()
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "failed"
    first, *rest = receipts(f)
    assert first.generation_started and not first.generation_completed
    assert first.charged_seconds >= 120
    assert all(r.provider_calls == 0 for r in rest)
    assert not Mutex.instances[0].acquired


def test_interrupt_after_generation_does_not_doublecharge_row_as_overhead(fixture):
    f = fixture
    original = f.runner.execute_condition

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt()

    f.monkeypatch.setattr(f.runner, "execute_condition", interrupted)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    first, *rest = receipts(f)
    assert first.generation_completed and first.charged_seconds == 3
    assert first.disposition == "attempted_failure"
    assert all(r.provider_calls == 0 for r in rest)
    ledger = Ledger.instances[0]
    overhead = next(h[2] for h in ledger.history if h[:2] == ("reserve", "overhead"))
    assert ledger.charge(overhead)[0] == 0


def test_mutex_acquisition_failure_never_releases_someone_elses_lock(fixture):
    f = fixture

    def acquire(self):
        raise RuntimeError("fictional existing owner")

    f.monkeypatch.setattr(Mutex, "acquire", acquire)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert Mutex.instances[0].releases == []
    assert not Ledger.instances and not f.children


def test_public_api_has_no_substitution_or_restart_bypass(fixture):
    signature = inspect.signature(fixture.runner.run_plan)
    assert list(signature.parameters) == ["plan_path", "dry_run", "approve_run"]


def test_authorization_rechecked_under_mutex_before_ledger(fixture):
    f = fixture
    calls = []

    def authorization(*a, **kw):
        calls.append(bool(Mutex.instances and Mutex.instances[0].acquired))
        if len(calls) == 2:
            raise ValueError("fictional ledger changed before mutex acquisition")
        return {"status": "authorized"}

    f.monkeypatch.setattr(f.runner, "authorization", authorization)
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert calls == [False, True]
    assert result["status"] == "failed"
    assert not Ledger.instances and not f.children
    assert all(r.provider_calls == 0 for r in receipts(f))


def test_row_log_refs_are_immutable_slices_not_global_mutable_hashes(fixture):
    f = fixture
    original_send = f.dependencies.child_factory.send
    original_close = f.dependencies.child_factory.close

    def send(self, request, *, deadline=None):
        self.output.mkdir(exist_ok=True)
        with (self.output / "stdout.log").open("ab") as handle:
            handle.write(f"{request.kind}:{request.row_id}\n".encode())
        return original_send(self, request)

    def close(self):
        with (self.output / "stdout.log").open("ab") as handle:
            handle.write(b"cleanup\n")
        return original_close(self)

    f.monkeypatch.setattr(f.dependencies.child_factory, "send", send)
    f.monkeypatch.setattr(f.dependencies.child_factory, "close", close)
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    rows = receipts(f)
    for row in rows:
        artifact = row.logs["stdout.log"]
        raw = (f.root / "out" / artifact.path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact.sha256
    assert (
        f.root / "out" / rows[1].logs["stdout.log"].path
    ).read_bytes() == b"generate:row-1\n"
    assert (
        f.root / "out" / rows[2].logs["stdout.log"].path
    ).read_bytes() == b"generate:row-2\n"
    final = json.loads((f.root / "out" / "final-worker-logs.json").read_bytes())
    assert final["cleanup_verified"] and not final["partial"]
    log = final["logs"]["stdout.log"]
    raw = (f.root / "out" / log["path"]).read_bytes()
    assert (
        raw.endswith(b"cleanup\n") and hashlib.sha256(raw).hexdigest() == log["sha256"]
    )


def test_prompt_parts_aliases_extraction_and_timing_are_explicit_bindings(fixture):
    f = fixture
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    for row in receipts(f):
        assert {"system", "common", "user", "messages", "aliases", "timing"} <= set(
            row.bindings
        )
        if row.condition == "A+B":
            assert "entities" in row.bindings and "extraction" in row.bindings
        if row.condition == "B":
            assert "entities" not in row.bindings and "extraction" in row.bindings


@pytest.fixture
def integrated(tmp_path):
    from run_fixtures import fake_plan
    from test_run_plan import permit

    from messaging_nlp import runner

    path, plan = fake_plan(tmp_path)
    permit(path, plan)
    children, clock = [], SimpleNamespace(now=100.0)

    class Child:
        def __init__(self, command, output, sensors, limits, **kwargs):
            self.requests, self.events = [], []
            children.append(self)

        def start(self):
            pass

        def send(self, request, *, deadline=None):
            self.requests.append(request)
            if request.kind == "load":
                self.events = [
                    WorkerEvent(kind="ready", payload={"runtime": "fictional"})
                ]
            elif request.kind == "generate":
                self.events = [
                    WorkerEvent(kind="generation_started", row_id=request.row_id),
                    WorkerEvent(
                        kind="fragment",
                        row_id=request.row_id,
                        payload={"text": nondraft()},
                    ),
                    WorkerEvent(
                        kind="terminal",
                        row_id=request.row_id,
                        payload={
                            "finish_reason": "stop",
                            "generation_tokens": 8,
                            "input_tokens": 3,
                            "elapsed_seconds": 0.1,
                        },
                    ),
                ]

        def wait_event(self, deadline):
            clock.now += 0.1
            return self.events.pop(0)

        def close(self):
            return CleanupReceipt(verified=True, returncode=0)

    dependencies = runner.Dependencies(
        boot=lambda: "darwin-kern-boottime:200:0",
        sensors_factory=lambda root: object(),
        child_factory=Child,
        model_checker=lambda root, model: {"fictional": True},
        clock=lambda: clock.now,
    )
    return SimpleNamespace(
        runner=runner,
        path=path,
        plan=plan,
        root=tmp_path,
        children=children,
        dependencies=dependencies,
    )


def integrated_receipts(f):
    output = f.root / f.plan.output_path
    return [
        load_row_receipt(output, p)[0]
        for p in sorted((output / "rows").glob("*/receipt.json"))
    ]


def test_actual_plan_and_ledger_integration_is_model_and_process_free(integrated):
    f = integrated
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "completed"
    assert [r.provider_calls for r in integrated_receipts(f)] == [0, 1, 0]
    assert [r.disposition for r in integrated_receipts(f)] == [
        "preparation_blocked",
        "completed",
        "preparation_blocked",
    ]
    assert len(f.children) == 1
    budget = json.loads(
        (f.root / f.plan.output_path / "budget-final.json").read_bytes()
    )
    assert budget["remaining_generations"] == 2999


@pytest.mark.parametrize(
    "gate", ["approval", "unchanged_boot", "missing_ledger", "stale_ledger"]
)
def test_actual_authorization_failures_leave_existing_budget_unchanged(
    integrated, gate
):
    f = integrated
    dependencies = f.dependencies
    ledger = f.root / f.plan.ledger_path
    if gate == "approval":
        (f.root / f.plan.approval_path).unlink()
    elif gate == "unchanged_boot":
        dependencies = replace(dependencies, boot=lambda: "darwin-kern-boottime:100:0")
    elif gate == "missing_ledger":
        ledger.unlink()
    else:
        ledger.write_bytes(ledger.read_bytes() + b" ")
    before = ledger.read_bytes() if ledger.exists() else None
    result = f.runner._run_plan(f.path, dependencies=dependencies)
    assert result["status"] == "failed"
    assert not f.children
    assert all(
        r.provider_calls == 0 and r.reservation_id is None
        for r in integrated_receipts(f)
    )
    assert (ledger.read_bytes() if ledger.exists() else None) == before


@pytest.mark.parametrize(
    "target",
    ["src/messaging_nlp/application.py", "prep/base.json", "public/requests.jsonl"],
)
def test_actual_validation_failure_retains_observed_inputs_and_all_rows(
    integrated, target
):
    f = integrated
    path = f.root / target
    if target.startswith("src"):
        path.write_bytes(path.read_bytes() + b"\n# mutated fictional source\n")
    else:
        path.unlink()
    ledger = f.root / f.plan.ledger_path
    before = ledger.read_bytes()
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "failed"
    assert len(integrated_receipts(f)) == 3
    assert all(
        r.provider_calls == 0 and r.reservation_id is None
        for r in integrated_receipts(f)
    )
    assert not f.children and ledger.read_bytes() == before
    assert (f.root / f.plan.output_path / "failure.json").exists()


def test_actual_dry_run_never_creates_outputs_or_mutates_ledger(integrated):
    f = integrated
    ledger = f.root / f.plan.ledger_path
    before = ledger.read_bytes()
    result = f.runner._run_plan(f.path, dry_run=True, dependencies=f.dependencies)
    assert result["status"] == "ready" and result["execution_authorized"] is False
    assert not f.children and ledger.read_bytes() == before
    assert not (f.root / f.plan.output_path).exists()


@pytest.mark.parametrize("cleanup", [True, False])
@pytest.mark.parametrize("fault", ["charge", "receipt", "run_row", "logs"])
def test_partial_row_commit_never_retries_or_masks_first_error(fixture, cleanup, fault):
    f = fixture
    f.behavior.cleanup = cleanup
    original_write = f.runner._write_json
    writes = []

    def write(path, value):
        if path.name == "run-row.json" and path.parent.name == "000000":
            writes.append(canonical_json(value))
            original_write(path, value)
            if fault == "run_row":
                raise OSError("FIRST: run row fsync uncertain")
            return
        return original_write(path, value)

    def reject(*args, **kwargs):
        raise OSError(f"FIRST: {fault}")

    f.monkeypatch.setattr(f.runner, "_write_json", write)
    if fault == "charge":
        f.monkeypatch.setattr(Ledger, "charge", reject)
    elif fault == "receipt":
        f.monkeypatch.setattr(f.runner, "RowReceipt", reject)
    elif fault == "logs":
        f.monkeypatch.setattr(f.runner, "_snapshot_logs", reject)
    final = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert final["status"] == "failed" and "FIRST:" in final["stop_reason"]
    assert Mutex.instances[0].releases == [cleanup]
    assert Mutex.instances[0].acquired is not cleanup
    assert len(writes) == 1
    assert (f.root / "out/rows/000000/run-row.json").read_bytes() == writes[0]
    pending = final["pending_rows"][0]
    assert pending["row_id"] == "row-0" and pending["generation_completed"]
    assert pending["raw_output"] == nondraft()
    assert pending["run_row_bytes_utf8"].encode() == writes[0]
    assert json.loads(pending["run_row_bytes_utf8"])["status"] == "ok"
    assert pending["persistence_errors"]
    assert final["rows_saved"] + len(final["pending_rows"]) == 3
    assert len([r for r in f.children[0].requests if r.kind == "generate"]) == 1
    assert (
        json.loads((f.root / "out/completion.json").read_bytes())["status"] == "failed"
    )
    assert json.loads((f.root / "out/pending-rows.json").read_bytes())[0] == pending


@pytest.mark.parametrize(
    "artifact", ["cleanup.json", "completion.json", "status.json", "budget-final.json"]
)
def test_finalization_evidence_fault_still_disposes_mutex(fixture, artifact):
    f = fixture
    original_write, original_atomic = f.runner._write_json, f.runner.atomic_json

    def wrap(original):
        def write(path, value):
            if path.name == artifact:
                raise OSError(f"FIRST: unwritable {artifact}")
            return original(path, value)

        return write

    f.monkeypatch.setattr(f.runner, "_write_json", wrap(original_write))
    f.monkeypatch.setattr(f.runner, "atomic_json", wrap(original_atomic))
    final = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert final["status"] == "failed" and artifact in final["stop_reason"]
    assert Mutex.instances[0].releases == [True]
    assert not Mutex.instances[0].acquired
    assert final["persistence_errors"]
    assert final["durability"] == "partial"


def test_provider_halt_precedes_row_and_failure_log_faults(fixture):
    f = fixture
    f.behavior.fail = RuntimeError("FIRST provider halt")
    original = f.runner._write_json

    def write(path, value):
        if path.name in {"failure.json", "run-row.json"}:
            raise OSError("SECOND evidence fault")
        return original(path, value)

    f.monkeypatch.setattr(f.runner, "_write_json", write)
    final = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert "FIRST provider halt" in final["stop_reason"]
    assert final["pending_rows"][0]["raw_output"] == nondraft()
    assert not Mutex.instances[0].acquired
    assert len([r for r in f.children[0].requests if r.kind == "generate"]) == 1


def test_runtime_ready_input_verification_failure_is_preparation_blocked(fixture):
    from messaging_nlp import application

    f = fixture

    def reject(*args, **kwargs):
        f.clock.now += 2
        raise ValueError("FIRST runtime verify_input drift")

    f.monkeypatch.setattr(application, "verify_input", reject)
    final = f.runner._run_plan(f.path, dependencies=f.dependencies)
    first, *rest = receipts(f)
    assert final["status"] == "failed"
    assert first.disposition == "preparation_blocked" and first.provider_calls == 0
    assert first.reservation_id is None and first.charged_generations == 0
    assert all(r.disposition == "unattempted_stop" for r in rest)
    row = json.loads((f.root / "out" / first.run_row.path).read_bytes())
    assert row["status"] == "unexecuted"
    assert "FIRST runtime verify_input drift" in row["failure"]["message"]
    assert "FIRST runtime verify_input drift" in final["stop_reason"]
    assert [r.kind for r in f.children[0].requests] == ["load"]
    ledger = Ledger.instances[0]
    assert not [h for h in ledger.history if h[:2] == ("reserve", "row")]
    startup = next(h[2] for h in ledger.history if h[:2] == ("reserve", "startup"))
    overhead = next(h[2] for h in ledger.history if h[:2] == ("reserve", "overhead"))
    assert ledger.charge(startup)[0] == 1 and ledger.charge(overhead)[0] >= 2


def test_zero_terminal_tokens_retains_raw_and_maximum_planned_charge(fixture):
    f = fixture
    original = f.dependencies.child_factory.send

    def send(self, request, *, deadline=None):
        original(self, request, deadline=deadline)
        if request.kind == "generate":
            event = self.events[-1]
            self.events[-1] = event.model_copy(
                update={"payload": {**event.payload, "generation_tokens": 0}}
            )

    f.monkeypatch.setattr(f.dependencies.child_factory, "send", send)
    final = f.runner._run_plan(f.path, dependencies=f.dependencies)
    first, *rest = receipts(f)
    assert final["status"] == "failed" and not first.generation_completed
    assert first.charged_seconds >= 120 and first.charged_generations == 1
    assert (f.root / "out" / first.raw_output.path).read_text() == nondraft()
    assert all(r.provider_calls == 0 for r in rest)


def test_consent_conflicting_flags_rejected_before_dependencies_or_validation(fixture):
    f = fixture

    def forbidden(*a, **kw):
        pytest.fail("conflicting flags must not touch dependencies or validation")

    f.monkeypatch.setattr(f.runner, "load_and_validate", forbidden)
    with pytest.raises(ValueError):
        f.runner._run_plan(
            f.path,
            dry_run=True,
            approve_run=True,
            dependencies=replace(f.dependencies, clock=forbidden),
        )
    assert not (f.root / "out").exists()


def test_consent_full_validation_failure_has_no_fallback_or_side_effects(fixture):
    f = fixture

    def reject(*a, **kw):
        raise ValueError("full validation failed")

    def forbidden(*a, **kw):
        pytest.fail("consent must not fall back or create output")

    f.monkeypatch.setattr(f.runner, "load_and_validate", reject)
    f.monkeypatch.setattr(f.runner, "_failed_validation_plan", forbidden)
    f.monkeypatch.setattr(f.runner, "create_output", forbidden)
    with pytest.raises(ValueError, match="full validation failed"):
        f.runner._run_plan(f.path, approve_run=True, dependencies=f.dependencies)
    assert not f.children and not Ledger.instances and not Mutex.instances


def test_consent_calls_helper_with_current_boot_before_any_output_or_budget(fixture):
    f = fixture
    seen = []

    def approve(validated, boot):
        assert validated is f.validated
        assert boot == f.dependencies.boot()
        assert not (f.root / "out").exists()
        assert not f.children and not Ledger.instances and not Mutex.instances
        seen.append(boot)
        return {"status": "approved fictional seam only"}

    f.monkeypatch.setattr(f.runner, "approve_execution", approve)
    final = f.runner._run_plan(f.path, approve_run=True, dependencies=f.dependencies)
    assert seen == [f.dependencies.boot()] and final["status"] == "completed"
    assert not (f.root / "approval.json").exists()


def test_fragment_known_before_event_log_failure_is_retained(fixture):
    f = fixture
    original = f.runner.append_bytes

    def append(path, data):
        if path.name == "events.jsonl" and b'"kind":"fragment"' in data:
            raise OSError("FIRST fragment event journal failure")
        return original(path, data)

    f.monkeypatch.setattr(f.runner, "append_bytes", append)
    final = f.runner._run_plan(f.path, dependencies=f.dependencies)
    first = receipts(f)[0]
    row = json.loads((f.root / "out" / first.run_row.path).read_bytes())
    assert final["status"] == "failed"
    assert row["raw_output"] == nondraft()
    assert not Mutex.instances[0].acquired


def test_failed_startup_is_conservatively_settled_and_not_refunded(fixture):
    f = fixture
    f.behavior.start_error = RuntimeError("FIRST launch failure")
    f.runner._run_plan(f.path, dependencies=f.dependencies)
    ledger = Ledger.instances[0]
    startup = next(h[2] for h in ledger.history if h[:2] == ("reserve", "startup"))
    assert ("settle", startup, "failed") in ledger.history
    assert ledger.charge(startup)[0] >= f.validated.plan.spending.startup_seconds


def test_transient_final_status_failure_does_not_leave_completed_summary(fixture):
    f = fixture
    original = f.runner.atomic_json
    blocked = []

    def atomic(path, value):
        if path.name == "status.json" and value["status"] == "completed":
            blocked.append(path)
            raise OSError("FIRST final status failure")
        return original(path, value)

    f.monkeypatch.setattr(f.runner, "atomic_json", atomic)
    final = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert final["status"] == "failed" and len(blocked) == 1
    assert (
        json.loads((f.root / "out/completion.json").read_bytes())["status"] == "failed"
    )
    assert not Mutex.instances[0].acquired


def test_provider_halt_survives_failure_settlement_error_with_original_raw(fixture):
    f = fixture
    f.behavior.fail = RuntimeError("FIRST provider infrastructure halt")
    original = Ledger.settle

    def settle(self, key, status, seconds, **kwargs):
        if status == "failed" and self.entries[key][1]:
            raise OSError("SECOND row failure settlement unavailable")
        return original(self, key, status, seconds, **kwargs)

    f.monkeypatch.setattr(Ledger, "settle", settle)
    final = f.runner._run_plan(f.path, dependencies=f.dependencies)
    first = receipts(f)[0]
    row = json.loads((f.root / "out" / first.run_row.path).read_bytes())
    assert "FIRST provider infrastructure halt" in final["stop_reason"]
    assert "FIRST provider infrastructure halt" in row["failure"]["message"]
    assert row["raw_output"] == nondraft()
    assert "SECOND row failure settlement unavailable" in json.dumps(final["errors"])
    assert first.charged_seconds >= 120 and first.charged_generations == 1
    assert len([r for r in f.children[0].requests if r.kind == "generate"]) == 1
    assert not Mutex.instances[0].acquired
