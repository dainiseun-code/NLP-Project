"""Bounded Stage5 preparation; never loads a model in the coordinator.

Actual execution requires the primary's Stage5 authorization and serial lease.
Private dependency seams are fictional tests, not CLI overrides or new study arms.
"""

from __future__ import annotations

import math
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from messaging_nlp.application_types import PreparationPacket
from messaging_nlp.context import load_public_dataset
from messaging_nlp.encoder_cache import (
    ExtractionArtifact,
    read_artifact,
    retained_identity,
    write_artifact,
)
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.paths import ensure_output_disjoint
from messaging_nlp.preparation import digest, policy_identity, prepare_inputs
from messaging_nlp.prompts import build_prompt
from messaging_nlp.run_budget import BudgetLedger
from messaging_nlp.run_contracts import (
    LEDGER_PATH,
    MUTEX_PATH,
    OPENING_SHA256,
    QWEN_RECEIPT_SHA256,
    RESET_SHA256,
    ArtifactRef,
    DecodePolicy,
    FrozenFile,
    ModelSpec,
    PlannedRow,
    ReadinessReceipt,
    ResourceLimits,
    RunPlan,
    SpendingPolicy,
    TokenReceipt,
    WorkerRequest,
)
from messaging_nlp.run_plan import load_and_validate
from messaging_nlp.run_storage import (
    append_bytes,
    atomic_json,
    read_ref,
    ref,
    write_new,
)
from messaging_nlp.run_supervision import (
    ChildSession,
    MacSensors,
    ModelMutex,
    boot_identity,
    check_sample,
)
from messaging_nlp.temporal import TimezoneBinding

REGISTRY = "runs/study-state/encoder-attempts.jsonl"
_ACCOUNTED_DESTINATIONS: set[Path] = set()


@dataclass(frozen=True)
class _Dependencies:
    boot: Callable = boot_identity
    child: Callable = ChildSession
    sensors: Callable = MacSensors
    ledger: Callable = BudgetLedger
    mutex: Callable = ModelMutex
    clock: Callable = time.monotonic


def _destination(root, destination):
    root = Path(root).resolve(strict=True)
    destination = Path(destination).absolute()
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("preparation destination cannot traverse symlinks")
    destination = destination.resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("preparation destination must remain inside project root")
    protected = [
        root / p
        for p in (
            "src",
            "data",
            "models",
            "configs",
            "docs",
            "environments",
            "prompts",
            ".git",
            "Old-version",
            "pyproject.toml",
            "uv.lock",
            "runs/study-state",
        )
    ]
    ensure_output_disjoint(destination, protected)
    if destination.exists():
        raise FileExistsError("fresh preparation destination required; no replay")
    return root, destination


def _ledger(root, dependencies, revision_authority=None):
    return dependencies.ledger(
        root / LEDGER_PATH,
        root / "docs/stage1-budget-opening.json",
        OPENING_SHA256,
        reset_path=root / "docs/budget-usage-reset-20261027.json",
        reset_sha256=RESET_SHA256,
        **(
            dict(
                revision_path=root / revision_authority.path,
                revision_sha256=revision_authority.sha256,
            )
            if revision_authority is not None
            else {}
        ),
    )


def preparation_sources(root, public_root):
    """Explicit inference-only byte inventory: never evaluator gold."""
    root = Path(root)
    paths = [
        p.relative_to(root).as_posix()
        for p in sorted((root / "src/messaging_nlp").rglob("*.py"))
    ]
    paths += [
        "pyproject.toml",
        "uv.lock",
        "environments/encoder/pyproject.toml",
        "environments/encoder/uv.lock",
        "environments/qwen/pyproject.toml",
        "environments/qwen/uv.lock",
        "configs/study-policy.json",
        "docs/stage1-budget-opening.json",
        "docs/budget-usage-reset-20261027.json",
    ]
    paths += [
        f"{public_root}/{name}.jsonl"
        for name in ("requests", "contacts", "calendar", "history", "snapshots")
    ]
    return [ref(root, root / name) for name in paths]


def _source_map(sources):
    prefix = "src/messaging_nlp/"
    return {
        r.path[len(prefix) :]: r.sha256 for r in sources if r.path.startswith(prefix)
    }


def _attempt_key(payload):
    return digest(
        canonical_json(
            {
                "message_id": payload["message_id"],
                "text": payload["text"],
                "extractor_sha256": retained_identity().digest(),
            }
        )
    )


def _registered(root):
    path = root / REGISTRY
    if not path.exists():
        return set()
    values = [loads_json(line) for line in path.read_text().splitlines()]
    keys = [value["attempt_key"] for value in values]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "duplicate encoder registry intent; explicit reconciliation required"
        )
    return set(keys)


def _consume(child, item_id, deadline, progress):
    fragments = []
    while True:
        event = child.wait_event(deadline)
        if event.row_id != item_id:
            raise ValueError("preparation event belongs to another item")
        if event.kind == "error":
            raise RuntimeError(f"preparation child error: {event.payload}")
        if event.kind == "generation_started":
            if (
                progress["operation_started"]
                or event.payload.get("qwen_generation") is not False
            ):
                raise ValueError("invalid preparation start acknowledgement")
            progress["operation_started"] = True
        elif event.kind == "fragment":
            if not progress["operation_started"] or fragments:
                raise ValueError("preparation requires one post-start complete reply")
            fragments.append(event.payload["text"])
        elif event.kind == "terminal":
            if (
                not progress["operation_started"]
                or len(fragments) != 1
                or event.payload.get("finish_reason") != "stop"
                or type(event.payload.get("generation_tokens")) is not int
                or event.payload["generation_tokens"] != 0
            ):
                raise ValueError(
                    "preparation is not a Qwen generation or partial result"
                )
            progress["terminal"] = event.payload
            return fragments[0].encode("utf-8")
        else:
            raise ValueError("unexpected preparation event")


def _accept_reply(root, destination, mode, payload, raw, model):
    value = loads_json(raw.decode())
    if value.get("mode") != mode or canonical_json(value) != raw:
        raise ValueError("noncanonical or foreign preparation reply")
    if mode == "encoder":
        if set(value) != {"mode", "artifact"}:
            raise ValueError("unexpected encoder reply fields")
        artifact = ExtractionArtifact.model_validate(value["artifact"])
        path = destination / "extraction.json"
        write_artifact(path, artifact)
        checked = read_artifact(
            path,
            message_id=payload["message_id"],
            text=payload["text"],
            identity=retained_identity(),
        )
        return dict(
            artifact=ref(root, path).model_dump(),
            result_status=checked.result.status,
            forward_performed=checked.diagnostics.get("forward_performed"),
        )
    if set(value) != {
        "mode",
        "messages_sha256",
        "model_receipt_sha256",
        "template_sha256",
        "token_ids",
        "token_count",
        "token_ids_sha256",
        "fits",
    }:
        raise ValueError("unexpected tokenizer reply fields")
    if (
        value.get("messages_sha256") != payload["messages_sha256"]
        or value.get("model_receipt_sha256") != model.receipt.sha256
        or value.get("template_sha256") != model.template_sha256
    ):
        raise ValueError("tokenizer result binding mismatch")
    ids = value.get("token_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(i) is not int or i < 0 for i in ids)
        or type(value.get("token_count")) is not int
        or value["token_count"] != len(ids)
        or value.get("token_ids_sha256") != digest(canonical_json(ids))
        or type(value.get("fits")) is not bool
        or value["fits"] != (len(ids) <= 2048)
    ):
        raise ValueError("invalid complete tokenizer result")
    if not value["fits"]:
        return dict(
            result_status="overlength", token_count=len(ids), token_receipt=None
        )
    receipt = TokenReceipt(
        **{
            key: value[key]
            for key in (
                "messages_sha256",
                "template_sha256",
                "model_receipt_sha256",
                "token_ids",
                "token_count",
                "token_ids_sha256",
            )
        }
    )
    path = destination / "token-receipt.json"
    write_new(path, canonical_json(receipt.model_dump()))
    return dict(
        result_status="fits",
        token_count=len(ids),
        token_receipt=ref(root, path).model_dump(),
    )


def _run_job(
    root,
    destination,
    *,
    mode,
    items,
    sources,
    model=None,
    max_seconds=1800,
    _dependencies=None,
):
    """One serial lease, once-only encoder intents, durable partial failure state."""
    dependencies = _dependencies or _Dependencies()
    root, destination = _destination(root, destination)
    if mode not in ("encoder", "tokenizer") or (mode == "tokenizer") != (
        model is not None
    ):
        raise ValueError("preparation mode/model mismatch")
    ids = [item["id"] for item in items]
    if not items or len(ids) != len(set(ids)) or max_seconds <= 10:
        raise ValueError(
            "nonempty unique items and cleanup-inclusive deadline required"
        )
    for item in items:
        WorkerRequest(kind="generate", row_id=item["id"], payload=item["payload"])
        if mode == "encoder" and set(item["payload"]) != {"message_id", "text"}:
            raise ValueError("encoder accepts only exact current request")
    if mode == "encoder" and len(
        {item["payload"]["message_id"] for item in items}
    ) != len(items):
        raise ValueError("duplicate encoder request; no repeated inference")
    (root / "runs/study-state").mkdir(parents=True, exist_ok=True)
    mutex = dependencies.mutex(root / MUTEX_PATH, f"stage5-{mode}", dependencies.boot())
    child = ledger = None
    cleanup_verified = True
    reservation = storage = None
    started = dependencies.clock()
    ledger_start_bytes = (
        (root / LEDGER_PATH).stat().st_size if (root / LEDGER_PATH).exists() else 0
    )
    registry_start_bytes = (
        (root / REGISTRY).stat().st_size if (root / REGISTRY).exists() else 0
    )
    errors = []
    outcomes = [
        dict(
            id=item["id"],
            ordinal=i,
            disposition="unattempted",
            operation_started=False,
            payload=item["payload"],
        )
        for i, item in enumerate(items)
    ]

    def fail(error):
        errors.append(f"{type(error).__name__}: {error}")

    mutex.acquire()
    try:
        try:
            destination.mkdir(parents=True, exist_ok=False)
            if mode == "encoder" and any(
                _attempt_key(item["payload"]) in _registered(root) for item in items
            ):
                raise ValueError(
                    "selected encoder request already attempted/reserved; no retry"
                )
            for source in sources:
                data = read_ref(root, source)
                write_new(destination / "sealed" / source.path, data)
            write_new(
                destination / "job.json",
                canonical_json(
                    dict(
                        schema_version="nlp3-preparation-job-v1",
                        mode=mode,
                        items=items,
                        sources=[s.model_dump() for s in sources],
                        encoder_identity=retained_identity().model_dump(),
                        model=model.model_dump() if model else None,
                        max_seconds=max_seconds,
                        qwen_generations=0,
                    )
                ),
            )
            ledger = _ledger(root, dependencies)
            if mode == "encoder":
                old = ledger.snapshot()["prompt_versions"]["assistant_prompt"][-1]
                ledger.register_versions(
                    dict(
                        assistant_version=old["version_id"],
                        assistant_version_refs=old["refs"],
                        extraction_version="retained-xlmr-"
                        + retained_identity().digest()[:16],
                        extraction_version_refs={
                            "extractor_sha256": retained_identity().digest()
                        },
                    )
                )
            refs = {
                "phase": f"stage5-{mode}",
                "destination": destination.relative_to(root).as_posix(),
                "run_id": f"stage5-{mode}-"
                + digest(destination.relative_to(root).as_posix().encode())[:16],
            }
            reservation = ledger.reserve("preparation", max_seconds, refs=refs)
            storage = ledger.reserve("storage", 0, storage_bytes=268435456, refs=refs)
            environment = (
                root / "environments" / ("encoder" if mode == "encoder" else "qwen")
            )
            command = [
                str(environment / ".venv/bin/python"),
                "-I",
                "-m",
                "messaging_nlp.preparation_worker",
            ]
            deadline = started + max_seconds - 10
            child = dependencies.child(
                command,
                destination / "worker",
                dependencies.sensors(root),
                ResourceLimits(),
            )
            cleanup_verified = False
            child.start()
            child.send(
                WorkerRequest(
                    kind="load",
                    payload=dict(
                        mode=mode,
                        root=str(root),
                        model=model.model_dump() if model else None,
                    ),
                ),
                deadline=min(deadline, dependencies.clock() + 300),
            )
            ready = child.wait_event(min(deadline, dependencies.clock() + 300))
            if (
                ready.kind != "ready"
                or ready.payload.get("mode") != mode
                or ready.payload.get("worker_source_sha256") != _source_map(sources)
            ):
                raise ValueError(
                    f"preparation load/installed source failed: {ready.model_dump()}"
                )
            if (
                mode == "encoder"
                and ready.payload.get("encoder_sha256") != retained_identity().digest()
            ):
                raise ValueError("loaded encoder identity mismatch")
            write_new(destination / "ready.json", canonical_json(ready.model_dump()))
            for item, progress in zip(items, outcomes, strict=True):
                for source in sources:
                    read_ref(root, source)
                directory = destination / "items" / f"{progress['ordinal']:06d}"
                write_new(directory / "intent.json", canonical_json(progress))
                if mode == "encoder":
                    append_bytes(
                        root / REGISTRY,
                        canonical_json(
                            dict(
                                attempt_key=_attempt_key(item["payload"]),
                                destination=directory.relative_to(root).as_posix(),
                                message_id=item["payload"]["message_id"],
                                text_sha256=digest(item["payload"]["text"].encode()),
                                observation="intent before dispatch; not proof of forward",
                            )
                        )
                        + b"\n",
                    )
                progress["disposition"] = "attempted_unresolved"
                row_started = dependencies.clock()
                row_deadline = min(deadline, row_started + 120)
                child.send(
                    WorkerRequest(
                        kind="generate", row_id=item["id"], payload=item["payload"]
                    ),
                    deadline=row_deadline,
                )
                raw = _consume(child, item["id"], row_deadline, progress)
                write_new(directory / "reply.json", raw)
                progress["reply"] = ref(root, directory / "reply.json").model_dump()
                progress.update(
                    _accept_reply(root, directory, mode, item["payload"], raw, model)
                )
                progress["elapsed_seconds"] = dependencies.clock() - row_started
                committed = dict(progress, disposition="completed")
                write_new(directory / "outcome.json", canonical_json(committed))
                progress.update(committed)
            for source in sources:
                read_ref(root, source)
        except BaseException as error:
            fail(error)
            try:
                write_new(
                    destination / "original-error.txt", traceback.format_exc().encode()
                )
            except BaseException as persistence_error:
                fail(persistence_error)
        finally:
            if child is not None:
                try:
                    cleanup = child.close()
                    cleanup_verified = cleanup.verified
                    if not cleanup_verified or child.failure:
                        fail(
                            RuntimeError(
                                f"preparation cleanup/sensor failure: {cleanup.model_dump()}; {child.failure}"
                            )
                        )
                except BaseException as error:
                    cleanup_verified = False
                    fail(error)
            elapsed = dependencies.clock() - started
            if reservation is not None:
                try:
                    ledger.settle(
                        reservation,
                        "failed" if errors else "finished",
                        math.ceil(elapsed),
                    )
                except BaseException as error:
                    fail(error)
            result = dict(
                schema_version="nlp3-preparation-status-v1",
                mode=mode,
                status="failed" if errors else "completed",
                errors=errors,
                outcomes=outcomes,
                cleanup_verified=cleanup_verified,
                elapsed_seconds=elapsed,
                qwen_generations=0,
                reservation_id=reservation,
                storage_reservation_id=storage,
                timing_interpretation="supervised preparation, not live request latency",
            )
            try:
                atomic_json(destination / "status.json", result)
                if storage is not None:
                    output_bytes = sum(
                        p.stat().st_size for p in destination.rglob("*") if p.is_file()
                    )
                    registry_bytes = (
                        (root / REGISTRY).stat().st_size
                        if (root / REGISTRY).exists()
                        else 0
                    )
                    ledger.settle_storage(
                        storage,
                        "failed" if errors else "finished",
                        output_bytes=output_bytes
                        + max(0, registry_bytes - registry_start_bytes),
                        ledger_start_bytes=ledger_start_bytes,
                    )
            except BaseException as error:
                fail(error)
                result["status"] = "failed"
                result["durability"] = "incomplete"
                try:
                    atomic_json(destination / "status.json", result)
                except BaseException as persistence_error:
                    fail(persistence_error)
    finally:
        # Evidence/accounting faults must never bypass owned-process disposition.
        try:
            mutex.release(cleanup_verified=cleanup_verified)
        except BaseException as error:
            fail(error)
            result["status"] = "failed"
            result["durability"] = "incomplete_mutex_disposition"
            try:
                atomic_json(destination / "status.json", result)
            except BaseException as persistence_error:
                fail(persistence_error)
    return result


def materialize(
    root,
    destination,
    encoder_status,
    *,
    public_root,
    owner_id,
    binding: TimezoneBinding,
):
    """Prepare baseline once and one shared assisted artifact, with observed cost."""
    if (
        encoder_status["status"] != "completed"
        or not encoder_status["cleanup_verified"]
    ):
        raise ValueError("stopped/unresolved encoder job cannot become a ready package")
    root, destination = _destination(root, destination)
    if destination not in _ACCOUNTED_DESTINATIONS:
        raise ValueError(
            "materialization requires its active preparation accounting context"
        )
    destination.mkdir(parents=True, exist_ok=False)
    record_budget = loads_json((root / "configs/study-policy.json").read_text())[
        "preparation"
    ]["max_records"]
    if type(record_budget) is not int or not 1 <= record_budget <= 40:
        raise ValueError(
            "study packing budget must be a positive whole-record bound <=40"
        )
    dataset = load_public_dataset(root / public_root, owner_id=owner_id)
    outcomes = {r["payload"]["message_id"]: r for r in encoder_status["outcomes"]}
    if set(outcomes) != {r.message_id for r in dataset.requests}:
        raise ValueError("encoder dispositions differ from complete public roster")
    rows, token_items = [], []
    for index, request in enumerate(dataset.requests):
        extraction_ref = outcomes[request.message_id]["artifact"]
        extraction_path = extraction_ref["path"]
        # Authenticate actual saved bytes, not merely the caller's path.
        from messaging_nlp.run_contracts import ArtifactRef

        read_ref(root, ArtifactRef.model_validate(extraction_ref))
        case = destination / "cases" / f"{index:06d}"
        times = {}
        started = time.monotonic()
        inputs = prepare_inputs(
            dataset,
            request.message_id,
            root=root,
            baseline_path=(case / "baseline.json").relative_to(root).as_posix(),
            assisted_path=(case / "shared.json").relative_to(root).as_posix(),
            extraction_path=extraction_path,
            identity=retained_identity(),
            binding=binding,
            timings=times,
            max_records=record_budget,
        )
        times["total_materialization_seconds"] = time.monotonic() - started
        record = dict(
            message_id=request.message_id,
            index=index,
            extraction=extraction_ref,
            baseline=ref(root, root / inputs.baseline.path).model_dump(),
            shared=ref(root, root / inputs.assisted.path).model_dump()
            if inputs.assisted
            else None,
            assisted_failure=inputs.assisted_failure.model_dump()
            if inputs.assisted_failure
            else None,
            timings=times,
            prompts={},
        )
        for condition in ("baseline", "B", "A+B"):
            sealed = inputs.baseline if condition == "baseline" else inputs.assisted
            if sealed is None:
                continue
            packet = PreparationPacket.model_validate_json(sealed.data)
            prompt = build_prompt(packet, condition)
            messages = [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user.decode()},
            ]
            name = {"baseline": "baseline", "B": "b", "A+B": "ab"}[condition]
            path = case / f"{name}-messages.json"
            write_new(path, canonical_json(messages))
            item_id = f"case-{index:06d}-{name}"
            record["prompts"][condition] = dict(
                id=item_id, prompt=ref(root, path).model_dump()
            )
            token_items.append(
                dict(
                    id=item_id,
                    payload=dict(
                        messages=messages,
                        messages_sha256=digest(canonical_json(messages)),
                    ),
                )
            )
        rows.append(record)
    manifest = dict(
        schema_version="nlp3-materialization-v1",
        record_budget=record_budget,
        rows=rows,
        token_items=token_items,
        qwen_generation=False,
        search_cost="baseline/shared measured separately, no live E2E inference",
    )
    write_new(destination / "manifest.json", canonical_json(manifest))
    return manifest


@contextmanager
def accounted_preparation(
    root,
    destination,
    *,
    phase,
    max_seconds=600,
    readiness_bytes=0,
    revision_authority=None,
    storage_bytes=268435456,
):
    """Charge model-free materialization/freeze once, not as live request latency.

    Only the fixed-size readiness control record may be charged immediately before
    creation: its two hashes have constant64-character widths. Freeze must verify
    its actual byte length after writing; this breaks the readiness/ledger cycle.
    """
    root, destination = _destination(root, destination)
    if type(readiness_bytes) is not int or not 0 <= readiness_bytes <= 4096:
        raise ValueError("only bounded fixed-size readiness metadata is anticipated")
    dependencies = _Dependencies()
    (root / "runs/study-state").mkdir(parents=True, exist_ok=True)
    mutex = ModelMutex(root / MUTEX_PATH, "stage5-software-" + phase, boot_identity())
    ledger_start = (
        (root / LEDGER_PATH).stat().st_size if (root / LEDGER_PATH).exists() else 0
    )
    started = time.monotonic()
    errors, original = [], None
    ledger = reservation = storage = None
    mutex.acquire()
    try:
        ledger = _ledger(root, dependencies, revision_authority)
        refs = dict(
            run_id="stage5-" + phase + "-" + digest(str(destination).encode())[:16],
            phase=phase,
        )
        reservation = ledger.reserve("preparation", max_seconds, refs=refs)
        storage = ledger.reserve("storage", 0, storage_bytes=storage_bytes, refs=refs)
        try:
            baseline = MacSensors(root).sample(0)
            if baseline.pressure_level != 1 or check_sample(
                baseline, baseline, ResourceLimits()
            ):
                raise RuntimeError(
                    "software preparation resource baseline is not ready"
                )
            _ACCOUNTED_DESTINATIONS.add(destination)
            yield ledger
            sample = MacSensors(root).sample(0)
            if check_sample(sample, baseline, ResourceLimits()):
                raise RuntimeError("software preparation resource limit")
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("software preparation exceeded reserved wall time")
        except BaseException as error:
            original = error
            errors.append(f"{type(error).__name__}: {error}")
            raise
        finally:
            _ACCOUNTED_DESTINATIONS.discard(destination)
            elapsed = time.monotonic() - started
            try:
                ledger.settle(
                    reservation, "failed" if errors else "finished", math.ceil(elapsed)
                )
            except BaseException as error:
                errors.append(f"{type(error).__name__}: {error}")
            try:
                write_new(
                    destination / "accounting.json",
                    canonical_json(
                        dict(
                            phase=phase,
                            elapsed_seconds=elapsed,
                            errors=errors,
                            reservation_id=reservation,
                            storage_reservation_id=storage,
                            fixed_readiness_bytes=readiness_bytes,
                            qwen_generations=0,
                        )
                    ),
                )
                output_bytes = sum(
                    p.stat().st_size for p in destination.rglob("*") if p.is_file()
                )
                ledger.settle_storage(
                    storage,
                    "failed" if errors else "finished",
                    output_bytes=output_bytes + readiness_bytes,
                    ledger_start_bytes=ledger_start,
                )
            except BaseException as error:
                errors.append(f"{type(error).__name__}: {error}")
    finally:
        try:
            mutex.release(cleanup_verified=True)  # no child/model is launched here
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
        if errors and original is None:
            raise RuntimeError(
                "software preparation/accounting failed: " + "; ".join(errors)
            )


def retained_qwen_spec(root):
    """Metadata only; actual supervised tokenizer loader verifies every asset."""
    root = Path(root)
    receipt = ref(root, root / "models/receipts/qwen.json")
    if receipt.sha256 != QWEN_RECEIPT_SHA256:
        raise ValueError("untrusted retained Qwen identity")
    body = loads_json((root / receipt.path).read_text())
    template = next(
        row["sha256"] for row in body["files"] if row["file"] == "chat_template.jinja"
    )
    return ModelSpec(path=body["directory"], receipt=receipt, template_sha256=template)


def freeze(
    root,
    destination,
    *,
    materialized,
    tokenized,
    phase_directories,
    model,
    run_id,
    output_path,
    readiness_path,
    approval_path,
    revision_authority=None,
    run_storage_bytes=1073741824,
    _model_checker=None,
    _fixture=None,
):
    """Seal all planned rows and portable bytes; never creates execution approval.

    No real scoring is called. The readiness receipt is written only after all
    preparation charges settle. Its known fixed byte length is charged before
    the final ledger hash exists, then checked against the actual durable bytes.
    """
    root, destination = _destination(root, destination)
    if _fixture is not None and _model_checker is None:
        raise ValueError("fictional freeze seam requires a private model checker")
    fixture = _fixture or {}
    owner_id = fixture.get("owner_id", "owner_001")
    timezone_name = fixture.get("timezone_name", "America/Vancouver")
    policy = loads_json((root / "configs/study-policy.json").read_text())
    if materialized.get("record_budget") != policy["preparation"]["max_records"]:
        raise ValueError(
            "materialization record budget differs from frozen study policy"
        )
    if tokenized["status"] != "completed" or not tokenized["cleanup_verified"]:
        raise ValueError("incomplete tokenizer supervision cannot be restart-ready")
    tokens = {row["id"]: row for row in tokenized["outcomes"]}
    expected_ids = {item["id"] for item in materialized["token_items"]}
    if set(tokens) != expected_ids or any(
        row.get("result_status") != "fits" or row["disposition"] != "completed"
        for row in tokens.values()
    ):
        raise ValueError("unresolved prompt fit or tokenizer artifact mismatch")
    for path in (output_path, readiness_path, approval_path):
        if (root / path).exists():
            raise FileExistsError("run/readiness/approval destination already exists")
    boot = boot_identity()
    readiness_size = len(
        canonical_json(
            ReadinessReceipt(
                plan_sha256="0" * 64, boot_identity=boot, ledger_sha256="0" * 64
            ).model_dump()
        )
    )
    with accounted_preparation(
        root,
        destination,
        phase="freeze",
        readiness_bytes=readiness_size,
        revision_authority=revision_authority,
        storage_bytes=536870912 if revision_authority else 268435456,
    ) as ledger:
        destination.mkdir(parents=True, exist_ok=False)
        roster_path = policy["data"]["roster"]
        roster = [
            loads_json(line) for line in (root / roster_path).read_text().splitlines()
        ]
        prepared = {row["message_id"]: row for row in materialized["rows"]}
        if set(prepared) != {row["message_id"] for row in roster}:
            raise ValueError("materialization differs from frozen roster")
        dataset = load_public_dataset(
            root / policy["data"]["public_root"], owner_id=owner_id
        )
        requests = {request.message_id: request for request in dataset.requests}
        spending = SpendingPolicy(
            startup_seconds=120,
            overhead_seconds=180,
            max_run_seconds=min(21600, ledger.snapshot()["remaining_seconds"]),
            storage_bytes=run_storage_bytes,
            assistant_version="standalone-common-"
            + digest(
                canonical_json(
                    dict(
                        policy=policy_identity(),
                        template=model.template_sha256,
                        decoding=DecodePolicy().model_dump(),
                    )
                )
            )[:16],
            assistant_version_refs=dict(
                prompt_policy_sha256=policy_identity(),
                template_sha256=model.template_sha256,
                decoding_sha256=digest(canonical_json(DecodePolicy().model_dump())),
            ),
            extraction_version="retained-xlmr-" + retained_identity().digest()[:16],
            extraction_version_refs={"extractor_sha256": retained_identity().digest()},
        )
        if (
            spending.max_run_seconds < 420
            or spending.storage_bytes > ledger.snapshot()["remaining_storage_bytes"]
        ):
            raise ValueError("insufficient bounded run allowance")
        ledger.register_versions(spending)
        planned = []
        for member in roster:
            case = prepared[member["message_id"]]
            for condition in member["condition_order"]:
                ready = condition == "baseline" or case["shared"] is not None
                prompt = case["prompts"].get(condition)
                token = tokens[prompt["id"]] if ready else None
                planned.append(
                    PlannedRow(
                        row_id=f"row-{len(planned):06d}",
                        ordinal=len(planned),
                        condition=condition,
                        request=requests[member["message_id"]],
                        preparation=ArtifactRef.model_validate(
                            case["baseline"]
                            if condition == "baseline"
                            else case["shared"]
                        )
                        if ready
                        else None,
                        preparation_failure=None if ready else case["assisted_failure"],
                        prompt=ArtifactRef.model_validate(prompt["prompt"])
                        if ready
                        else None,
                        token_receipt=ArtifactRef.model_validate(token["token_receipt"])
                        if ready
                        else None,
                    )
                )
        if ledger.snapshot()["remaining_generations"] < sum(
            r.preparation_failure is None for r in planned
        ):
            raise ValueError(
                "insufficient generation allowance for runnable planned rows"
            )
        paths = {
            p.relative_to(root).as_posix()
            for p in (root / "src/messaging_nlp").rglob("*.py")
        }
        paths.update(
            [
                "pyproject.toml",
                "uv.lock",
                "environments/encoder/pyproject.toml",
                "environments/encoder/uv.lock",
                "environments/qwen/pyproject.toml",
                "environments/qwen/uv.lock",
                "configs/study-policy.json",
                "configs/evaluation-policy.json",
                "docs/stage4b-evaluation-contract.md",
                "docs/stage1-budget-opening.json",
                "docs/budget-usage-reset-20261027.json",
                REGISTRY,
            ]
        )
        paths.update(
            item.path
            for item in preparation_sources(root, policy["data"]["public_root"])
        )
        paths.update(
            p.relative_to(root).as_posix()
            for p in (root / fixture.get("data_root", "data/stage1-v1")).rglob("*")
            if p.is_file()
        )
        paths.update(
            [
                model.receipt.path,
                f"{model.path}/chat_template.jinja",
                roster_path,
                policy["temporal"]["timezone_identity"]["tzif_path"],
            ]
        )
        if _fixture is None:
            paths.update(
                [
                    "models/receipts/xlmr.json",
                    "models/xlmr-final-block-head/block-head-config.json",
                    "models/xlm-roberta-base/config.json",
                    "models/xlm-roberta-base/tokenizer_config.json",
                ]
            )
        for phase in phase_directories:
            phase = Path(phase).resolve(strict=True)
            if not phase.is_relative_to(root) or destination.is_relative_to(phase):
                raise ValueError(
                    "phase inventory must be disjoint retained project inputs"
                )
            paths.update(
                p.relative_to(root).as_posix() for p in phase.rglob("*") if p.is_file()
            )
        if revision_authority is not None:
            from messaging_nlp.run_revision import load_authority

            revision = load_authority(
                root / revision_authority.path, revision_authority.sha256
            )
            paths.add(revision_authority.path)
            paths.update(
                revision["body"][name]["path"]
                for name in (
                    "predecessor_budget",
                    "predecessor_plan",
                    "predecessor_status",
                    "protected_evidence",
                )
            )
        for name in (
            ()
            if _fixture
            else (
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "generation_config.json",
                "kv_config.json",
                "model.safetensors.index.json",
                "optiq_metadata.json",
            )
        ):
            paths.add(f"{model.path}/{name}")
        frozen = []
        for name in sorted(paths):
            path = root / name
            if path.is_symlink():
                raise ValueError("frozen input must not be a symlink")
            role = (
                "source"
                if name.startswith("src/messaging_nlp/")
                else "extraction"
                if name.endswith("/extraction.json")
                else "lock"
                if name.endswith("uv.lock")
                else "other"
            )
            frozen.append(FrozenFile(**ref(root, path).model_dump(), role=role))
        zone = policy["temporal"]["timezone_identity"]
        plan = RunPlan(
            project_root=os.path.relpath(root, destination),
            run_id=run_id,
            output_path=output_path,
            readiness_path=readiness_path,
            approval_path=approval_path,
            opening=ref(root, root / "docs/stage1-budget-opening.json"),
            usage_reset=ref(root, root / "docs/budget-usage-reset-20261027.json"),
            public_root=policy["data"]["public_root"],
            owner_id=owner_id,
            timezone_name=timezone_name,
            timezone=ref(root, root / zone["tzif_path"]),
            roster=ref(root, root / roster_path),
            frozen_files=frozen,
            model=model,
            spending=spending,
            rows=planned,
            resources=ResourceLimits(memory_warning_seconds=10)
            if revision_authority
            else ResourceLimits(),
            revision_authority=revision_authority,
        )
        write_new(destination / "plan.json", canonical_json(plan.model_dump()))
        # Actual source/config/data bytes, not just a commit or references.
        portable = []
        for entry in frozen:
            path = destination / "portable/project" / entry.path
            write_new(path, read_ref(root, entry))
            portable.append(ref(destination, path).model_dump())
        write_new(
            destination / "portable-manifest.json",
            canonical_json(
                dict(
                    files=portable,
                    purpose="frozen source/config/data and inputs; weights remain local",
                    evaluator_boundary="evaluator/gold/requirements are private offline-scoring bytes; never inference preparation or prompts",
                    readiness_location=readiness_path,
                    ledger_location=LEDGER_PATH,
                )
            ),
        )
        validated = load_and_validate(
            destination / "plan.json", model_checker=_model_checker
        )
        if len(validated.plan.rows) != len(roster) * 3:
            raise ValueError("missing planned conditions")
        from zoneinfo import ZoneInfo

        zones = sorted({request.time_zone for request in dataset.requests})
        for name in zones:
            ZoneInfo(name)  # name catalog only; frozen offsets use retained TZif
        write_new(
            destination / "preflight.json",
            canonical_json(
                dict(
                    iana_name_catalog_recognized=zones,
                    boot_identity=boot,
                    planned_rows=len(planned),
                    blocked_rows=sum(
                        r.preparation_failure is not None for r in planned
                    ),
                    input_token_max=max(row["token_count"] for row in tokens.values()),
                    budget=ledger.snapshot(),
                    run_bound="attempt-to-complete, not a completion guarantee",
                    qwen_weight_loading=False,
                    qwen_generation=False,
                )
            ),
        )
    ready = ReadinessReceipt(
        plan_sha256=digest((destination / "plan.json").read_bytes()),
        boot_identity=boot,
        ledger_sha256=digest((root / LEDGER_PATH).read_bytes()),
    )
    ready_bytes = canonical_json(ready.model_dump())
    if len(ready_bytes) != readiness_size:
        raise ValueError("fixed-size readiness accounting mismatch")
    write_new(root / readiness_path, ready_bytes)
    if (root / readiness_path).stat().st_size != readiness_size:
        raise ValueError("readiness persisted byte count mismatch")
    return dict(
        plan=ref(root, destination / "plan.json").model_dump(),
        readiness=ref(root, root / readiness_path).model_dump(),
        ledger=ref(root, root / LEDGER_PATH).model_dump(),
        command=f"uv run --locked --offline messaging-nlp infer --plan {destination.relative_to(root).as_posix()}/plan.json --approve-run",
        qwen_generation=False,
    )
