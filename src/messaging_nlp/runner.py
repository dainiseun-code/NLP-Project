"""Serial, durable Stage 4A coordinator; optional model code stays in the child.

The public entry point has no alternate worker, authorization bypass or retry.
Private dependencies are deliberately limited to model-free unit-test seams.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from messaging_nlp.application import execute_condition
from messaging_nlp.application_types import (
    ApplicationResult,
    GenerationOutput,
    PreparationPacket,
    Prompt,
)
from messaging_nlp.contracts import RunRow, TechnicalFailure
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.limits import CALL_TIMEOUT_SECONDS
from messaging_nlp.preparation import artifact_path, digest
from messaging_nlp.run_budget import BudgetLedger
from messaging_nlp.run_contracts import (
    CleanupReceipt,
    PendingRowRecord,
    RowReceipt,
    RunPlan,
    WorkerEvent,
    WorkerRequest,
)
from messaging_nlp.run_plan import (
    approve_execution,
    authorization,
    create_output,
    load_and_validate,
)
from messaging_nlp.run_storage import append_bytes, atomic_json, ref, write_new
from messaging_nlp.run_supervision import (
    ChildSession,
    MacSensors,
    ModelMutex,
    SupervisionError,
    boot_identity,
)


def _check_model(root, model):
    # verify_model is static/local-byte validation, not QwenRuntime construction.
    from messaging_nlp.qwen_worker import verify_model

    return verify_model(root, model)


@dataclass(frozen=True)
class Dependencies:
    boot: Callable = boot_identity
    sensors_factory: Callable = MacSensors
    child_factory: Callable = ChildSession
    model_checker: Callable = _check_model
    clock: Callable = time.monotonic


@dataclass
class _Observation:
    calls: int = 0
    intent: bool = False
    dispatched: bool = False
    started: bool = False
    completed: bool = False
    finish: str | None = None
    reservation: str | None = None
    elapsed: float = 0.0
    wall_elapsed: float = 0.0
    wall_accounted: bool = False
    application_started: float | None = None
    preprocessing_seconds: float = 0.0
    settled: bool = False
    generation_tokens: int | None = None
    input_tokens: int | None = None
    raw: str = ""
    halt: BaseException | None = None
    events: list = field(default_factory=list)
    failures: list = field(default_factory=list)


@dataclass
class _RowProgress:
    row: object
    seen: _Observation = field(default_factory=_Observation)
    result: ApplicationResult | None = None
    persist_attempted: bool = False
    errors: list[str] = field(default_factory=list)


def _pending_record(output, progress, reason):
    """Index provable partial bytes, never repair/re-persist an incomplete row."""
    row, seen, result = progress.row, progress.seen, progress.result
    destination = _row_dir(output, row)
    artifacts, unavailable = {}, ["receipt.json"]
    errors = list(progress.errors)
    expected = {"run-row.json"}
    if result is not None and result.envelope is not None:
        expected.add("resolved-envelope.json")
    raw = result.row.raw_output if result is not None else None
    if raw is None and seen.calls:
        raw = seen.raw
    if raw is not None:
        expected.add("raw-output.txt")
    try:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and not path.is_symlink():
                name = path.relative_to(destination).as_posix()
                try:
                    artifacts[name] = ref(output, path)
                except BaseException as error:
                    unavailable.append(name)
                    errors.append(f"index {name}: {type(error).__name__}: {error}")
    except BaseException as error:
        errors.append(f"index row artifacts: {type(error).__name__}: {error}")
    # Even existing receipt bytes are not a committed receipt if its write/fsync
    # did not return. A reference here proves bytes, not their durable commit.
    unavailable.extend(sorted(expected - artifacts.keys()))
    return PendingRowRecord(
        row_id=row.row_id,
        ordinal=row.ordinal,
        condition=row.condition,
        message_id=row.request.message_id,
        provider_calls=seen.calls,
        dispatch_intent=seen.intent,
        dispatched=seen.dispatched,
        generation_started=seen.started,
        generation_completed=seen.completed,
        finish_reason=seen.finish,
        reservation_id=seen.reservation,
        raw_output=raw,
        run_row_bytes_utf8=(
            canonical_json(result.row.model_dump()).decode("utf-8")
            if result is not None
            else None
        ),
        stop_reason=reason or "row receipt could not be committed",
        persistence_errors=errors,
        artifacts=artifacts,
        unavailable_artifacts=unavailable,
    )


def _failure(error, *, stage="runner"):
    code = getattr(error, "code", "runtime_error")
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        code = "interrupted"
    if code not in {
        "timeout",
        "invalid_output",
        "missing",
        "runtime_error",
        "truncated",
        "launch_error",
        "sensor_error",
        "interrupted",
        "resource_stop",
    }:
        code = "runtime_error"
    return TechnicalFailure(
        stage=stage, code=code, message=f"{type(error).__name__}: {error}"
    )


def _write_json(path, value):
    write_new(path, canonical_json(value))


def _durable_raw(journal, row_id, prefix):
    try:
        lines = journal.read_bytes().splitlines()
    except OSError:
        return prefix
    pieces = []
    for line in lines:
        try:
            event = loads_json(line.decode("utf-8"))
            if event.get("row_id") == row_id and isinstance(
                event.get("payload", {}).get("text"), str
            ):
                pieces.append(event["payload"]["text"])
        except (ValueError, AttributeError):
            continue  # Exact malformed bytes remain in the owned log.
    captured = "".join(pieces)
    return captured if captured.startswith(prefix) else prefix


def _row_dir(output, row):
    # Untrusted row IDs never become filesystem path components.
    return output / "rows" / f"{row.ordinal:06d}"


def _unexecuted(validated, row, reason):
    if validated.dataset is not None:
        context = validated.dataset.bundle(row.request.message_id).context.model_dump()
    else:
        # A failed public-data gate may make reconstruction impossible. Bind an
        # explicit unavailable-context marker, not an invented successful context.
        context = {
            "context_unavailable": True,
            "snapshot_id": row.request.context_snapshot_id,
        }
    return ApplicationResult(
        RunRow(
            schema_version=1,
            run_id=validated.plan.run_id,
            condition=row.condition,
            request=row.request,
            request_sha256=digest(canonical_json(row.request.model_dump())),
            context_sha256=digest(canonical_json(context)),
            preprocessing_sha256=row.preparation.sha256 if row.preparation else None,
            extraction_result=None,
            candidates=None,
            status="unexecuted",
            decision=None,
            failure=row.preparation_failure or _failure(RuntimeError(reason)),
            raw_output=None,
            attempt=1,
        ),
        0,
        None,
        None,
    )


def _retain_inputs(validated, output):
    bindings = {}
    for relative, data in validated.frozen_bytes.items():
        target = artifact_path(output, f"frozen/{relative}")
        # create_output has already copied/fsynced these exact bytes.
        if target.read_bytes() != data:
            raise ValueError(f"retained frozen copy differs: {relative}")
        bindings[relative] = ref(output, target)
    return bindings


def _snapshot_logs(output, destination, offsets):
    logs = {}
    ranges = {}
    worker = output / "worker"
    if worker.exists():
        for source in sorted(worker.iterdir()):
            if source.is_file() and not source.is_symlink():
                target = destination / "logs" / source.name
                if source.name == "supervision.json":
                    write_new(target, source.read_bytes())
                else:
                    start, end = offsets.get(source.name, 0), source.stat().st_size
                    if end < start:
                        raise RuntimeError("worker append-only log was truncated")
                    with source.open("rb") as handle:
                        handle.seek(start)
                        data = handle.read(end - start)
                    if len(data) != end - start:
                        raise RuntimeError(
                            "worker append-only log changed during snapshot"
                        )
                    write_new(target, data)
                    offsets[source.name] = end
                    ranges[source.name] = {
                        "source": source.relative_to(output).as_posix(),
                        "start": start,
                        "end": end,
                    }
                logs[source.name] = ref(output, target)
    if ranges:
        path = destination / "log-ranges.json"
        _write_json(path, ranges)
        logs["log_ranges"] = ref(output, path)
    return logs


def _persist_row(
    validated,
    output,
    row,
    result,
    observation,
    *,
    ledger,
    startup,
    reason,
    input_bindings,
    log_offsets,
    cleanup=None,
):
    destination = _row_dir(output, row)
    run_row_path = destination / "run-row.json"
    _write_json(run_row_path, result.row.model_dump())
    if observation.events:
        _write_json(destination / "observed-events.json", observation.events)
    envelope = None
    if result.envelope is not None:
        path = destination / "resolved-envelope.json"
        _write_json(path, result.envelope.model_dump())
        envelope = ref(output, path)
    raw = None
    raw_text = result.row.raw_output
    if raw_text is not None:
        path = destination / "raw-output.txt"
        write_new(path, raw_text.encode("utf-8"))
        raw = ref(output, path)
    bindings = {f"frozen:{key}": value for key, value in input_bindings.items()}
    bindings["plan"] = ref(output, output / "plan.json")
    if validated.dataset is None:
        marker = {
            "context_unavailable": True,
            "snapshot_id": row.request.context_snapshot_id,
        }
        marker_path = destination / "context-unavailable.json"
        _write_json(marker_path, marker)
        bindings["context_unavailable"] = ref(output, marker_path)
    for name in ("preparation", "prompt", "token_receipt"):
        original = getattr(row, name)
        if original is not None:
            if original.path not in input_bindings:
                # Missing bytes can themselves be the gate failure. Retain the
                # expected binding without pretending it is an observed artifact.
                target = destination / f"expected-{name}.json"
                _write_json(target, original.model_dump())
                bindings[f"expected_{name}"] = ref(output, target)
            else:
                bindings[name] = input_bindings[original.path]
    if result.prompt is not None:
        prepared = validated.prepared[row.request.message_id]
        sealed = prepared.baseline if row.condition == "baseline" else prepared.assisted
        packet = PreparationPacket.model_validate_json(sealed.data)
        aliases_path = destination / "aliases.json"
        _write_json(
            aliases_path,
            {key: value.model_dump() for key, value in packet.aliases.items()},
        )
        bindings["aliases"] = ref(output, aliases_path)
        if packet.extraction_artifact_path in input_bindings:
            bindings["extraction"] = input_bindings[packet.extraction_artifact_path]
    for name, filename in {
        "messages": "messages.json",
        "system": "system.txt",
        "common": "common.txt",
        "entities": "entities.txt",
        "user": "user.txt",
        "timing": "timing.json",
        "events": "events.jsonl",
        "partial_raw": "partial-raw.txt",
        "worker_request": "worker-request.json",
        "observed_events": "observed-events.json",
    }.items():
        path = destination / filename
        if path.exists():
            bindings[name] = ref(output, path)
    seconds, generations = 0, 0
    if observation.reservation is not None:
        seconds, generations, _ = ledger.charge(observation.reservation)
    if result.row.status == "ok":
        disposition = "completed"
    elif observation.calls:
        disposition = "attempted_failure"
    elif (
        row.preparation_failure is not None
        or result.row.failure.stage == "preprocessing"
    ):
        disposition = "preparation_blocked"
    else:
        disposition = "unattempted_stop"
    receipt = RowReceipt(
        run_id=validated.plan.run_id,
        plan_sha256=validated.plan_sha256,
        row_id=row.row_id,
        ordinal=row.ordinal,
        condition=row.condition,
        message_id=row.request.message_id,
        disposition=disposition,
        provider_calls=result.provider_calls,
        dispatch_intent=observation.intent,
        dispatched=observation.dispatched,
        generation_started=observation.started,
        generation_completed=observation.completed,
        finish_reason=observation.finish,
        stop_reason=reason,
        reservation_id=observation.reservation,
        startup_reservation_id=startup,
        charged_seconds=seconds,
        charged_generations=generations,
        elapsed_seconds=observation.elapsed,
        generation_tokens=observation.generation_tokens,
        input_tokens=observation.input_tokens,
        run_row=ref(output, run_row_path),
        resolved_envelope=envelope,
        raw_output=raw,
        bindings=bindings,
        logs=_snapshot_logs(output, destination, log_offsets),
        cleanup=cleanup,
    )
    # This exclusive fsynced write is the row's commit point. It must return before
    # a subsequent reservation, not merely before the subsequent send.
    _write_json(destination / "receipt.json", receipt.model_dump())
    return receipt


class _Provider:
    def __init__(
        self,
        validated,
        row,
        output,
        child,
        ledger,
        dependencies,
        deadline,
        observation,
        reporter=None,
    ):
        self.validated, self.row, self.output = validated, row, output
        self.child, self.ledger, self.dependencies = child, ledger, dependencies
        self.deadline, self.observation = deadline, observation
        self.reporter = reporter

    def generate(self, prompt: Prompt, *, max_input_tokens, max_output_tokens):
        seen = self.observation
        if seen.calls:
            raise RuntimeError("a row may invoke its provider only once")
        seen.calls = 1
        started = self.dependencies.clock()
        if seen.application_started is not None:
            seen.preprocessing_seconds = max(0.0, started - seen.application_started)
        directory = _row_dir(self.output, self.row)
        last_progress_count = 0
        try:
            policy = self.validated.plan.decoding
            if (max_input_tokens, max_output_tokens) != (
                policy.max_input_tokens,
                policy.max_output_tokens,
            ):
                raise ValueError("unexpected generation ceilings")
            if prompt != self.validated.prompts[self.row.row_id]:
                raise ValueError("rendered prompt differs from frozen approved prompt")
            messages = [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user.decode("utf-8")},
            ]
            token = self.validated.token_receipts[self.row.row_id]
            if digest(canonical_json(messages)) != token.messages_sha256:
                raise ValueError("messages differ from sealed token receipt")
            _write_json(directory / "messages.json", messages)
            write_new(directory / "system.txt", prompt.system.encode("utf-8"))
            write_new(directory / "common.txt", prompt.common)
            write_new(directory / "user.txt", prompt.user)
            if prompt.entities is not None:
                write_new(directory / "entities.txt", prompt.entities)
            if self.dependencies.clock() + CALL_TIMEOUT_SECONDS > self.deadline:
                raise SupervisionError(
                    "timeout", "full call runway exhausted during preparation"
                )
            refs = {
                "run_id": self.validated.plan.run_id,
                "plan_sha256": self.validated.plan_sha256,
                "row_id": self.row.row_id,
                "attempt_key": digest(
                    canonical_json(
                        {
                            "request": self.row.request.model_dump(),
                            "condition": self.row.condition,
                        }
                    )
                ),
            }
            if self.validated.plan.revision_authority is not None:
                from messaging_nlp.run_revision import EXPERIMENT_ID

                refs.update(
                    experiment_id=EXPERIMENT_ID,
                    revision_authority_sha256=self.validated.plan.revision_authority.sha256,
                )
            seen.reservation = self.ledger.reserve(
                "row", CALL_TIMEOUT_SECONDS, generations=1, refs=refs
            )
            _write_json(
                directory / "dispatch-intent.json",
                {
                    **refs,
                    "reservation_id": seen.reservation,
                    "attempt": 1,
                    "monotonic_seconds": self.dependencies.clock(),
                    "messages_sha256": token.messages_sha256,
                    "token_receipt": self.row.token_receipt.model_dump(),
                },
            )
            seen.intent = True
            request = WorkerRequest(
                kind="generate",
                row_id=self.row.row_id,
                payload={"messages": messages, "token_receipt": token.model_dump()},
            )
            _write_json(directory / "worker-request.json", request.model_dump())
            call_started = self.dependencies.clock()
            if call_started + CALL_TIMEOUT_SECONDS > self.deadline:
                # Proof is local and explicit: send has never been called.
                self.ledger.settle(
                    seen.reservation, "cancelled", 0, never_launched=True
                )
                seen.settled = True
                raise SupervisionError(
                    "timeout", "full call runway exhausted before dispatch"
                )
            call_deadline = call_started + CALL_TIMEOUT_SECONDS
            self.child.send(request, deadline=call_deadline)
            seen.dispatched = True
            while True:
                if self.dependencies.clock() >= call_deadline:
                    raise SupervisionError("timeout", "generation wall deadline")
                event = self.child.wait_event(call_deadline)
                if (
                    not isinstance(event, WorkerEvent)
                    or event.row_id != self.row.row_id
                ):
                    raise RuntimeError("worker event does not belong to dispatched row")
                seen.events.append(event.model_dump())
                if event.kind == "generation_started":
                    if seen.started:
                        raise RuntimeError("duplicate generation-start acknowledgement")
                    seen.started = True
                    _report(self.reporter, phase="prefill")
                elif event.kind in ("fragment", "progress"):
                    if not seen.started:
                        raise RuntimeError(
                            "progress without model-start acknowledgement"
                        )
                    count = event.payload.get("generation_tokens")
                    if count is not None:
                        if (
                            type(count) is not int
                            or not 1 <= count <= max_output_tokens
                            or count < last_progress_count
                        ):
                            raise RuntimeError("invalid observed progress token count")
                        if count > last_progress_count:
                            _report(
                                self.reporter,
                                phase="generating",
                                generation_tokens=count,
                                last_token_monotonic=time.monotonic(),
                            )
                            last_progress_count = count
                    if event.kind == "progress":
                        if not seen.started or count is None:
                            raise RuntimeError(
                                "progress requires model-start and token count"
                            )
                    else:
                        text = event.payload.get("text")
                        if not seen.started or not isinstance(text, str):
                            raise RuntimeError(
                                "fragment without start acknowledgement/text"
                            )
                        seen.raw += text
                elif event.kind == "terminal":
                    if not seen.started:
                        raise RuntimeError(
                            "terminal without generation-start acknowledgement"
                        )
                    finish = event.payload.get("finish_reason")
                    duration = event.payload.get("elapsed_seconds")
                    if (
                        finish not in {"stop", "length"}
                        or type(duration) not in (int, float)
                        or not math.isfinite(duration)
                        or duration < 0
                    ):
                        raise RuntimeError("invalid terminal metadata")
                    for key, ceiling in (
                        ("generation_tokens", max_output_tokens),
                        ("input_tokens", max_input_tokens),
                    ):
                        count = event.payload.get(key)
                        if type(count) is not int or not 1 <= count <= ceiling:
                            raise RuntimeError(f"invalid terminal {key}")
                    if event.payload["input_tokens"] != token.token_count:
                        raise RuntimeError(
                            "runtime token count differs from sealed receipt"
                        )
                    if event.payload["generation_tokens"] < last_progress_count:
                        raise RuntimeError("terminal token count regressed")
                    _report(
                        self.reporter,
                        generation_tokens=event.payload["generation_tokens"],
                    )
                    seen.finish, seen.completed = finish, True
                    seen.generation_tokens = event.payload["generation_tokens"]
                    seen.input_tokens = event.payload["input_tokens"]
                    seen.elapsed = max(
                        self.dependencies.clock() - started, float(duration)
                    )
                    seen.wall_elapsed = max(0.0, self.dependencies.clock() - started)
                # Observation precedes persistence: even an unwritable event log
                # must not erase a delivered fragment/start/terminal from memory.
                append_bytes(
                    directory / "events.jsonl",
                    canonical_json(event.model_dump()) + b"\n",
                )
                if event.kind == "fragment":
                    append_bytes(directory / "partial-raw.txt", text.encode("utf-8"))
                elif event.kind == "terminal":
                    self.ledger.settle(
                        seen.reservation,
                        "finished",
                        math.ceil(seen.elapsed),
                        generations=1,
                    )
                    seen.settled = True
                    failure = (
                        None
                        if finish == "stop"
                        else TechnicalFailure(
                            stage="generation",
                            code="truncated",
                            message="generation finished by length, not stop",
                        )
                    )
                    return GenerationOutput(seen.raw, failure)
                elif event.kind == "error":
                    raise RuntimeError(f"worker error: {event.payload}")
                elif event.kind not in ("generation_started", "progress"):
                    raise RuntimeError(f"unexpected worker event: {event.kind}")
        except BaseException as error:
            seen.halt = error
            # A timeout/crash may occur after the supervisor fsynced fragments but
            # before it returned those queued events. Use the durable journal,
            # not only the provider's in-memory delivered prefix.
            try:
                journal = self.output / "worker" / "raw-fragments.jsonl"
                if journal.exists():
                    captured = _durable_raw(journal, self.row.row_id, seen.raw)
                    extra = captured[len(seen.raw) :]
                    seen.raw = captured
                    if extra:
                        append_bytes(
                            directory / "partial-raw.txt", extra.encode("utf-8")
                        )
            except BaseException as secondary:
                seen.failures.append(("provider partial evidence", secondary))
            seen.elapsed = max(seen.elapsed, self.dependencies.clock() - started)
            seen.wall_elapsed = max(0.0, self.dependencies.clock() - started)
            if seen.reservation is not None and not seen.settled and not seen.completed:
                # Even a failed send may have reached the worker. No proof of
                # non-launch exists, so never silently cancel this reservation.
                try:
                    self.ledger.settle(
                        seen.reservation,
                        "failed",
                        math.ceil(seen.elapsed),
                        generations=1 if seen.started else 0,
                    )
                    seen.settled = True
                except BaseException as secondary:
                    seen.failures.append(("row failure settlement", secondary))
            return GenerationOutput(seen.raw, _failure(error, stage="generation"))


def run_plan(plan_path, *, dry_run=False, approve_run=False) -> dict:
    """Refuse public model execution before any plan, progress or worker effects.

    Historical internal kernels remain for inspection and fictional software tests;
    they are not a supported execution API or a sandbox for arbitrary Python code.
    """
    from messaging_nlp.public_scope import disabled_execution_message

    raise RuntimeError(disabled_execution_message("infer"))


def _report(reporter, **fields):
    if reporter is not None:
        reporter.check()
        reporter.update(**fields)


def _failed_validation_plan(plan_path):
    """Recover only a structurally valid plan with independently safe destinations.

    Never write to an unsafe/malformed plan's nominated destination. Such failures
    must remain exceptions to the caller rather than overwrite protected evidence.
    """
    from messaging_nlp.context import load_public_dataset
    from messaging_nlp.run_plan import _canonical, _safe_paths

    plan_path = Path(plan_path).resolve(strict=True)
    raw = plan_path.read_bytes()
    plan = _canonical(raw, RunPlan)
    if Path(plan.project_root).is_absolute():
        raise ValueError("absolute project root is not permitted")
    root = (plan_path.parent / plan.project_root).resolve(strict=True)
    if not plan_path.is_relative_to(root):
        raise ValueError("plan leaves explicit project root")
    _safe_paths(root, plan, plan_path)
    # Preserve observed bytes even when they fail the frozen identity check;
    # the original expected bindings remain in the immutable plan.
    frozen = {}
    for item in plan.frozen_files:
        try:
            frozen[item.path] = artifact_path(root, item.path).read_bytes()
        except (ValueError, OSError):
            continue
    try:
        dataset = load_public_dataset(
            artifact_path(root, plan.public_root), owner_id=plan.owner_id
        )
        if any(
            dataset.bundle(r.request.message_id).request != r.request for r in plan.rows
        ):
            dataset = None
    except (ValueError, OSError):
        dataset = None
    return SimpleNamespace(
        root=root,
        plan=plan,
        plan_sha256=digest(raw),
        dataset=dataset,
        frozen_bytes=frozen,
        plan_bytes=raw,
        plan_path=plan_path,
    )


def _run_plan(
    plan_path, *, dry_run=False, approve_run=False, dependencies=None, reporter=None
) -> dict:
    if dry_run and approve_run:
        raise ValueError("dry-run and explicit run approval are mutually exclusive")
    dependencies = dependencies or Dependencies()
    entered = dependencies.clock()
    _report(reporter, phase="validating")
    validation_error = None
    try:
        validated = load_and_validate(
            Path(plan_path), model_checker=dependencies.model_checker
        )
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        if dry_run or approve_run:
            raise
        validated = _failed_validation_plan(plan_path)
        validation_error = error
    plan = validated.plan
    _report(
        reporter,
        run_id=plan.run_id,
        planned_rows=len(plan.rows),
        phase="authorization",
        rows_finished=0,
        technical_ok=0,
        attempted_failures=0,
        unattempted=len(plan.rows),
    )
    if dry_run:
        return authorization(validated, dependencies.boot(), dry_run=True)
    if approve_run:
        approve_execution(validated, dependencies.boot())

    # Fresh destination creation is deliberately outside the failure handler:
    # collisions must not turn into writes into an existing evidence directory.
    output = Path(create_output(validated))
    input_bindings = {}
    ledger = child = mutex = None
    mutex_acquired = False
    startup = overhead = storage = None
    startup_elapsed = row_elapsed = 0.0
    ledger_start_bytes = 0
    load_started = None
    load_finished = None
    startup_settlement_attempted = False
    cleanup = CleanupReceipt(verified=True, detail="no owned child created")
    outcomes = []
    progress = {row.ordinal: _RowProgress(row) for row in plan.rows}
    pending_rows = []
    persistence_errors = []
    errors = []
    unavailable_artifacts = set()
    final = {
        "status": "failed",
        "run_id": plan.run_id,
        "plan_sha256": validated.plan_sha256,
        "output_path": str(output),
        "planned_rows": len(plan.rows),
        "rows_saved": 0,
        "pending_rows": pending_rows,
        "semantic_success": "not assessed",
        "authoritative_budget_ledger": plan.ledger_path,
    }
    log_offsets = {}
    pending = None
    reason = None
    fatal = None
    deadline = entered + plan.spending.max_run_seconds

    def failed(error, *, where=None):
        """First-error-wins, including when recording the error itself fails."""
        nonlocal reason, fatal
        detail = f"{type(error).__name__}: {error}"
        errors.append(f"{where}: {detail}" if where else detail)
        if where:
            persistence_errors.append(f"{where}: {detail}")
        first = fatal is None
        if fatal is None:
            fatal = error
            reason = f"{getattr(error, 'code', type(error).__name__)}: {error}"
        path = output / ("failure.json" if first else "additional-failures.jsonl")
        try:
            if first:
                _write_json(path, {"failure": _failure(error).model_dump()})
            else:
                append_bytes(path, canonical_json(_failure(error).model_dump()) + b"\n")
        except BaseException as evidence_error:
            persistence_errors.append(
                f"{path.name}: {type(evidence_error).__name__}: {evidence_error}"
            )
            unavailable_artifacts.add(path.name)

    def attempt(where, action):
        try:
            return action()
        except BaseException as error:
            unavailable_artifacts.add(where)
            failed(error, where=where)
            return None

    def persist(state, *, cleanup=None):
        # Once entered, this row is never persisted again, even if only a prefix
        # of an artifact was written or a receipt fsync has uncertain durability.
        state.persist_attempted = True
        try:
            receipt = _persist_row(
                validated,
                output,
                state.row,
                state.result,
                state.seen,
                ledger=ledger,
                startup=startup,
                reason=str(state.seen.halt) if state.seen.halt else reason,
                input_bindings=input_bindings,
                log_offsets=log_offsets,
                cleanup=cleanup,
            )
        except BaseException as error:
            state.errors.append(f"row persistence: {type(error).__name__}: {error}")
            raise
        outcomes.append(receipt)

    try:
        if reporter is not None:
            reporter.attach(output)
        _report(reporter, phase="retaining_inputs")
        input_bindings = _retain_inputs(validated, output)
        _write_json(
            output / "run-start.json",
            {
                "run_id": plan.run_id,
                "plan_sha256": validated.plan_sha256,
                "monotonic_seconds": entered,
                "status": "gating",
                "planned_rows": len(plan.rows),
            },
        )
        write_new(
            output / "planned-rows.jsonl",
            b"".join(canonical_json(row.model_dump()) + b"\n" for row in plan.rows),
        )
        if validation_error is not None:
            raise validation_error
        boot = dependencies.boot()
        _write_json(output / "current-boot.json", {"boot_identity": boot})
        gate_bytes = {}
        for name in ("readiness", "approval"):
            path = artifact_path(validated.root, getattr(plan, f"{name}_path"))
            try:
                gate_bytes[name] = path.read_bytes()
            except OSError:
                continue
            write_new(output / "gates" / f"{name}.json", gate_bytes[name])
        approval = authorization(validated, boot)
        for name, data in gate_bytes.items():
            path = artifact_path(validated.root, getattr(plan, f"{name}_path"))
            if path.read_bytes() != data:
                raise ValueError(f"{name} changed during authorization")
        _write_json(output / "authorization.json", approval)
        mutex = ModelMutex(validated.root / plan.mutex_path, plan.run_id, boot)
        mutex.acquire()
        mutex_acquired = True
        # An earlier gate observation is not a lease: another completed run may
        # have changed the preparation ledger before this mutex was acquired.
        # Recheck the sealed ledger hash while owning the one-model boundary.
        _write_json(
            output / "authorization-under-mutex.json", authorization(validated, boot)
        )
        ledger_path = validated.root / plan.ledger_path
        if ledger_path.exists():
            opening_ledger = ledger_path.read_bytes()
            ledger_start_bytes = len(opening_ledger)
            write_new(output / "budget-active-opening.json", opening_ledger)
            input_bindings["active_budget_opening"] = ref(
                output, output / "budget-active-opening.json"
            )
        ledger = BudgetLedger(
            validated.root / plan.ledger_path,
            validated.root / plan.opening.path,
            plan.opening.sha256,
            reset_path=validated.root / plan.usage_reset.path,
            reset_sha256=plan.usage_reset.sha256,
            **(
                dict(
                    revision_path=validated.root / plan.revision_authority.path,
                    revision_sha256=plan.revision_authority.sha256,
                )
                if plan.revision_authority is not None
                else {}
            ),
        )
        ledger.register_versions(plan.spending)
        reservation_refs = {"run_id": plan.run_id, "plan_sha256": validated.plan_sha256}
        startup = ledger.reserve(
            "startup", plan.spending.startup_seconds, refs=reservation_refs
        )
        overhead = ledger.reserve(
            "overhead", plan.spending.overhead_seconds, refs=reservation_refs
        )
        storage = ledger.reserve(
            "storage",
            0,
            storage_bytes=plan.spending.storage_bytes,
            refs=reservation_refs,
        )
        _write_json(
            output / "startup-reservations.json",
            {
                "startup": startup,
                "overhead": overhead,
                "storage": storage,
                "budget": ledger.snapshot(),
            },
        )
        if (
            dependencies.clock()
            + plan.spending.startup_seconds
            + plan.resources.cleanup_seconds
            > deadline
        ):
            raise SupervisionError("timeout", "insufficient startup and cleanup runway")
        sensors = dependencies.sensors_factory(validated.root)
        command = [
            str(validated.root / "environments/qwen/.venv/bin/python"),
            "-I",
            "-m",
            "messaging_nlp.qwen_worker",
        ]
        # No source-path injection; retain the exact explicit environment overlays.
        environment = {
            key: os.environ[key]
            for key in ("HOME", "TMPDIR", "LANG", "LC_ALL")
            if key in os.environ
        }
        environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        _write_json(
            output / "runtime-launch.json",
            {
                "command": command,
                "environment_overrides": environment,
                "inherited_environment_sha256": digest(
                    canonical_json(dict(os.environ))
                ),
                "environment_policy": "ChildSession overlays these overrides on inherited environment; Python -I ignores Python path variables",
                "model": plan.model.model_dump(),
                "decoding": plan.decoding.model_dump(),
                "resources": plan.resources.model_dump(),
            },
        )
        child = dependencies.child_factory(
            command, output / "worker", sensors, plan.resources, env=environment
        )
        if reporter is not None:

            def tick():
                from messaging_nlp.run_progress import ProgressPersistenceError

                try:
                    reporter.check()
                except ProgressPersistenceError as error:
                    # Progress is durable execution evidence, unlike the console.
                    raise SupervisionError(
                        "runtime_error", f"progress_persistence: {error}"
                    ) from error
                latest = getattr(child, "latest_sample", None)
                if latest is not None:
                    reporter.update(resource_sample=latest.model_dump())

            child.on_tick = tick
        _report(reporter, phase="loading_model")
        load_started = dependencies.clock()
        child.start()
        load_request = WorkerRequest(
            kind="load",
            payload={
                "root": str(validated.root),
                "model": plan.model.model_dump(),
                "decoding": plan.decoding.model_dump(),
                "source_sha256": {
                    name.removeprefix("src/messaging_nlp/"): digest(raw)
                    for name, raw in validated.frozen_bytes.items()
                    if name.startswith("src/messaging_nlp/") and name.endswith(".py")
                },
            },
        )
        _write_json(output / "load-request.json", load_request.model_dump())
        load_deadline = min(
            load_started + plan.spending.startup_seconds,
            deadline - plan.resources.cleanup_seconds,
        )
        if dependencies.clock() >= load_deadline:
            raise SupervisionError(
                "timeout", "startup allocation expired before load dispatch"
            )
        child.send(load_request, deadline=load_deadline)
        ready = child.wait_event(load_deadline)
        if ready.kind != "ready" or ready.row_id is not None:
            raise RuntimeError(f"worker did not become ready: {ready.model_dump()}")
        _write_json(output / "runtime-ready.json", ready.model_dump())
        startup_elapsed = dependencies.clock() - load_started
        load_finished = dependencies.clock()
        startup_settlement_attempted = True
        ledger.settle(startup, "finished", math.ceil(startup_elapsed))
        _report(reporter, phase="model_ready")
        for row in plan.rows:
            state = progress[row.ordinal]
            seen = state.seen
            _report(
                reporter,
                phase="row_start",
                row_ordinal=row.ordinal,
                condition=row.condition,
                generation_tokens=None,
                last_token_monotonic=None,
                row_started_monotonic=time.monotonic(),
                token_limit=plan.decoding.max_output_tokens,
                outcome=None,
            )
            if row.preparation_failure is not None:
                result = _unexecuted(validated, row, "frozen preparation blocked")
                state.result = result
            else:
                now = dependencies.clock()
                if (
                    now + CALL_TIMEOUT_SECONDS + plan.resources.cleanup_seconds
                    > deadline
                ):
                    raise SupervisionError(
                        "timeout", "next full 120-second row plus cleanup cannot fit"
                    )
                overhead_used = now - entered - startup_elapsed - row_elapsed
                if (
                    overhead_used + plan.resources.cleanup_seconds
                    > plan.spending.overhead_seconds
                ):
                    raise SupervisionError(
                        "timeout", "supervision/cleanup allocation exhausted"
                    )
                available = ledger.snapshot()
                if (
                    available["remaining_seconds"] < CALL_TIMEOUT_SECONDS
                    or available["remaining_generations"] < 1
                ):
                    raise RuntimeError(
                        "next full 120-second/one-generation reservation cannot fit budget"
                    )
                pending = (row, seen)
                provider = _Provider(
                    validated,
                    row,
                    output,
                    child,
                    ledger,
                    dependencies,
                    deadline - plan.resources.cleanup_seconds,
                    seen,
                    reporter=reporter,
                )
                application_started = dependencies.clock()
                seen.application_started = application_started
                result = execute_condition(
                    validated.dataset,
                    row.request.message_id,
                    row.condition,
                    validated.prepared[row.request.message_id],
                    provider,
                    root=validated.root,
                    run_id=plan.run_id,
                    binding=validated.binding,
                    max_output_tokens=plan.decoding.max_output_tokens,
                )
                state.result = result
                if not result.provider_calls:
                    seen.preprocessing_seconds = max(
                        0.0, dependencies.clock() - application_started
                    )
                    seen.halt = RuntimeError(
                        "a frozen ready preparation was rejected before provider invocation: "
                        f"{result.row.failure.message if result.row.failure else 'unknown preparation failure'}"
                    )
                # Provider exceptions may be returned as ApplicationResult failures;
                # record them BEFORE any timing/row/receipt writes can fail.
                if seen.halt is not None:
                    failed(seen.halt)
                for where, error in seen.failures:
                    failed(error, where=where)
                _write_json(
                    _row_dir(output, row) / "timing.json",
                    {
                        "application_seconds": dependencies.clock()
                        - application_started,
                        "generation_seconds": seen.wall_elapsed,
                        "generation_charge_observation_seconds": seen.elapsed,
                        "preprocessing_verification_seconds": seen.preprocessing_seconds,
                        "output_validation_seconds": max(
                            0.0,
                            dependencies.clock()
                            - application_started
                            - seen.preprocessing_seconds
                            - seen.wall_elapsed,
                        ),
                        "preparation": "frozen pre-restart artifact; not recomputed",
                    },
                )
                row_elapsed += seen.wall_elapsed
                seen.wall_accounted = True
            persist(state)
            _report(
                reporter,
                phase="row_saved",
                rows_saved=len(outcomes),
                rows_finished=sum(r.generation_completed for r in outcomes),
                technical_ok=sum(r.disposition == "completed" for r in outcomes),
                attempted_failures=sum(
                    r.disposition == "attempted_failure" for r in outcomes
                ),
                outcome=outcomes[-1].disposition,
                unattempted=len(plan.rows)
                - sum(r.provider_calls > 0 for r in outcomes),
            )
            pending = None
            atomic_json(
                output / "status.json",
                {
                    "status": "running",
                    "rows_saved": len(outcomes),
                    "planned_rows": len(plan.rows),
                },
            )
            if seen.halt:
                raise seen.halt
            if (
                sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                > plan.spending.storage_bytes
            ):
                raise SupervisionError(
                    "resource_stop", "reserved evidence storage exceeded"
                )
    except BaseException as error:
        if pending is not None and pending[1].halt is not None and fatal is None:
            failed(pending[1].halt)
        if pending is not None and not pending[1].wall_accounted:
            row_elapsed += pending[1].wall_elapsed
            pending[1].wall_accounted = True
        if load_started is not None and load_finished is None:
            load_finished = dependencies.clock()
            startup_elapsed = max(0.0, load_finished - load_started)
        if error is not fatal:
            state_errors = any(state.errors for state in progress.values())
            failed(
                error,
                where="row persistence"
                if state_errors
                else "execution evidence"
                if isinstance(error, OSError)
                else None,
            )
    finally:
        try:
            if reporter is not None:
                attempt("progress cleanup", lambda: reporter.update(phase="cleanup"))
            if child is not None:
                # Retain the mutex unless close returns independently verified
                # absence. Logging failures must never change that observation.
                cleanup = CleanupReceipt(verified=False, detail="close not verified")
                try:
                    closed = child.close()
                    if not isinstance(closed, CleanupReceipt):
                        raise RuntimeError("child close returned no cleanup receipt")
                    cleanup = closed
                    if not cleanup.verified:
                        failed(
                            RuntimeError(
                                f"owned child cleanup unverified: {cleanup.detail}"
                            )
                        )
                except BaseException as error:
                    cleanup = CleanupReceipt(
                        verified=False,
                        detail=f"close failed: {type(error).__name__}: {error}",
                    )
                    failed(error)

                def supervisor_evidence():
                    supervisor_failure = getattr(child, "failure", None)
                    if supervisor_failure is not None:
                        failed(
                            SupervisionError(
                                supervisor_failure.get("code", "runtime_error"),
                                supervisor_failure.get(
                                    "message", str(supervisor_failure)
                                ),
                            )
                        )
                        _write_json(
                            output / "supervisor-failure.json", supervisor_failure
                        )

                attempt("supervisor-failure.json", supervisor_evidence)
            attempt(
                "cleanup.json",
                lambda: _write_json(output / "cleanup.json", cleanup.model_dump()),
            )

            def global_logs():
                # Global logs become hashable only after the child is closed.
                worker = output / "worker"
                if not worker.exists():
                    return
                if cleanup.verified:
                    logs = {
                        path.name: ref(output, path).model_dump()
                        for path in sorted(worker.iterdir())
                        if path.is_file() and not path.is_symlink()
                    }
                else:
                    logs = {
                        name: artifact.model_dump()
                        for name, artifact in _snapshot_logs(
                            output, output / "unverified-log-snapshot", {}
                        ).items()
                    }
                _write_json(
                    output / "final-worker-logs.json",
                    {
                        "cleanup_verified": cleanup.verified,
                        "partial": not cleanup.verified,
                        "logs": logs,
                    },
                )

            attempt("final-worker-logs.json", global_logs)
            if (
                ledger is not None
                and startup is not None
                and not startup_settlement_attempted
            ):
                attempt(
                    "startup settlement",
                    lambda: ledger.settle(
                        startup, "failed", math.ceil(startup_elapsed)
                    ),
                )

            saved = {receipt.ordinal for receipt in outcomes}
            for row in plan.rows:
                if row.ordinal in saved:
                    continue
                state = progress[row.ordinal]

                def save_remaining():
                    if state.result is None:
                        # Only if Stage 3 never returned an ApplicationResult may
                        # infrastructure failure need an observed-failure wrapper.
                        result = _unexecuted(
                            validated, row, reason or "run stopped before dispatch"
                        )
                        if state.seen.calls:
                            data = result.row.model_dump()
                            data.update(
                                status="failed",
                                raw_output=state.seen.raw,
                                failure=_failure(
                                    state.seen.halt
                                    or fatal
                                    or RuntimeError("coordinator stopped")
                                ).model_dump(),
                            )
                            result = ApplicationResult(
                                RunRow.model_validate(data),
                                state.seen.calls,
                                None,
                                None,
                            )
                        state.result = result
                    persist(state, cleanup=cleanup)

                if not state.persist_attempted:
                    before_errors = len(persistence_errors)
                    attempt(f"row {row.row_id} persistence", save_remaining)
                    state.errors.extend(persistence_errors[before_errors:])
                if row.ordinal not in {receipt.ordinal for receipt in outcomes}:
                    # A missing receipt is explicitly pending, never completed or
                    # repersisted with a different outcome after a partial write.
                    record = _pending_record(output, state, reason)
                    pending_rows.append(record.model_dump())
            if pending_rows:
                if fatal is None:
                    failed(RuntimeError("one or more row receipts are incomplete"))
                attempt(
                    "pending-rows.json",
                    lambda: _write_json(output / "pending-rows.json", pending_rows),
                )

            if reporter is not None:

                def close_progress():
                    reporter.update(
                        phase="finalizing",
                        rows_saved=len(outcomes),
                        rows_finished=sum(r.generation_completed for r in outcomes),
                        technical_ok=sum(
                            r.disposition == "completed" for r in outcomes
                        ),
                        attempted_failures=sum(
                            r.disposition == "attempted_failure" for r in outcomes
                        ),
                        unattempted=sum(r.provider_calls == 0 for r in outcomes),
                        stop_reason=reason,
                    )
                    reporter.close()
                    reporter.check()

                attempt("progress finalization", close_progress)

            if ledger is not None:
                if overhead is not None:
                    # Include cleanup, all unattempted-row receipts and storage
                    # finalization. Subtract only disjoint startup/provider wall
                    # spans, never charge overlapping run/row seconds twice.
                    elapsed = max(
                        0,
                        dependencies.clock() - entered - startup_elapsed - row_elapsed,
                    )
                    attempt(
                        "overhead settlement",
                        lambda: ledger.settle(
                            overhead,
                            "finished" if cleanup.verified else "failed",
                            math.ceil(elapsed),
                        ),
                    )

                def budget_evidence():
                    budget = ledger.snapshot()
                    budget["storage_settlement"] = {
                        "snapshot_stage": "before final output closure and storage settlement",
                        "authoritative_ledger_path": plan.ledger_path,
                        "reservation_id": storage,
                        "ledger_start_bytes": ledger_start_bytes,
                        "scope": "owned output bytes plus positive canonical-ledger growth including settlement metadata",
                    }
                    _write_json(output / "budget-final.json", budget)

                attempt("budget-final.json", budget_evidence)
            if dependencies.clock() > deadline:
                failed(
                    SupervisionError(
                        "timeout",
                        "overall run deadline exceeded including cleanup/storage",
                    )
                )

            def update_final():
                final.update(
                    {
                        "status": "failed"
                        if fatal is not None or pending_rows
                        else "completed",
                        "run_id": plan.run_id,
                        "plan_sha256": validated.plan_sha256,
                        "output_path": str(output),
                        "planned_rows": len(plan.rows),
                        "rows_saved": len(outcomes),
                        "pending_rows": pending_rows,
                        "generation_completed": sum(
                            r.generation_completed for r in outcomes
                        ),
                        "technical_ok": sum(
                            r.disposition == "completed" for r in outcomes
                        ),
                        "cleanup_verified": cleanup.verified,
                        "stop_reason": reason,
                        "elapsed_seconds": max(0.0, dependencies.clock() - entered),
                        "semantic_success": "not assessed",
                        "authoritative_budget_ledger": plan.ledger_path,
                        "storage_reservation_id": storage,
                        "errors": list(errors),
                        "persistence_errors": list(persistence_errors),
                        "unavailable_artifacts": sorted(unavailable_artifacts),
                        "durability": "partial"
                        if persistence_errors or pending_rows or not cleanup.verified
                        else "available",
                    }
                )

            before = len(errors)
            update_final()
            attempt(
                "completion.json",
                lambda: _write_json(output / "completion.json", final),
            )
            update_final()
            attempt("status.json", lambda: atomic_json(output / "status.json", final))

            def refresh_summaries():
                update_final()
                # Only explicitly mutable, successfully created run summaries may
                # change. Never retry an artifact whose creation/update failed.
                if "completion.json" not in unavailable_artifacts:
                    attempt(
                        "completion.json",
                        lambda: atomic_json(output / "completion.json", final),
                    )
                if "status.json" not in unavailable_artifacts:
                    attempt(
                        "status.json",
                        lambda: atomic_json(output / "status.json", final),
                    )

            if len(errors) != before:
                # Close evidence for already observed failures BEFORE measuring
                # storage, so those summary bytes are included in settlement.
                refresh_summaries()

            def storage_settlement():
                if ledger is not None and storage is not None:
                    size = sum(
                        p.stat().st_size for p in output.rglob("*") if p.is_file()
                    )
                    ledger.settle_storage(
                        storage,
                        "failed" if fatal is not None else "finished",
                        output_bytes=size,
                        ledger_start_bytes=ledger_start_bytes,
                    )
                    if ledger.snapshot().get("caps_exceeded"):
                        raise SupervisionError(
                            "resource_stop",
                            "observed final artifacts exceed budget caps",
                        )
                if dependencies.clock() > deadline:
                    raise SupervisionError(
                        "timeout", "final accounting exceeded the aggregate deadline"
                    )

            before = len(errors)
            attempt("storage settlement", storage_settlement)
            if len(errors) != before:
                refresh_summaries()
            update_final()
        except BaseException as error:
            # Defensive finalization guard; the mutex finally below still runs.
            failed(error, where="finalization")
            final.update(
                status="failed",
                stop_reason=reason,
                cleanup_verified=cleanup.verified,
                rows_saved=len(outcomes),
                pending_rows=pending_rows,
                persistence_errors=list(persistence_errors),
                errors=list(errors),
                durability="partial",
            )
        finally:
            # OUTERMOST disposition: neither evidence writes, receipt validation,
            # accounting nor even the defensive finalizer can bypass this point.
            if mutex_acquired:
                try:
                    mutex.release(cleanup_verified=cleanup.verified)
                except BaseException as error:
                    failed(error, where="mutex disposition")
                    final.update(
                        status="failed",
                        stop_reason=reason,
                        persistence_errors=list(persistence_errors),
                        errors=list(errors),
                        durability="partial",
                    )
                    if "completion.json" not in unavailable_artifacts:
                        attempt(
                            "completion.json",
                            lambda: atomic_json(output / "completion.json", final),
                        )
                    if "status.json" not in unavailable_artifacts:
                        attempt(
                            "status.json",
                            lambda: atomic_json(output / "status.json", final),
                        )
    # Return the latest in-memory failures even if the very last best-effort
    # evidence write also failed. This is not a promise of durable artifacts.
    final.update(
        status="failed" if fatal is not None or pending_rows else "completed",
        stop_reason=reason,
        errors=list(errors),
        persistence_errors=list(persistence_errors),
        unavailable_artifacts=sorted(unavailable_artifacts),
        durability="partial"
        if persistence_errors or pending_rows or not cleanup.verified
        else "available",
    )
    return final
