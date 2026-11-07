"""Serial 738-generation role-hints ablation with post-hoc binding analysis.

This is a temporary fix-validation experiment, not the canonical study ledger.
Each prompt is generated once; every raw response is then resolved with
``bind_addressed=True`` and ``False`` so the binding factor costs no extra model
call. Progress is printed as a single percentage line.
"""

from __future__ import annotations

import hashlib
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from messaging_nlp.application import resolve_envelope
from messaging_nlp.application_types import PreparationPacket
from messaging_nlp.context import PublicBundle
from messaging_nlp.drafting import validate_envelope
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.run_contracts import (
    MUTEX_PATH,
    DecodePolicy,
    ModelSpec,
    ResourceLimits,
    TokenReceipt,
    WorkerRequest,
)
from messaging_nlp.run_storage import atomic_json, write_new
from messaging_nlp.run_supervision import (
    ChildSession,
    MacSensors,
    ModelMutex,
    boot_identity,
)
from messaging_nlp.stage5 import preparation_sources
from messaging_nlp.temporal import TimezoneBinding


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_map(root: Path) -> dict[str, str]:
    policy = loads_json((root / "configs/study-policy.json").read_text())
    refs = preparation_sources(root, policy["data"]["public_root"])
    prefix = "src/messaging_nlp/"
    return {
        ref.path[len(prefix) :]: ref.sha256
        for ref in refs
        if ref.path.startswith(prefix)
    }


def _binding(root: Path) -> TimezoneBinding:
    policy = loads_json((root / "configs/study-policy.json").read_text())
    identity = loads_json(
        (root / policy["temporal"]["timezone_identity"]["identity_file"]).read_text()
    )
    return TimezoneBinding(
        identity["time_zone"],
        root / identity["tzif_path"],
        identity["tzif_sha256"],
    )


def _safe(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("ablation artifact leaves the project root")
    return path


def _load_plan(root: Path, plan_path: Path):
    plan_path = plan_path.resolve(strict=True)
    raw = plan_path.read_bytes()
    plan = loads_json(raw.decode("utf-8"))
    if canonical_json(plan) != raw:
        raise ValueError("ablation plan is not canonical JSON")
    if plan.get("schema_version") != "nlp3-stage7-ablation-v1":
        raise ValueError("unexpected ablation plan schema")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 738:
        raise ValueError("ablation plan requires exactly 738 generated rows")
    if [row.get("ordinal") for row in rows] != list(range(738)):
        raise ValueError("ablation ordinals are not contiguous")
    if len({row.get("row_id") for row in rows}) != 738:
        raise ValueError("duplicate ablation row ID")
    if {row.get("variant") for row in rows} != {"hints_on", "hints_off"}:
        raise ValueError("ablation must contain both prompt variants")
    if {row.get("condition") for row in rows} != {"baseline", "B", "A+B"}:
        raise ValueError("ablation must contain all three conditions")
    readiness_path = plan_path.with_name("readiness.json")
    readiness = loads_json(readiness_path.read_text())
    if readiness.get("plan_sha256") != _digest(raw):
        raise ValueError("ablation plan changed after readiness")
    if _source_map(root) != readiness.get("source_sha256"):
        raise ValueError("installed source differs from readiness")
    if boot_identity() == readiness.get("boot_identity"):
        raise RuntimeError("restart required: boot identity is unchanged")
    output = _safe(root, plan["output"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("ablation output already exists; no replay")
    model = ModelSpec.model_validate(plan["model"])
    if model.receipt.sha256 != readiness.get("model_receipt_sha256"):
        raise ValueError("ablation model receipt differs from readiness")
    decoding = DecodePolicy.model_validate(plan["decoding"])
    return plan, readiness, rows, output, model, decoding


def _messages(
    root: Path, row: dict
) -> tuple[list[dict], TokenReceipt, PreparationPacket]:
    path = _safe(root, row["messages"])
    messages = loads_json(path.read_text())
    if _digest(canonical_json(messages)) != row["messages_sha256"]:
        raise ValueError(f"ablation message bytes changed: {row['row_id']}")
    receipt_path = _safe(root, row["token_receipt"])
    if _digest(receipt_path.read_bytes()) != row.get("token_receipt_sha256"):
        raise ValueError(f"ablation token receipt bytes changed: {row['row_id']}")
    receipt = TokenReceipt.model_validate(loads_json(receipt_path.read_text()))
    if receipt.messages_sha256 != row["messages_sha256"]:
        raise ValueError(f"ablation token receipt differs: {row['row_id']}")
    if receipt.token_count > 2048:
        raise ValueError(f"ablation prompt is overlength: {row['row_id']}")
    packet_path = _safe(root, row["packet"])
    packet = PreparationPacket.model_validate(loads_json(packet_path.read_text()))
    if _digest(packet_path.read_bytes()) != row["packet_sha256"]:
        raise ValueError(f"ablation packet bytes changed: {row['row_id']}")
    return messages, receipt, packet


def _consume(child, row_id: str, deadline: float):
    fragments = []
    terminal = None
    while True:
        event = child.wait_event(deadline)
        if event.row_id != row_id:
            raise ValueError("ablation received another row's event")
        if event.kind == "generation_started":
            continue
        if event.kind == "fragment":
            fragments.append(event.payload["text"])
            continue
        if event.kind == "terminal":
            terminal = event.payload
            break
        if event.kind == "progress":
            continue
        raise RuntimeError(f"unexpected ablation worker event: {event.kind}")
    return "".join(fragments), terminal


def _resolve(
    raw: str, packet: PreparationPacket, binding: TimezoneBinding, *, bind: bool
):
    envelope = resolve_envelope(raw, packet, bind_addressed=bind)
    validate_envelope(
        envelope,
        bundle=PublicBundle(request=packet.request, context=packet.context),
        candidates=packet.search.candidates,
        anchors=packet.anchors,
        binding=binding,
    )
    return envelope


def _status(output: Path, total: int, outcomes: list[dict], started: float, **extra):
    atomic_json(
        output / "status.json",
        dict(
            schema_version="nlp3-stage7-ablation-status-v1",
            status=extra.pop("status", "running"),
            total=total,
            completed=len(outcomes),
            technical_valid_on=sum(row["technical_valid_on"] for row in outcomes),
            technical_valid_off=sum(row["technical_valid_off"] for row in outcomes),
            qwen_generations=len(outcomes),
            evaluation_cells=len(outcomes) * 2,
            elapsed_seconds=time.monotonic() - started,
            accounting="temporary ablation; canonical study ledger untouched",
            **extra,
        ),
    )


def run(plan_path: Path, *, approve: bool, dry_run: bool) -> int:
    """Refuse the historical public runner without inspecting the supplied path."""
    from messaging_nlp.public_scope import disabled_execution_message

    raise RuntimeError(disabled_execution_message("ablation"))


def _historical_run(plan_path: Path, *, approve: bool, dry_run: bool) -> int:
    """Retained historical implementation, not a supported public execution API."""
    root = Path.cwd().resolve()
    plan_path = Path(plan_path)
    try:
        plan, readiness, rows, output, model, decoding = _load_plan(root, plan_path)
    except RuntimeError as error:
        print(str(error), flush=True)
        return 2
    except Exception as error:
        print(f"ablation refused: {error}", flush=True)
        return 1
    if dry_run:
        print("ready", flush=True)
        return 0
    if not approve:
        print("approval required", flush=True)
        return 2
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "plan.json", plan_path.read_bytes())
    write_new(
        output / "readiness.json", plan_path.with_name("readiness.json").read_bytes()
    )
    binding = _binding(root)
    outcomes: list[dict] = []
    started = time.monotonic()
    last_percent = -1
    child = None
    mutex = ModelMutex(root / MUTEX_PATH, "stage7-ablation", boot_identity())
    cleanup_verified = False
    mutex.acquire()
    try:
        child = ChildSession(
            [
                str(root / "environments/qwen/.venv/bin/python"),
                "-I",
                "-m",
                "messaging_nlp.qwen_worker",
            ],
            output / "worker",
            MacSensors(root),
            ResourceLimits(
                memory_warning_seconds=10,
                min_free_disk_bytes=10737418240,
            ),
        )
        child.start()
        load_started = time.monotonic()
        child.send(
            WorkerRequest(
                kind="load",
                payload={
                    "root": str(root),
                    "model": model.model_dump(),
                    "decoding": decoding.model_dump(),
                    "source_sha256": _source_map(root),
                },
            ),
            deadline=load_started + 300,
        )
        ready = child.wait_event(load_started + 300)
        if (
            ready.kind != "ready"
            or ready.payload.get("worker_source_sha256") != _source_map(root)
            or ready.payload.get("decoding") != decoding.model_dump()
            or ready.payload.get("receipt_sha256") != model.receipt.sha256
        ):
            raise ValueError("ablation runtime identity mismatch")
        write_new(output / "runtime-ready.json", canonical_json(ready.model_dump()))
        for row in rows:
            row_id = row["row_id"]
            messages, receipt, packet = _messages(root, row)
            directory = output / "rows" / row_id
            write_new(directory / "messages.json", canonical_json(messages))
            outcome = dict(
                row_id=row_id,
                request_id=row["request_id"],
                condition=row["condition"],
                variant=row["variant"],
                stratum=row["stratum"],
                attempted=True,
                generation_started=False,
                generation_completed=False,
                technical_valid_on=False,
                technical_valid_off=False,
                disposition="attempted_unresolved",
            )
            row_started = time.monotonic()
            child.send(
                WorkerRequest(
                    kind="generate",
                    row_id=row_id,
                    payload={
                        "messages": messages,
                        "token_receipt": receipt.model_dump(),
                    },
                ),
                deadline=row_started + 120,
            )
            raw, terminal = _consume(child, row_id, row_started + 120)
            write_new(directory / "raw-output.txt", raw.encode())
            outcome.update(
                generation_started=True,
                generation_completed=True,
                terminal=terminal,
                elapsed_seconds=time.monotonic() - row_started,
            )
            if terminal.get("finish_reason") == "length":
                outcome["disposition"] = "length_failure"
            else:
                for bind, label in ((True, "on"), (False, "off")):
                    try:
                        envelope = _resolve(raw, packet, binding, bind=bind)
                    except Exception as error:
                        outcome[f"validation_error_{label}"] = (
                            f"{type(error).__name__}: {error}"
                        )
                    else:
                        outcome[f"technical_valid_{label}"] = True
                        outcome[f"action_{label}"] = envelope.decision.action
                        if envelope.decision.selection is not None:
                            outcome[f"recipient_{label}"] = (
                                envelope.decision.selection.recipient_contact_ids
                            )
                        atomic_json(
                            directory / f"resolved-{label}.json",
                            envelope.model_dump(),
                        )
                if outcome["technical_valid_on"] or outcome["technical_valid_off"]:
                    outcome["disposition"] = "technically_valid"
            outcomes.append(outcome)
            atomic_json(directory / "outcome.json", outcome)
            _status(output, len(rows), outcomes, started)
            percent = len(outcomes) * 100 // len(rows)
            if percent != last_percent:
                print(f"progress: {percent}%", flush=True)
                last_percent = percent
        cleanup = child.close()
        cleanup_verified = cleanup.verified
        child = None
        if not cleanup_verified:
            raise RuntimeError("ablation worker cleanup failed")
        _status(output, len(rows), outcomes, started, status="completed")
        atomic_json(
            output / "usage.json",
            dict(
                qwen_generations=len(outcomes),
                evaluation_cells=len(outcomes) * 2,
                elapsed_seconds=time.monotonic() - started,
                canonical_ledger_charges=0,
                accounting="temporary ablation; canonical study ledger untouched",
                completed_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        print(
            f"completed {len(outcomes)}/{len(rows)}; "
            f"technical_valid_on={sum(row['technical_valid_on'] for row in outcomes)}; "
            f"technical_valid_off={sum(row['technical_valid_off'] for row in outcomes)}; "
            f"output={output}",
            flush=True,
        )
        return 0
    except BaseException as error:
        try:
            (output / "error.txt").write_text(traceback.format_exc())
        except OSError:
            pass
        try:
            _status(
                output,
                len(rows),
                outcomes,
                started,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        except Exception:
            pass
        print(f"ablation failed after {len(outcomes)}/{len(rows)}: {error}", flush=True)
        return 1
    finally:
        if child is not None:
            try:
                cleanup = child.close()
                cleanup_verified = cleanup.verified
            except Exception:
                cleanup_verified = False
        try:
            mutex.release(cleanup_verified=cleanup_verified)
        except Exception:
            pass
