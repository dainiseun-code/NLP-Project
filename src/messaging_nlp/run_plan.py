"""Model-free frozen-plan authentication and execution consent gates."""

from __future__ import annotations

import platform
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Callable

from messaging_nlp.application_types import PreparedInputs, Prompt, SealedInput
from messaging_nlp.context import PublicDataset, load_public_dataset
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.paths import ensure_output_disjoint
from messaging_nlp.preparation import (
    artifact_path,
    digest,
    policy_identity,
    verify_input,
)
from messaging_nlp.prompts import build_prompt
from messaging_nlp.run_contracts import (
    OPENING_SHA256,
    RESET_SHA256,
    ArtifactRef,
    ExecutionApproval,
    ReadinessReceipt,
    RunPlan,
    TokenReceipt,
    valid_boot_identity,
)
from messaging_nlp.run_revision import load_authority
from messaging_nlp.run_storage import read_ref, write_new
from messaging_nlp.temporal import TimezoneBinding


@dataclass(frozen=True)
class ValidatedPlan:
    root: Path
    plan: RunPlan
    plan_sha256: str
    dataset: PublicDataset
    binding: TimezoneBinding
    prepared: dict[str, PreparedInputs]
    prompts: dict[str, Prompt]
    token_receipts: dict[str, TokenReceipt]
    frozen_bytes: dict[str, bytes]
    plan_bytes: bytes
    plan_path: Path


def _canonical(raw: bytes, model):
    value = model.model_validate(loads_json(raw.decode("utf-8")))
    if canonical_json(value.model_dump()) != raw:
        raise ValueError("frozen document requires exact canonical bytes")
    return value


def _no_symlink_chain(root: Path, path: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("execution paths must not traverse symlinks")


def _safe_paths(root: Path, plan: RunPlan, plan_path: Path) -> None:
    protected = [
        root / name
        for name in (
            "src",
            "data",
            "models",
            "environments",
            "configs",
            "prompts",
            "docs",
            "pyproject.toml",
            "uv.lock",
            ".git",
            "Old-version",
        )
    ]
    protected += [root / item.path for item in plan.frozen_files]
    for control in (root / plan.readiness_path, root / plan.approval_path):
        _no_symlink_chain(root, control)
        ensure_output_disjoint(control, protected + [plan_path])
    ensure_output_disjoint(root / plan.readiness_path, [root / plan.approval_path])
    protected += [plan_path, root / plan.readiness_path, root / plan.approval_path]
    destinations = [
        root / p for p in (plan.output_path, plan.ledger_path, plan.mutex_path)
    ]
    for destination in destinations:
        _no_symlink_chain(root, destination)
        ensure_output_disjoint(destination, protected)
        ensure_output_disjoint(
            destination, [p for p in destinations if p != destination]
        )
    if len(set(destinations)) != len(destinations):
        raise ValueError("execution destinations must be distinct")


def load_and_validate(
    plan_path: Path, *, model_checker: Callable | None = None
) -> ValidatedPlan:
    """No tokenizer, generation, retrieval or extraction is executed here."""
    plan_path = Path(plan_path).resolve(strict=True)
    raw = plan_path.read_bytes()
    plan = _canonical(raw, RunPlan)
    if Path(plan.project_root).is_absolute():
        raise ValueError("project root must be explicit plan-relative, not absolute")
    root = (plan_path.parent / plan.project_root).resolve(strict=True)
    if not plan_path.is_relative_to(root):
        raise ValueError("sealed plan must remain inside explicit project root")
    _safe_paths(root, plan, plan_path)
    frozen = {item.path: read_ref(root, item) for item in plan.frozen_files}

    def bound(item: ArtifactRef) -> bytes:
        data = read_ref(root, item)
        if frozen.get(item.path) != data:
            raise ValueError(f"artifact absent from frozen byte inventory: {item.path}")
        return data

    if plan.opening.sha256 != OPENING_SHA256 or plan.usage_reset.sha256 != RESET_SHA256:
        raise ValueError("opening or active reset trust anchor mismatch")
    bound(plan.opening)
    bound(plan.usage_reset)
    bound(plan.timezone)
    policy_raw = frozen.get("configs/study-policy.json")
    if policy_raw is None:
        raise ValueError("missing frozen study configuration/policy bytes")
    policy = loads_json(policy_raw.decode("utf-8"))
    for name in (
        "temperature",
        "seed",
        "enable_thinking",
        "max_input_tokens",
        "max_output_tokens",
        "call_timeout_seconds",
    ):
        expected = getattr(plan.decoding, name)
        actual = policy["decoding"].get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("study policy and frozen decoding disagree")
    reset = policy["resources"].get("active_usage_reset", {})
    if (
        reset.get("record") != plan.usage_reset.path
        or reset.get("sha256") != plan.usage_reset.sha256
    ):
        raise ValueError("study policy and active usage reset disagree")
    template_path = f"{plan.model.path}/chat_template.jinja"
    if (
        template_path not in frozen
        or digest(frozen[template_path]) != plan.model.template_sha256
    ):
        raise ValueError("missing/mismatched retained chat-template bytes")
    expected_version_refs = {
        "prompt_policy_sha256": policy_identity(),
        "template_sha256": plan.model.template_sha256,
        "decoding_sha256": digest(canonical_json(plan.decoding.model_dump())),
    }
    if plan.spending.assistant_version_refs != expected_version_refs:
        raise ValueError("common prompt version must bind policy/template/decoding")
    # Match actual installed source bytes, not a commit that may be unavailable.
    package = Path(str(files("messaging_nlp")))
    sources = {
        f"src/messaging_nlp/{p.relative_to(package).as_posix()}": p.read_bytes()
        for p in package.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    supplied = {item.path for item in plan.frozen_files if item.role == "source"}
    if supplied != set(sources):
        raise ValueError("frozen source inventory differs from installed package")
    for name, data in sources.items():
        if frozen[name] != data:
            raise ValueError(f"installed source identity mismatch: {name}")
    if plan.revision_authority is not None:
        bound(plan.revision_authority)
        revision = load_authority(
            root / plan.revision_authority.path, plan.revision_authority.sha256
        )
        previous = revision["predecessor_plan"]
        if (
            revision["root"] != root
            or plan.run_id != revision["body"]["run_id"]
            or plan.decoding.max_output_tokens
            != revision["body"]["scope"]["max_output_tokens"]
            or plan.resources.memory_warning_seconds
            != revision["body"]["scope"]["memory_warning_seconds"]
            or plan.model.model_dump() != previous["model"]
            or plan.roster.model_dump() != previous["roster"]
            or [(r.request.model_dump(), r.condition) for r in plan.rows]
            != [(r["request"], r["condition"]) for r in previous["rows"]]
        ):
            raise ValueError("plan is outside the authorized full-roster revision")
        current_files = {item.path: item for item in plan.frozen_files}
        for item in previous["frozen_files"]:
            if item["path"].startswith("data/") or item["path"] in (
                "configs/evaluation-policy.json",
                "docs/stage4b-evaluation-contract.md",
            ):
                actual = current_files.get(item["path"])
                if actual is None or (actual.sha256, actual.bytes) != (
                    item["sha256"],
                    item["bytes"],
                ):
                    raise ValueError("revision changed frozen research/review bytes")
        for row, prior in zip(plan.rows, previous["rows"], strict=True):
            if row.prompt is None or prior["prompt"] is None:
                raise ValueError("full revision requires all retained ready prompts")
            if row.prompt.sha256 != prior["prompt"]["sha256"]:
                raise ValueError("revision changed original prompt bytes")
            retained = loads_json(
                read_ref(
                    root, ArtifactRef.model_validate(prior["preparation"])
                ).decode()
            )
            rebound = loads_json(bound(row.preparation).decode())
            retained["policy_sha256"] = rebound["policy_sha256"]
            if retained != rebound:
                raise ValueError("revision changed extraction or Python preprocessing")
    for name in (
        "pyproject.toml",
        "uv.lock",
        "environments/encoder/pyproject.toml",
        "environments/encoder/uv.lock",
        "environments/qwen/pyproject.toml",
        "environments/qwen/uv.lock",
    ):
        if name not in frozen:
            raise ValueError(f"missing frozen environment/lock identity: {name}")
    locked = {
        p["name"]: p["version"]
        for p in tomllib.loads(frozen["uv.lock"].decode())["package"]
    }
    if set(plan.main_environment) != {"python", "pydantic", "messaging-nlp"}:
        raise ValueError("incomplete main environment identity")
    for name, expected in plan.main_environment.items():
        actual = platform.python_version() if name == "python" else version(name)
        if actual != expected or (name != "python" and locked.get(name) != expected):
            raise ValueError(
                f"main environment identity differs from freeze/lock: {name}"
            )
    if model_checker is None:
        from messaging_nlp.qwen_worker import verify_model

        model_checker = verify_model
    model_checker(root, plan.model)
    bound(plan.model.receipt)
    public = artifact_path(root, plan.public_root)
    for name in ("requests", "snapshots", "contacts", "calendar", "history"):
        if f"{plan.public_root}/{name}.jsonl" not in frozen:
            raise ValueError("missing public input frozen identity")
    dataset = load_public_dataset(public, owner_id=plan.owner_id)
    binding = TimezoneBinding(
        plan.timezone_name, root / plan.timezone.path, plan.timezone.sha256
    )
    roster_raw = bound(plan.roster).decode("utf-8")
    roster = [loads_json(line) for line in roster_raw.splitlines()]
    if not roster or any(not isinstance(r, dict) for r in roster):
        raise ValueError("invalid frozen roster")
    expected_order = []
    for row in roster:
        order = row.get("condition_order")
        if not isinstance(order, list) or sorted(order) != ["A+B", "B", "baseline"]:
            raise ValueError("invalid roster condition order")
        expected_order.extend((row["message_id"], condition) for condition in order)
    if expected_order != [(r.request.message_id, r.condition) for r in plan.rows]:
        raise ValueError("plan differs from frozen roster order")
    if {r["message_id"] for r in roster} != {r.message_id for r in dataset.requests}:
        raise ValueError("roster and public requests differ")
    prepared = {}
    prompts = {}
    token_receipts = {}
    by_request = {}
    for row in plan.rows:
        bundle = dataset.bundle(row.request.message_id)
        if row.request != bundle.request:
            raise ValueError("planned original request mismatch")
        by_request.setdefault(row.request.message_id, {})[row.condition] = row
    for message_id, rows in by_request.items():
        sealed = {}
        for condition, row in rows.items():
            if row.preparation_failure is not None:
                if row.preparation is not None:
                    bound(row.preparation)
                continue
            data = bound(row.preparation)
            item = SealedInput(row.preparation.path, data, row.preparation.sha256)
            packet = verify_input(
                item, dataset=dataset, message_id=message_id, root=root, binding=binding
            )
            if (condition == "baseline") != (packet.extraction_result is None):
                raise ValueError("preparation condition mismatch")
            if packet.extraction_artifact_path is not None:
                if (
                    plan.spending.extraction_version is None
                    or plan.spending.extraction_version_refs
                    != {"extractor_sha256": packet.extraction_result.extractor_sha256}
                ):
                    raise ValueError(
                        "assisted preparation requires registered extraction identity"
                    )
                if frozen.get(packet.extraction_artifact_path) != (
                    packet.extraction_artifact_bytes.encode("utf-8")
                ):
                    raise ValueError("shared extraction missing frozen bytes")
            prompt = build_prompt(packet, condition)
            messages = [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user.decode("utf-8")},
            ]
            if bound(row.prompt) != canonical_json(messages):
                raise ValueError("frozen rendered prompt mismatch")
            token = _canonical(bound(row.token_receipt), TokenReceipt)
            if (
                token.messages_sha256 != digest(canonical_json(messages))
                or token.template_sha256 != plan.model.template_sha256
                or token.model_receipt_sha256 != plan.model.receipt.sha256
            ):
                raise ValueError("token receipt identity mismatch")
            sealed[condition] = item
            prompts[row.row_id] = prompt
            token_receipts[row.row_id] = token
        b, ab = rows["B"], rows["A+B"]
        if (
            b.preparation != ab.preparation
            or b.preparation_failure != ab.preparation_failure
        ):
            raise ValueError("B and A+B must share exact preparation/failure")
        if b.preparation_failure is None:
            if prompts[b.row_id].common != prompts[ab.row_id].common:
                raise ValueError("assisted common prompt mismatch")
        prepared[message_id] = PreparedInputs(
            sealed.get("baseline"), sealed.get("B"), b.preparation_failure
        )
    return ValidatedPlan(
        root,
        plan,
        digest(raw),
        dataset,
        binding,
        prepared,
        prompts,
        token_receipts,
        frozen,
        raw,
        plan_path,
    )


def _recheck_bound_inputs(value: ValidatedPlan) -> None:
    """Authenticate the validated snapshot again; never trust its hash alone."""
    _safe_paths(value.root, value.plan, value.plan_path)
    raw = value.plan_path.read_bytes()
    if (
        raw != value.plan_bytes
        or digest(raw) != value.plan_sha256
        or canonical_json(value.plan.model_dump()) != raw
    ):
        raise ValueError("sealed plan identity changed after validation")
    for item in value.plan.frozen_files:
        if read_ref(value.root, item) != value.frozen_bytes.get(item.path):
            raise ValueError("frozen input changed after validation")


def _sealed_readiness(value: ValidatedPlan) -> tuple[ReadinessReceipt, bytes]:
    ready_path = artifact_path(value.root, value.plan.readiness_path)
    try:
        ready_raw = ready_path.read_bytes()
        ready = _canonical(ready_raw, ReadinessReceipt)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"missing/invalid sealed readiness boot receipt: {error}"
        ) from error
    if ready.plan_sha256 != value.plan_sha256:
        raise ValueError("stale readiness plan binding")
    try:
        ledger_raw = artifact_path(value.root, value.plan.ledger_path).read_bytes()
    except OSError as error:
        raise ValueError(
            "sealed preparation budget ledger is missing; cannot reinitialize"
        ) from error
    if digest(ledger_raw) != ready.ledger_sha256:
        raise ValueError(
            "budget ledger changed after readiness; explicit reconciliation required"
        )
    return ready, ready_raw


def _changed_boot(ready: ReadinessReceipt, boot: str | None) -> bool:
    if boot is None or boot == "":
        return False
    if type(boot) is not str:
        raise ValueError("invalid current OS boot identity")
    valid_boot_identity(boot)
    return ready.boot_identity != boot


def approve_execution(
    value: ValidatedPlan, boot: str | None, *, approved_by: str = "terminal_user"
) -> dict:
    """Persist explicit consent exclusively, then use the existing execution gate.

    Full load_and_validate is the caller's prerequisite. No approval is created
    during preparation, dry-run or an unchanged/missing/invalid current boot.
    If a post-write recheck fails, preserve the approval as immutable evidence;
    refusal is not permission to delete, replace or reuse a changed input binding.
    """
    _recheck_bound_inputs(value)
    ready, ready_raw = _sealed_readiness(value)
    if not _changed_boot(ready, boot):
        raise ValueError(
            "OS restart required: missing or unchanged current boot identity"
        )
    approval_path = artifact_path(value.root, value.plan.approval_path)
    output_path = artifact_path(value.root, value.plan.output_path)

    def fresh():
        if approval_path.exists() or output_path.exists():
            raise FileExistsError(
                "execution approval or run output already exists; no replay"
            )

    fresh()
    approval = ExecutionApproval(
        plan_sha256=value.plan_sha256,
        readiness_sha256=digest(ready_raw),
        approved_by=approved_by,
    )
    # Recheck immediately before and after the exclusive durable write. Never
    # overwrite history or roll back a consent record, even on a detected race.
    _recheck_bound_inputs(value)
    if _sealed_readiness(value)[1] != ready_raw:
        raise ValueError("readiness identity changed before approval write")
    fresh()
    write_new(approval_path, canonical_json(approval.model_dump()))
    _recheck_bound_inputs(value)
    if _sealed_readiness(value)[1] != ready_raw:
        raise ValueError("readiness identity changed after approval write")
    if output_path.exists():
        raise FileExistsError("run output already exists; no replay")
    return authorization(value, boot)


def authorization(value: ValidatedPlan, boot: str | None, *, dry_run=False) -> dict:
    required = {
        "status": "restart_and_approval_required",
        "plan_sha256": value.plan_sha256,
        "execution_authorized": False,
    }
    ready_path = artifact_path(value.root, value.plan.readiness_path)
    if dry_run and not ready_path.exists():
        if boot is not None and boot != "":
            if type(boot) is not str:
                raise ValueError("invalid current OS boot identity")
            valid_boot_identity(boot)
        return required
    ready, ready_raw = _sealed_readiness(value)
    changed = _changed_boot(ready, boot)
    if dry_run and not artifact_path(value.root, value.plan.approval_path).exists():
        return {
            **required,
            "status": "approval_required" if changed else required["status"],
        }
    try:
        approval = _canonical(
            artifact_path(value.root, value.plan.approval_path).read_bytes(),
            ExecutionApproval,
        )
    except (ValueError, OSError) as error:
        raise ValueError(f"missing/invalid execution approval: {error}") from error
    if approval.plan_sha256 != value.plan_sha256 or approval.readiness_sha256 != digest(
        ready_raw
    ):
        raise ValueError("stale execution approval binding")
    if not changed:
        if dry_run:
            return {**required, "status": "restart_required"}
        raise ValueError(
            "OS restart required: missing or unchanged current boot identity"
        )
    if dry_run:
        return {
            "status": "ready",
            "plan_sha256": value.plan_sha256,
            "execution_authorized": False,
        }
    return {"status": "authorized", "plan_sha256": value.plan_sha256}


def create_output(value: ValidatedPlan) -> Path:
    _safe_paths(value.root, value.plan, value.plan_path)
    output = artifact_path(value.root, value.plan.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    from messaging_nlp.run_storage import _sync_directory

    _sync_directory(output.parent)
    write_new(output / "plan.json", value.plan_bytes)
    for name, data in value.frozen_bytes.items():
        write_new(output / "frozen" / name, data)
    return output
