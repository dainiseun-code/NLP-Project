"""Offline bundle sealing and model-free reproduction (Stage 4B).

``build_bundle`` copies actual retained runner-output bytes and evaluator inputs
into one portable, hash-bound bundle and seals ``manifest.json``. ``reproduce``
is the pure offline consumer: it verifies the bundle, reconciles the plan
against the frozen roster/requirements/gold/policy, classifies every planned
row, scores successful rows structurally and semantically, and reports costs.
Neither function opens a model, calls a reviewer, imports an optional model
runtime, runs retrieval, or writes into the immutable bundle.

The scoring kernels live in ``messaging_nlp.evaluation`` (owned by another
writer). ``reproduce`` imports them lazily and only when a score is actually
needed, so bundle verification and row classification never require them. The
kernel return shapes that ``reproduce`` depends on are documented inline and in
``messaging_nlp.evaluation``:

* ``score_structure(row, requirement) -> {"action", "recipient", "selected_event", "event_start"}`` (each ``pass``/``fail``/``unmeasured``/``not_applicable``)
* ``score_extraction(text, gold, result) -> {"status": "measured"|"unmeasured", ...}``
* ``evidence_coverage(candidate_ids, alternatives) -> {"status": "measured"|"unmeasured", "complete", "best_fraction"}``
"""

from __future__ import annotations

import hashlib
import os
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from messaging_nlp.contracts import Entity
from messaging_nlp.encoder_cache import portable_path
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.preparation import digest
from messaging_nlp.reviews import JudgmentRecord, load_review_binding, resolve_reviews
from messaging_nlp.run_contracts import (
    ArtifactRef,
    PendingRowRecord,
    RunContract,
    RunPlan,
    load_row_receipt,
)
from messaging_nlp.run_storage import write_new

MANIFEST_SCHEMA_VERSION = "nlp3-reproduction-manifest-v1"
RESULT_SCHEMA_VERSION = "nlp3-reproduce-result-v1"
RUN_ROOT_NAME = "run"
CONDITIONS = ("baseline", "B", "A+B")

# Explicit, fixed destination names so reproduction is independent of the
# source evaluator-input filenames.
_EVALUATOR_NAMES = {
    "requirements": "requirements.jsonl",
    "roster": "roster.jsonl",
    "gold": "gold.jsonl",
    "policy": "policy.json",
    "preparation_costs": "preparation-costs.json",
}

# A live model directory is identified by retained weight payloads. The runner
# output freezes template/tokenizer/prompt bytes, never weights; refusing these
# suffixes enforces "never copy live model directories".
_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".h5",
    ".npz",
)


class ReproductionManifest(RunContract):
    """Strict versioned bundle manifest; every retained byte is inventoried."""

    schema_version: Literal["nlp3-reproduction-manifest-v1"] = MANIFEST_SCHEMA_VERSION
    files: list[ArtifactRef] = Field(min_length=1)
    plan: ArtifactRef
    run_root: Literal["run"] = RUN_ROOT_NAME
    requirements: ArtifactRef
    roster: ArtifactRef
    gold: ArtifactRef
    policy: ArtifactRef
    review_records: dict[str, list[JudgmentRecord]] = Field(default_factory=dict)
    adjudications: dict[str, ArtifactRef] = Field(default_factory=dict)
    preparation_costs: ArtifactRef | None = None

    @model_validator(mode="after")
    def unique_inventory_paths(self):
        paths = [artifact.path for artifact in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("bundle inventory repeats a path")
        return self


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(path: str, data: bytes) -> ArtifactRef:
    return ArtifactRef(path=path, sha256=_hash(data), bytes=len(data))


def _is_weight_file(relative: str) -> bool:
    return relative.endswith(_WEIGHT_SUFFIXES)


def _unsymlinked(path):
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("bundle paths must not traverse symlinks")
    return path.resolve()


def _copy_tree(source: Path, destination: Path) -> list[ArtifactRef]:
    """Copy every retained run byte; reject symlinks and live model payloads."""
    source = source.resolve(strict=True)
    references = []
    for directory, dirnames, filenames in os.walk(source):
        directory = Path(directory)
        for name in dirnames:
            if (directory / name).is_symlink():
                raise ValueError(
                    f"run root must not traverse symlinked directories: {name}"
                )
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                raise ValueError(f"run root must not contain symlinks: {path}")
            relative = path.relative_to(source).as_posix()
            if _is_weight_file(relative):
                raise ValueError(f"live model file must not be copied: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            write_new(target, data)
            references.append(_artifact(relative, data))
    return references


def _copy_one(source_path: Path, destination: Path) -> ArtifactRef:
    source_path = _unsymlinked(source_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"evaluator input must be a regular file: {source_path}")
    data = source_path.read_bytes()
    write_new(destination, data)
    return _artifact(f"evaluator/{destination.name}", data)


def _remap_run_reference(
    copied: dict[str, ArtifactRef], reference: ArtifactRef, label: str
) -> ArtifactRef:
    entry = copied.get(reference.path)
    if (
        entry is None
        or entry.sha256 != reference.sha256
        or entry.bytes != reference.bytes
    ):
        raise ValueError(
            f"{label} reference is not a copied run artifact: {reference.path}"
        )
    return ArtifactRef(
        path=f"{RUN_ROOT_NAME}/{reference.path}",
        sha256=reference.sha256,
        bytes=reference.bytes,
    )


def build_bundle(
    run_root,
    destination,
    *,
    requirements_path,
    roster_path,
    gold_path,
    policy_path,
    review_records=None,
    adjudications=None,
    preparation_costs_path=None,
) -> dict:
    """Seal a portable, hash-bound reproduction bundle and return its manifest.

    ``run_root`` is the durable runner output (contains ``plan.json``,
    ``frozen/``, ``rows/`` and ``status.json``), not the project root. The
    destination must be fresh and disjoint from the source, must not traverse
    symlinks, and must never copy a live model directory. ``review_records``
    maps ``row_id`` to a list of ``JudgmentRecord`` dicts whose references are
    run-root-relative; ``adjudications`` maps ``row_id`` to a run-root-relative
    ``ArtifactRef``. Both are persisted in the manifest with bundle-relative
    references. No packet is created and no model or reviewer is invoked.
    """
    run_root = _unsymlinked(run_root)
    if not run_root.is_dir():
        raise ValueError("run_root must be the durable runner output directory")
    if not (run_root / "plan.json").is_file() or not (run_root / "frozen").is_dir():
        raise ValueError("run_root must contain plan.json and frozen/")

    destination = _unsymlinked(destination)
    if destination.exists():
        raise FileExistsError("bundle destination must be fresh")
    if destination.is_relative_to(run_root) or run_root.is_relative_to(destination):
        raise ValueError("bundle destination must be disjoint from the run root")
    inputs = {
        name: _unsymlinked(path)
        for name, path in (
            ("requirements", requirements_path),
            ("roster", roster_path),
            ("gold", gold_path),
            ("policy", policy_path),
        )
    }
    if preparation_costs_path is not None:
        inputs["preparation_costs"] = _unsymlinked(preparation_costs_path)
    for name, path in inputs.items():
        if destination.is_relative_to(path) or path.is_relative_to(destination):
            raise ValueError(f"bundle destination overlaps evaluator input: {name}")

    destination.mkdir(parents=True, exist_ok=False)
    run_destination = destination / RUN_ROOT_NAME
    run_destination.mkdir()
    run_refs = _copy_tree(run_root, run_destination)
    copied = {reference.path: reference for reference in run_refs}
    if "plan.json" not in copied:
        raise ValueError("run root lost its plan.json during copy")

    evaluator_destination = destination / "evaluator"
    evaluator_destination.mkdir()
    evaluator_refs = {
        name: _copy_one(inputs[name], evaluator_destination / _EVALUATOR_NAMES[name])
        for name in ("requirements", "roster", "gold", "policy")
    }
    preparation_costs = None
    if "preparation_costs" in inputs:
        preparation_costs = _copy_one(
            inputs["preparation_costs"],
            evaluator_destination / _EVALUATOR_NAMES["preparation_costs"],
        )

    remapped_records: dict[str, list[JudgmentRecord]] = {}
    for row_id, records in (review_records or {}).items():
        validated = []
        for record in records:
            judgment = JudgmentRecord.model_validate(record)
            validated.append(
                JudgmentRecord(
                    binding=_remap_run_reference(copied, judgment.binding, "binding"),
                    response=_remap_run_reference(
                        copied, judgment.response, "response"
                    ),
                    identity=_remap_run_reference(
                        copied, judgment.identity, "identity"
                    ),
                )
            )
        remapped_records[row_id] = validated

    remapped_adjudications: dict[str, ArtifactRef] = {}
    for row_id, reference in (adjudications or {}).items():
        ref = ArtifactRef.model_validate(reference)
        remapped_adjudications[row_id] = _remap_run_reference(
            copied, ref, "adjudication"
        )

    files = [
        ArtifactRef(
            path=f"{RUN_ROOT_NAME}/{reference.path}",
            sha256=reference.sha256,
            bytes=reference.bytes,
        )
        for reference in run_refs
    ] + list(evaluator_refs.values())
    if preparation_costs is not None:
        files.append(preparation_costs)
    files.sort(key=lambda artifact: artifact.path)
    by_path = {artifact.path: artifact for artifact in files}

    manifest = ReproductionManifest(
        files=files,
        plan=by_path[f"{RUN_ROOT_NAME}/plan.json"],
        run_root=RUN_ROOT_NAME,
        requirements=by_path[f"evaluator/{_EVALUATOR_NAMES['requirements']}"],
        roster=by_path[f"evaluator/{_EVALUATOR_NAMES['roster']}"],
        gold=by_path[f"evaluator/{_EVALUATOR_NAMES['gold']}"],
        policy=by_path[f"evaluator/{_EVALUATOR_NAMES['policy']}"],
        review_records=remapped_records,
        adjudications=remapped_adjudications,
        preparation_costs=preparation_costs,
    )
    data = canonical_json(manifest.model_dump())
    write_new(destination / "manifest.json", data)
    return manifest.model_dump()


def _no_symlink_chain(root: Path, relative: str) -> Path:
    portable_path(relative)
    root = root.resolve(strict=True)
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"bundle path traverses a symlink: {relative}")
    resolved = current.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"bundle path escapes the bundle root: {relative}")
    return resolved


def _read_bundle(bundle_root: Path, relative: str) -> bytes:
    return _no_symlink_chain(bundle_root, relative).read_bytes()


def _indexed(
    inventory: dict[str, ArtifactRef], reference: ArtifactRef, label: str
) -> None:
    entry = inventory.get(reference.path)
    if entry is None or entry != reference:
        raise ValueError(
            f"{label} indexed reference disagrees with inventory: {reference.path}"
        )


def _parse_jsonl(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rows = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {index}")
        value = loads_json(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row at line {index} is not an object")
        rows.append(value)
    return rows


def _run_relative(reference: ArtifactRef, label: str) -> ArtifactRef:
    prefix = f"{RUN_ROOT_NAME}/"
    if not reference.path.startswith(prefix):
        raise ValueError(f"{label} reference must live under the retained run root")
    return ArtifactRef(
        path=reference.path[len(prefix) :],
        sha256=reference.sha256,
        bytes=reference.bytes,
    )


def _structural_outcome(structural) -> str:
    """Collapse score_structure's four dimensions to one verdict.

    ``score_structure`` returns ``{"action", "recipient", "selected_event",
    "event_start"}`` where each field is ``pass``/``fail``/``unmeasured``/
    ``not_applicable``. A failure anywhere fails; otherwise any unmeasured
    applicable dimension leaves the row unmeasured; otherwise all applicable
    dimensions passed.
    """
    if not isinstance(structural, dict):
        return "unmeasured"
    values = [
        structural.get(field)
        for field in ("action", "recipient", "selected_event", "event_start")
    ]
    if "fail" in values:
        return "fail"
    if "unmeasured" in values:
        return "unmeasured"
    return "pass"


def _semantic_outcome(semantic) -> str:
    status = semantic.get("status") if isinstance(semantic, dict) else None
    return status if status in ("pass", "fail") else "unmeasured"


def _crosscheck_binding_inputs(inputs, requirement, policy, row, envelope, preparation):
    """Fail closed unless every bound review input matches the retained bytes.

    ``inputs`` is the dict returned by ``load_review_binding``. A supplied but
    stale or foreign binding must raise, never silently pass.
    """
    if inputs["requirements"] != requirement:
        raise ValueError("review binding requirements differ from retained requirement")
    if inputs["policy"] != policy:
        raise ValueError("review binding policy differs from retained policy")
    if inputs["raw"] != row.raw_output:
        raise ValueError("review binding raw output differs from retained row")
    parsed = envelope.model_dump() if envelope is not None else None
    if inputs["parsed"] != parsed:
        raise ValueError("review binding parsed output differs from retained envelope")
    if inputs["request"] != row.request.model_dump():
        raise ValueError("review binding request differs from retained row")
    expected_preparation = preparation.model_dump() if preparation is not None else None
    if inputs["preparation"] != expected_preparation:
        raise ValueError("review binding preparation/context differs from frozen plan")


def reproduce(manifest_path) -> dict:
    """Pure offline consumer of a sealed bundle; never writes or calls a model.

    Validation and reconciliation failures raise (fail closed); a valid bundle
    returns a JSON-serializable dict with per-row disposition/success, separate
    structural and semantic dimensions, per-request extraction/coverage and
    per-condition costs.
    """
    from messaging_nlp.application_types import PreparationPacket

    manifest_path = Path(manifest_path).resolve(strict=True)
    bundle_root = manifest_path.parent
    raw = manifest_path.read_bytes()
    manifest = ReproductionManifest.model_validate(loads_json(raw.decode("utf-8")))
    if canonical_json(manifest.model_dump()) != raw:
        raise ValueError("reproduction manifest must be canonical")
    if manifest.run_root != RUN_ROOT_NAME:
        raise ValueError("unsupported bundle run root")

    inventory = {artifact.path: artifact for artifact in manifest.files}
    actual_paths = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("bundle contains an unindexed symlink")
        if path.is_file() and path != manifest_path:
            actual_paths.add(path.relative_to(bundle_root).as_posix())
    if actual_paths != set(inventory):
        raise ValueError("bundle inventory differs from actual retained files")
    if manifest.plan.path != "run/plan.json":
        raise ValueError("bundle plan must be the retained run/plan.json")
    for artifact in manifest.files:
        data = _read_bundle(bundle_root, artifact.path)
        if len(data) != artifact.bytes or _hash(data) != artifact.sha256:
            raise ValueError(f"inventory hash mismatch: {artifact.path}")
    for label, reference in (
        ("plan", manifest.plan),
        ("requirements", manifest.requirements),
        ("roster", manifest.roster),
        ("gold", manifest.gold),
        ("policy", manifest.policy),
    ):
        _indexed(inventory, reference, label)
    if manifest.preparation_costs is not None:
        _indexed(inventory, manifest.preparation_costs, "preparation_costs")
    for row_id, records in manifest.review_records.items():
        for record in records:
            for label, reference in (
                ("binding", record.binding),
                ("response", record.response),
                ("identity", record.identity),
            ):
                _indexed(inventory, reference, f"review_records[{row_id}].{label}")
    for row_id, reference in manifest.adjudications.items():
        _indexed(inventory, reference, f"adjudications[{row_id}]")

    run_dir = bundle_root / manifest.run_root
    plan_raw = _read_bundle(bundle_root, manifest.plan.path)
    plan = RunPlan.model_validate(loads_json(plan_raw.decode("utf-8")))
    if canonical_json(plan.model_dump()) != plan_raw:
        raise ValueError("retained plan must be canonical")
    row_ids = {row.row_id for row in plan.rows}
    if (
        not set(manifest.review_records) <= row_ids
        or not set(manifest.adjudications) <= row_ids
    ):
        raise ValueError("review/adjudication references a foreign planned row")

    roster = _parse_jsonl(_read_bundle(bundle_root, manifest.roster.path))
    roster_by_message = {}
    expected_order = []
    for row in roster:
        if "message_id" not in row:
            raise ValueError("roster row missing message_id")
        message_id = row["message_id"]
        order = row.get("condition_order")
        if not isinstance(order, list) or sorted(order) != ["A+B", "B", "baseline"]:
            raise ValueError("invalid roster condition order")
        expected_order.extend((message_id, condition) for condition in order)
        roster_by_message[message_id] = row
    if expected_order != [
        (planned.request.message_id, planned.condition) for planned in plan.rows
    ]:
        raise ValueError("plan differs from frozen roster order")
    if {row["message_id"] for row in roster} != {
        planned.request.message_id for planned in plan.rows
    }:
        raise ValueError("roster and plan message set differ")

    requirements = _parse_jsonl(_read_bundle(bundle_root, manifest.requirements.path))
    requirements_by_message = {}
    for requirement in requirements:
        if "message_id" not in requirement:
            raise ValueError("requirement row missing message_id")
        if requirement["message_id"] in requirements_by_message:
            raise ValueError("duplicate requirement message")
        requirements_by_message[requirement["message_id"]] = requirement
    if set(requirements_by_message) != {
        planned.request.message_id for planned in plan.rows
    }:
        raise ValueError("requirements and plan message set differ")
    for planned in plan.rows:
        message_id = planned.request.message_id
        requirement = requirements_by_message[message_id]
        roster_row = roster_by_message[message_id]
        if requirement.get("request_text_sha256") != digest(
            planned.request.text.encode("utf-8")
        ):
            raise ValueError("requirement request hash differs from planned request")
        if (
            requirement.get("context_snapshot_id")
            != planned.request.context_snapshot_id
        ):
            raise ValueError("requirement snapshot differs from planned request")
        if roster_row.get("text_sha256") != digest(
            planned.request.text.encode("utf-8")
        ):
            raise ValueError("roster text hash differs from planned request")
        for field in ("scenario_id", "family_id", "variant_type", "category_tags"):
            if requirement.get(field) != roster_row.get(field):
                raise ValueError(f"roster/requirement {field} mismatch: {message_id}")

    gold_rows = _parse_jsonl(_read_bundle(bundle_root, manifest.gold.path))
    gold_by_message: dict[str, list[Entity]] = {}
    for row in gold_rows:
        if "message_id" not in row:
            raise ValueError("gold row missing message_id")
        if row["message_id"] not in requirements_by_message:
            raise ValueError("gold references a foreign request")
        gold_by_message.setdefault(row["message_id"], []).append(
            Entity(
                type=row["type"],
                text=row["text"],
                start=row["start"],
                end=row["end"],
            )
        )

    policy = loads_json(_read_bundle(bundle_root, manifest.policy.path).decode("utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("policy must be a JSON object")

    from messaging_nlp.evaluation import normalize_human_review, paired_summary

    normalized_human_review = {}
    for message_id, requirement in requirements_by_message.items():
        statuses = [
            requirement["human_review_status"],
            roster_by_message[message_id]["human_review_status"],
        ]
        normalized_human_review[message_id] = [
            normalize_human_review(s) for s in statuses
        ]
    frozen_identities = {(item.sha256, item.bytes) for item in plan.frozen_files}
    if (manifest.roster.sha256, manifest.roster.bytes) != (
        plan.roster.sha256,
        plan.roster.bytes,
    ):
        raise ValueError("evaluator roster differs from execution-frozen roster")
    for reference in (manifest.requirements, manifest.gold, manifest.policy):
        if (reference.sha256, reference.bytes) not in frozen_identities:
            raise ValueError("evaluation inputs were not frozen before execution")

    for item in plan.frozen_files:
        reference = inventory.get(f"{RUN_ROOT_NAME}/frozen/{item.path}")
        if (
            reference is None
            or reference.sha256 != item.sha256
            or reference.bytes != item.bytes
        ):
            raise ValueError(f"frozen file missing or mismatched: {item.path}")
    # Retained source is executable provenance, not decorative bytes. Refuse a
    # different installed scorer instead of silently calculating new results.
    package = Path(str(files("messaging_nlp")))
    installed_sources = {
        f"src/messaging_nlp/{p.relative_to(package).as_posix()}": digest(p.read_bytes())
        for p in package.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    frozen_sources = {
        item.path: item.sha256 for item in plan.frozen_files if item.role == "source"
    }
    if installed_sources != frozen_sources:
        raise ValueError(
            "installed scoring source differs from retained source; use frozen source bytes"
        )

    # Detect duplicate or foreign receipts (paths outside the planned ordinals).
    planned_ordinals = {planned.ordinal for planned in plan.rows}
    for path in inventory:
        if not path.startswith(f"{RUN_ROOT_NAME}/rows/") or not path.endswith(
            "/receipt.json"
        ):
            continue
        stem = path[len(f"{RUN_ROOT_NAME}/rows/") : -len("/receipt.json")]
        if "/" in stem or len(stem) != 6 or not stem.isdigit():
            raise ValueError(f"foreign receipt path: {path}")
        if int(stem) not in planned_ordinals:
            raise ValueError(f"foreign receipt ordinal: {path}")

    status_document = None
    if f"{RUN_ROOT_NAME}/status.json" in inventory:
        status_document = loads_json(
            _read_bundle(bundle_root, f"{RUN_ROOT_NAME}/status.json").decode("utf-8")
        )
        if not isinstance(status_document, dict):
            raise ValueError("status.json must be a JSON object")
    pending_by_row = {}
    if status_document is not None and status_document.get("pending_rows"):
        for record in status_document["pending_rows"]:
            pending = PendingRowRecord.model_validate(record)
            if pending.row_id in pending_by_row or pending.ordinal >= len(plan.rows):
                raise ValueError("duplicate or foreign pending row")
            planned = plan.rows[pending.ordinal]
            if (pending.row_id, pending.condition, pending.message_id) != (
                planned.row_id,
                planned.condition,
                planned.request.message_id,
            ):
                raise ValueError("pending row record disagrees with planned row")
            for artifact in pending.artifacts.values():
                _indexed(
                    inventory,
                    ArtifactRef(
                        path=f"run/{artifact.path}",
                        sha256=artifact.sha256,
                        bytes=artifact.bytes,
                    ),
                    "pending artifact",
                )
            pending_by_row[pending.row_id] = pending

    by_message = {}
    for planned in plan.rows:
        by_message.setdefault(planned.request.message_id, {})[planned.condition] = (
            planned
        )
    for rows in by_message.values():
        b, ab = rows["B"], rows["A+B"]
        if (
            b.preparation != ab.preparation
            or b.preparation_failure != ab.preparation_failure
        ):
            raise ValueError("assisted conditions do not share exact preparation")

    def assisted_preparation(message_id):
        rows = by_message[message_id]
        assisted = rows.get("B")
        if assisted is None or assisted.preparation is None:
            return None
        return PreparationPacket.model_validate(
            loads_json(
                _read_bundle(
                    bundle_root,
                    f"{RUN_ROOT_NAME}/frozen/{assisted.preparation.path}",
                ).decode("utf-8")
            )
        )

    def row_preparation(planned):
        """The exact frozen preparation packet bound to one planned row."""
        if planned.preparation is None:
            return None
        return PreparationPacket.model_validate(
            loads_json(
                _read_bundle(
                    bundle_root,
                    f"{RUN_ROOT_NAME}/frozen/{planned.preparation.path}",
                ).decode("utf-8")
            )
        )

    # Lazy kernel import: extraction/coverage and structural scoring are the only
    # consumers of the evaluation kernels.
    kernels = None

    def evaluation():
        nonlocal kernels
        if kernels is None:
            from messaging_nlp.evaluation import (  # noqa: PLC0415
                evidence_coverage,
                score_extraction,
                score_structure,
            )

            kernels = (score_structure, score_extraction, evidence_coverage)
        return kernels

    extraction_results = {}
    coverage_results = {}
    for message_id, rows in by_message.items():
        requirement = requirements_by_message[message_id]
        request = next(iter(rows.values())).request
        gold = gold_by_message.get(message_id, [])
        preparation = assisted_preparation(message_id)
        _, score_extraction, evidence_coverage = evaluation()
        alternatives = requirement.get("support_alternatives", [])
        extraction_results[message_id] = score_extraction(
            request.text, gold, preparation.extraction_result if preparation else None
        )
        coverage_results[message_id] = {}
        for condition, planned in rows.items():
            packet = row_preparation(planned)
            coverage_results[message_id][condition] = evidence_coverage(
                packet.search.candidates.source_record_ids if packet else None,
                alternatives,
            )

    row_results = []
    costs = {
        condition: {
            "elapsed_seconds": 0.0,
            "charged_seconds": 0,
            "charged_generations": 0,
            "completed_rows": 0,
            "attempted_rows": 0,
            "provider_calls": 0,
            "generation_started_rows": 0,
            "generation_completed_rows": 0,
        }
        for condition in CONDITIONS
    }
    for planned in plan.rows:
        message_id = planned.request.message_id
        roster_row = roster_by_message[message_id]
        requirement = requirements_by_message[message_id]
        base = {
            "row_id": planned.row_id,
            "ordinal": planned.ordinal,
            "condition": planned.condition,
            "message_id": message_id,
            "scenario_id": roster_row.get("scenario_id"),
            "family_id": roster_row.get("family_id"),
            "variant_type": roster_row.get("variant_type"),
            "category_tags": roster_row.get("category_tags", []),
        }
        receipt_relative = f"{RUN_ROOT_NAME}/rows/{planned.ordinal:06d}/receipt.json"
        condition_costs = costs[planned.condition]
        has_review = bool(manifest.review_records.get(planned.row_id))
        has_adjudication = planned.row_id in manifest.adjudications
        if has_adjudication and not has_review:
            raise ValueError("adjudication cannot replace missing original reviews")
        if receipt_relative not in inventory or planned.row_id in pending_by_row:
            if has_review or has_adjudication:
                raise ValueError(
                    "pending/missing execution cannot carry outcome reviews"
                )
            if planned.row_id in pending_by_row:
                pending = pending_by_row[planned.row_id]
                if (
                    pending.ordinal != planned.ordinal
                    or pending.condition != planned.condition
                    or pending.message_id != message_id
                ):
                    raise ValueError("pending row record disagrees with planned row")
                row_results.append(
                    {
                        **base,
                        "disposition": "pending_persistence",
                        "technical_ok": False,
                        "success": None,
                        "structural": None,
                        "semantic": None,
                    }
                )
            else:
                row_results.append(
                    {
                        **base,
                        "disposition": "missing_receipt",
                        "technical_ok": False,
                        "success": None,
                        "structural": None,
                        "semantic": None,
                    }
                )
            continue
        receipt, row, envelope = load_row_receipt(
            run_dir, bundle_root / receipt_relative
        )
        if (
            receipt.run_id != plan.run_id
            or receipt.ordinal != planned.ordinal
            or receipt.row_id != planned.row_id
            or receipt.condition != planned.condition
            or receipt.message_id != message_id
            or receipt.plan_sha256 != digest(plan_raw)
        ):
            raise ValueError("receipt disagrees with the top-level plan")
        # One observed provider call at most per row; invocation does not prove
        # dispatch/model start. Keep all three observations separately visible.
        condition_costs["attempted_rows"] += int(receipt.provider_calls > 0)
        condition_costs["provider_calls"] += receipt.provider_calls
        condition_costs["generation_started_rows"] += int(receipt.generation_started)
        condition_costs["generation_completed_rows"] += int(
            receipt.generation_completed
        )
        if receipt.generation_completed and (
            receipt.generation_tokens is None
            or not 1 <= receipt.generation_tokens <= plan.decoding.max_output_tokens
            or receipt.input_tokens is None
            or not 1 <= receipt.input_tokens <= plan.decoding.max_input_tokens
        ):
            raise ValueError("completed execution has invalid token evidence")
        if receipt.disposition == "completed":
            condition_costs["completed_rows"] += 1
            score_structure, _, _ = evaluation()
            structural = score_structure(row, requirement)
            records = manifest.review_records.get(planned.row_id, [])
            adjudication = manifest.adjudications.get(planned.row_id)
            if not records:
                semantic = {
                    "status": "unmeasured",
                    "dimensions": {},
                    "reason": "missing_review",
                }
            else:
                run_records = [
                    JudgmentRecord(
                        binding=_run_relative(record.binding, "binding"),
                        response=_run_relative(record.response, "response"),
                        identity=_run_relative(record.identity, "identity"),
                    )
                    for record in records
                ]
                binding_path = bundle_root / records[0].binding.path
                review_binding, _, inputs = load_review_binding(run_dir, binding_path)
                if (
                    review_binding.row_id != planned.row_id
                    or review_binding.plan_sha256 != digest(plan_raw)
                ):
                    raise ValueError("review binding belongs to another planned row")
                preparation = row_preparation(planned)
                _crosscheck_binding_inputs(
                    inputs, requirement, policy, row, envelope, preparation
                )
                run_adjudication = (
                    _run_relative(adjudication, "adjudication")
                    if adjudication is not None
                    else None
                )
                semantic = resolve_reviews(
                    run_dir, binding_path, run_records, run_adjudication
                )
            structural_status = _structural_outcome(structural)
            semantic_status = _semantic_outcome(semantic)
            if structural_status == "fail" or semantic_status == "fail":
                success = False
            elif structural_status == "pass" and semantic_status == "pass":
                success = True
            else:
                success = None
            row_results.append(
                {
                    **base,
                    "disposition": "completed",
                    "technical_ok": True,
                    "success": success,
                    "structural": structural,
                    "semantic": semantic,
                }
            )
        elif receipt.disposition == "attempted_failure":
            if has_review or has_adjudication:
                raise ValueError("technical failure cannot carry outcome reviews")
            row_results.append(
                {
                    **base,
                    "disposition": "attempted_failure",
                    "technical_ok": False,
                    "success": False,
                    "structural": None,
                    "semantic": None,
                }
            )
        else:
            if has_review or has_adjudication:
                raise ValueError("unexecuted work cannot carry outcome reviews")
            row_results.append(
                {
                    **base,
                    "disposition": receipt.disposition,
                    "technical_ok": False,
                    "success": None,
                    "structural": None,
                    "semantic": None,
                }
            )
        condition_costs["elapsed_seconds"] += receipt.elapsed_seconds
        condition_costs["charged_seconds"] += receipt.charged_seconds
        condition_costs["charged_generations"] += receipt.charged_generations

    startup_overhead = None
    if f"{RUN_ROOT_NAME}/budget-final.json" in inventory:
        budget = loads_json(
            _read_bundle(bundle_root, f"{RUN_ROOT_NAME}/budget-final.json").decode(
                "utf-8"
            )
        )
        reservations = budget.get("reservations") if isinstance(budget, dict) else None
        if not isinstance(reservations, list) or any(
            not isinstance(r, dict) or not isinstance(r.get("refs"), dict)
            for r in reservations
        ):
            raise ValueError("invalid retained budget reservation/refs")
        if isinstance(reservations, list):
            startup_overhead = [
                {
                    key: reservation[key]
                    for key in (
                        "reservation_id",
                        "kind",
                        "status",
                        "planned_seconds",
                        "actual_seconds",
                        "planned_generations",
                        "actual_generations",
                    )
                    if key in reservation
                }
                for reservation in reservations
                if reservation.get("kind") in ("startup", "overhead")
                and reservation.get("refs", {}).get("run_id") == plan.run_id
            ]

    preparation_costs = None
    if manifest.preparation_costs is not None:
        preparation_costs = loads_json(
            _read_bundle(bundle_root, manifest.preparation_costs.path).decode("utf-8")
        )

    statistics = policy["statistics"]
    settings = dict(repetitions=statistics["repetitions"], seed=statistics["seed"])
    summaries = {
        "overall": paired_summary(row_results, **settings),
        "variants": {
            variant: paired_summary(
                [r for r in row_results if r["variant_type"] == variant], **settings
            )
            for variant in sorted({r["variant_type"] for r in row_results})
        },
        "categories": {
            category: paired_summary(
                [r for r in row_results if category in r["category_tags"]], **settings
            )
            for category in sorted(
                {tag for r in row_results for tag in r["category_tags"]}
            )
        },
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "reproduced",
        "plan": {
            "run_id": plan.run_id,
            "plan_sha256": digest(plan_raw),
            "planned_rows": len(plan.rows),
        },
        "rows": row_results,
        "extraction": extraction_results,
        "coverage": coverage_results,
        "summaries": summaries,
        "human_review_status": normalized_human_review,
        "costs": {
            "qwen_by_condition": costs,
            "preparation": preparation_costs,
            "startup_overhead": startup_overhead,
            "unattributed_row_ids": [
                r["row_id"]
                for r in row_results
                if r["disposition"] in ("pending_persistence", "missing_receipt")
            ],
            "interpretation": "row totals cover committed receipts only; unresolved costs remain in retained budget evidence; preprocessing is separate, not live end-to-end latency",
        },
    }
