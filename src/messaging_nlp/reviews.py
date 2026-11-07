"""Blinded bytes and evidence-bound review ingestion; never calls a judge."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from messaging_nlp.application_types import PreparationPacket
from messaging_nlp.contracts import Digest, NonBlank
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.preparation import digest
from messaging_nlp.prompts import _AliasIndex, _contact, _event, _message
from messaging_nlp.run_contracts import ArtifactRef, RunContract, load_row_receipt
from messaging_nlp.run_storage import read_ref, ref, write_new


class ReviewBinding(RunContract):
    schema_version: Literal["nlp3-review-binding-v1"] = "nlp3-review-binding-v1"
    row_id: NonBlank
    plan_sha256: Digest
    packet: ArtifactRef
    inputs: dict[str, ArtifactRef]


class DimensionVerdict(RunContract):
    verdict: Literal["pass", "fail", "unresolved"]
    quotes: list[NonBlank]
    omission: NonBlank | None
    source_ids: list[NonBlank]
    rationale: NonBlank

    @model_validator(mode="after")
    def evidenced(self):
        if not self.quotes and self.omission is None:
            raise ValueError("verdict needs an exact quote or identified omission")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("duplicate cited source")
        return self


class Judgment(RunContract):
    schema_version: Literal["nlp3-judgment-v1"] = "nlp3-judgment-v1"
    judge_id: NonBlank
    packet_sha256: Digest
    inputs_sha256: Digest
    dimensions: dict[str, DimensionVerdict]


class JudgmentRecord(RunContract):
    """Private wrapper avoids a response/identity-receipt hash cycle."""

    binding: ArtifactRef
    response: ArtifactRef
    identity: ArtifactRef


class Adjudication(RunContract):
    schema_version: Literal["nlp3-adjudication-v1"] = "nlp3-adjudication-v1"
    packet_sha256: Digest
    original_judgments: list[Digest] = Field(min_length=2, max_length=2)
    adjudicator: Literal["nlp_orchestrator"]
    reason: NonBlank
    dimensions: dict[str, DimensionVerdict]


def _json(root, reference):
    return loads_json(read_ref(root, reference).decode("utf-8"))


def input_digest(binding: ReviewBinding) -> str:
    return digest(
        canonical_json({k: v.model_dump() for k, v in binding.inputs.items()})
    )


def _alias_value(value, aliases):
    """Alias structured references only; original prose must never be rewritten."""
    if isinstance(value, list):
        return [_alias_value(v, aliases) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    list_kinds = {
        "recipient_contact_ids": "contact",
        "candidate_contact_ids": "contact",
        "source_record_ids": "record",
    }
    one_kinds = {
        "event_id": "event",
        "selected_event_id": "event",
        "contact_id": "contact",
    }
    for key, item in value.items():
        if key in list_kinds:
            result[key] = [aliases.one(list_kinds[key], x) for x in item]
        elif key in one_kinds:
            result[key] = None if item is None else aliases.one(one_kinds[key], item)
        elif key == "anchor_id":
            result[key] = None if item is None else aliases.one("anchor", item)
        else:
            result[key] = _alias_value(item, aliases)
    return result


def _visible(inputs, packet_id):
    packet = PreparationPacket.model_validate(inputs["preparation"])
    aliases = _AliasIndex(packet.aliases)
    requirement = inputs["requirements"]
    context = packet.context
    expected = {
        key: requirement[key]
        for key in (
            "expected_action",
            "recipient_contact_ids",
            "required_meaning",
            "forbidden_meaning",
            "support_alternatives",
            "actual_blockers",
            "normalized_facts",
            "clarification_slot",
            "unsupported_reason",
        )
    }
    expected["support_alternatives"] = [
        [aliases.one("record", rid) for rid in alternative]
        for alternative in expected["support_alternatives"]
    ]
    return {
        "packet_id": packet_id,
        "request": {
            "text": packet.request.text,
            "owner": aliases.one("owner", packet.request.owner_id),
            "request_time": packet.request.request_time,
            "time_zone": packet.request.time_zone,
        },
        "eligible_context": {
            "contacts": [_contact(r, aliases) for r in context.contacts],
            "calendar": [_event(r, aliases) for r in context.calendar],
            "history": [_message(r, aliases) for r in context.history],
        },
        "requirements": _alias_value(expected, aliases),
        "rubric": {
            key: inputs["policy"][key]
            for key in ("dimensions", "evidence_rule", "disagreement")
        },
        "output": _alias_value(inputs["parsed"], aliases),
    }


def create_review_packet(run_root, receipt_path, requirement, policy, destination):
    """Only packet.json may be shown to a fresh isolated outcome judge."""
    root, destination = Path(run_root).resolve(), Path(destination).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("review destination must remain in retained run root")
    if destination.exists():
        raise FileExistsError("review destination already exists")
    receipt, row, envelope = load_row_receipt(root, receipt_path)
    if row.status != "ok":
        raise ValueError("no fabricated review packet for technical failure")
    if (
        requirement["message_id"] != row.request.message_id
        or requirement["request_text_sha256"] != digest(row.request.text.encode())
        or requirement["context_snapshot_id"] != row.request.context_snapshot_id
    ):
        raise ValueError("requirement differs from exact request")
    prep_ref = receipt.bindings.get("preparation")
    if prep_ref is None:
        raise ValueError("review requires actual retained full preparation/context")
    prep = _json(root, prep_ref)
    if (
        prep["request"] != row.request.model_dump()
        or digest(canonical_json(prep["context"])) != row.context_sha256
    ):
        raise ValueError("review context/request binding mismatch")
    inputs = {
        "request": row.request.model_dump(),
        "context": prep["context"],
        "preparation": prep,
        "requirements": requirement,
        "policy": policy,
        "raw": row.raw_output,
        "parsed": envelope.model_dump(),
    }
    visible = _visible(inputs, uuid.uuid4().hex)
    destination.mkdir(parents=True, exist_ok=False)
    references = {}
    for key, value in inputs.items():
        path = destination / "private" / f"{key}.json"
        write_new(path, canonical_json(value))
        references[key] = ref(root, path)
    path = destination / "packet.json"
    write_new(path, canonical_json(visible))
    binding = ReviewBinding(
        row_id=receipt.row_id,
        plan_sha256=receipt.plan_sha256,
        packet=ref(root, path),
        inputs=references,
    )
    write_new(destination / "binding.json", canonical_json(binding.model_dump()))
    return binding


def load_review_binding(root, path):
    root, path = Path(root).resolve(), Path(path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("review binding leaves retained run root")
    raw = path.read_bytes()
    binding = ReviewBinding.model_validate(loads_json(raw.decode()))
    if canonical_json(binding.model_dump()) != raw:
        raise ValueError("noncanonical review binding")
    if set(binding.inputs) != {
        "request",
        "context",
        "preparation",
        "requirements",
        "policy",
        "raw",
        "parsed",
    }:
        raise ValueError("incomplete review byte bindings")
    inputs = {key: _json(root, value) for key, value in binding.inputs.items()}
    packet = _json(root, binding.packet)
    if (
        inputs["request"] != inputs["preparation"]["request"]
        or inputs["context"] != inputs["preparation"]["context"]
        or _visible(inputs, packet["packet_id"]) != packet
    ):
        raise ValueError("blinded packet differs from bound source bytes")
    return binding, packet, inputs


def judge_assignment(root, binding_path, judge_id):
    """Safe dispatch bytes; creates no session and makes no review call.

    Stage7 must send ONLY these bytes into a fresh isolated session. Supply
    opaque hashes because a judge cannot inspect private binding files.
    """
    binding, packet, inputs = load_review_binding(root, binding_path)
    if judge_id not in inputs["policy"]["judges"]:
        raise ValueError("judge is outside frozen review policy")
    return {
        "packet": packet,
        "response_bindings": {
            "schema_version": "nlp3-judgment-v1",
            "judge_id": judge_id,
            "packet_sha256": binding.packet.sha256,
            "inputs_sha256": input_digest(binding),
        },
        "response_schema": Judgment.model_json_schema(),
        "instruction": "Return one JSON judgment using these response bindings. "
        "Independently assess every rubric dimension against this packet only.",
    }


def _validate_verdicts(dimensions, packet, inputs):
    if set(dimensions) != set(inputs["policy"]["dimensions"]):
        raise ValueError("review must cover every frozen dimension exactly")
    sources = {
        row["record"] for rows in packet["eligible_context"].values() for row in rows
    }
    output = canonical_json(packet["output"]).decode()
    raw = inputs["raw"]
    for verdict in dimensions.values():
        if any(q not in output and q not in raw for q in verdict.quotes):
            raise ValueError("review quote absent from exact candidate output")
        if not set(verdict.source_ids) <= sources:
            raise ValueError("review cites ineligible/unaliased source")


def validate_judgment(root, record: JudgmentRecord):
    binding, packet, inputs = load_review_binding(
        root, Path(root) / record.binding.path
    )
    # Authenticate wrapper bytes too, not merely their parsed content.
    read_ref(root, record.binding)
    raw = read_ref(root, record.response)
    judgment = Judgment.model_validate(loads_json(raw.decode()))
    native = _json(root, record.identity)
    expected = inputs["policy"]["judges"].get(judgment.judge_id)
    if (
        expected is None
        or native.get("model") != expected
        or native.get("finish") != "stop"
        or not native.get("session_id")
        or not native.get("message_id")
        or native.get("response_sha256") != digest(raw)
    ):
        raise ValueError("missing/unavailable/mismatched actual judge identity")
    if (
        judgment.packet_sha256 != binding.packet.sha256
        or judgment.inputs_sha256 != input_digest(binding)
    ):
        raise ValueError("stale judgment content binding")
    _validate_verdicts(judgment.dimensions, packet, inputs)
    return judgment


def resolve_reviews(root, binding_path, records, adjudication=None):
    """Missing review is unmeasured; stale supplied evidence raises, never passes."""
    binding, packet, inputs = load_review_binding(root, binding_path)
    wanted = set(inputs["policy"]["judges"])
    judgments = {}
    response_hashes = []
    sessions = set()
    for record in records:
        if record.binding.sha256 != digest(Path(binding_path).read_bytes()):
            raise ValueError("judgment record belongs to another packet")
        judgment = validate_judgment(root, record)
        native = _json(root, record.identity)
        if judgment.judge_id in judgments or native["session_id"] in sessions:
            raise ValueError("duplicate judge/session is not independent review")
        judgments[judgment.judge_id] = judgment
        sessions.add(native["session_id"])
        response_hashes.append(record.response.sha256)
    if set(judgments) != wanted:
        if adjudication is not None:
            raise ValueError("adjudication cannot replace missing original judges")
        return {"status": "unmeasured", "dimensions": {}, "reason": "missing_review"}
    resolved = {}
    if adjudication is not None:
        adj = Adjudication.model_validate(_json(root, adjudication))
        if adj.packet_sha256 != binding.packet.sha256 or sorted(
            adj.original_judgments
        ) != sorted(response_hashes):
            raise ValueError("stale adjudication/original judgment binding")
        _validate_verdicts(adj.dimensions, packet, inputs)
    for key in inputs["policy"]["dimensions"]:
        verdicts = {j.dimensions[key].verdict for j in judgments.values()}
        resolved[key] = (
            next(iter(verdicts))
            if len(verdicts) == 1
            else adj.dimensions[key].verdict
            if adjudication is not None
            else "unresolved"
        )
    status = (
        "fail"
        if "fail" in resolved.values()
        else "unmeasured"
        if "unresolved" in resolved.values()
        else "pass"
    )
    return {"status": status, "dimensions": resolved, "reason": None}
