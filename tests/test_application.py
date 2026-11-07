"""Independent Stage 3 requirements 18–28; fictional inputs, no model runs.

Expected behavior comes from the Stage 3 contract/behavioral specification, not
the preparation/application implementations. Token ceilings are interface checks,
not measurements of actual tokenizer feasibility or semantic quality.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from zoneinfo import TZPATH

import pytest

from messaging_nlp.application import execute_condition
from messaging_nlp.application_types import GenerationOutput
from messaging_nlp.context import PublicDataset
from messaging_nlp.contracts import (
    Entity,
    Extraction,
    ExtractionResult,
    TechnicalFailure,
)
from messaging_nlp.encoder_cache import (
    ExtractionArtifact,
    retained_identity,
    text_sha256,
    write_artifact,
)
from messaging_nlp.io import canonical_json
from messaging_nlp.preparation import prepare_inputs, verify_input
from messaging_nlp.temporal import TimezoneBinding

CONDITIONS = ("baseline", "B", "A+B")
INSTRUCTION = (
    "When entity annotations are provided, treat their span/type labels as correct "
    "when interpreting the request."
)
NOISY = "  Ask Ren about supper — plz don’t ‘fix’ cafe\u0301 🙂\nthanks!  "
STAMP = "2026-10-26T10:00:00+00:00"
START = "2026-10-27T15:00:00+00:00"


class ScriptedProvider:
    """Only generation is available: no send, retry, state or live effects API."""

    def __init__(self, raw, failure=None):
        self.output = GenerationOutput(raw_output=raw, failure=failure)
        self.calls = []

    def generate(self, prompt, *, max_input_tokens, max_output_tokens):
        assert (max_input_tokens, max_output_tokens) == (2048, 1024)
        self.calls.append(prompt)
        assert len(self.calls) == 1, "a condition must never retry or repair"
        return self.output


def nondraft(action="unsupported", reason="Fictional blocker"):
    return {
        "decision": {
            "action": action,
            "selection": None,
            "draft": None,
            "clarification": "Which supper?" if action == "clarify" else None,
            "reason": reason,
        },
        "mentions": [],
    }


def fictional_dataset(text=NOISY):
    """No public export, evaluator metadata or historical study IDs are read."""
    return PublicDataset.model_validate(
        {
            "owner_id": "descriptive-owner-private",
            "requests": [
                {
                    "message_id": "descriptive-request-private",
                    "text": text,
                    "owner_id": "descriptive-owner-private",
                    "request_time": STAMP,
                    "time_zone": "UTC",
                    "context_snapshot_id": "descriptive-snapshot-private",
                }
            ],
            "contacts": [
                {
                    "record_id": "descriptive-contact-record-private",
                    "contact_id": "descriptive-ren-private",
                    "version": 1,
                    "name": "Ren",
                    "aliases": [],
                    "relationship": "friend",
                    "available_at": STAMP,
                    "provenance": "PROVENANCE_MUST_NOT_LEAK",
                }
            ],
            "calendar": [
                {
                    "record_id": "descriptive-event-record-private",
                    "event_id": "descriptive-supper-private",
                    "version": 1,
                    "title": "supper",
                    "participant_contact_ids": ["descriptive-ren-private"],
                    "start": START,
                    "end": None,
                    "location": None,
                    "status": "scheduled",
                    "available_at": STAMP,
                    "provenance": "PROVENANCE_MUST_NOT_LEAK",
                }
            ],
            "history": [
                {
                    "record_id": "descriptive-message-record-private",
                    "sender_contact_id": "descriptive-ren-private",
                    "recipient_contact_id": "descriptive-owner-private",
                    "text": "Bring the blue book to supper.",
                    "timestamp": STAMP,
                    "available_at": STAMP,
                    "provenance": "PROVENANCE_MUST_NOT_LEAK",
                }
            ],
            "snapshots": [
                {
                    "context_snapshot_id": "descriptive-snapshot-private",
                    "timestamp": STAMP,
                    "record_ids": [
                        "descriptive-contact-record-private",
                        "descriptive-event-record-private",
                        "descriptive-message-record-private",
                    ],
                }
            ],
        }
    )


@pytest.fixture
def binding(tmp_path):
    # The fixture explicitly trusts UTC's system TZif, then copies and hashes it.
    # No production retained-data or model assets are accessed.
    source = next(
        (Path(p) / "UTC" for p in TZPATH if (Path(p) / "UTC").is_file()), None
    )
    assert source is not None, "fixture requires a system UTC TZif"
    content = source.read_bytes()
    assert content.startswith(b"TZif")
    path = tmp_path / "fictional-UTC.tzif"
    path.write_bytes(content)
    return TimezoneBinding("UTC", path, hashlib.sha256(content).hexdigest())


def prepare(
    tmp_path,
    binding,
    *,
    text=NOISY,
    mode="ok",
    extra_context=False,
    duplicate_name=False,
):
    dataset = fictional_dataset(text)
    if duplicate_name:
        data = dataset.model_dump()
        other = dict(data["contacts"][0])
        other.update(record_id="inviter-record", contact_id="second-ren")
        data["contacts"].append(other)
        data["history"][0].update(
            sender_contact_id="second-ren",
            text="Please join me for supper. Bring the blue book.",
        )
        data["calendar"][0]["participant_contact_ids"] = ["second-ren"]
        data["snapshots"][0]["record_ids"].append("inviter-record")
        dataset = PublicDataset.model_validate(data)
    if extra_context:
        data = dataset.model_dump()
        other = dict(data["contacts"][0])
        other.update(
            record_id="unmatched-person-record",
            contact_id="unmatched-person",
            name="Mira",
        )
        data["contacts"].append(other)
        other_event = dict(data["calendar"][0])
        other_event.update(
            record_id="unmatched-event-record",
            event_id="unmatched-event",
            title="Cobalt kite",
            participant_contact_ids=[],
        )
        data["calendar"].append(other_event)
        data["snapshots"][0]["record_ids"].extend(
            ["unmatched-person-record", "unmatched-event-record"]
        )
        dataset = PublicDataset.model_validate(data)
    identity = retained_identity()  # Identity-only: opens no model/checkpoint files.
    message_id = dataset.requests[0].message_id
    extraction_path = tmp_path / "fictional-extraction.json"
    if mode != "missing":
        entities = []
        if mode != "empty":
            for surface, kind in (("Ren", "PERSON"), ("supper", "EVENT")):
                start = text.index(surface)
                entities.append(
                    Entity(
                        type=kind, text=surface, start=start, end=start + len(surface)
                    )
                )
            entities.sort(key=lambda entity: entity.start)
        failure = TechnicalFailure(
            stage="extraction",
            code="runtime_error",
            message="fictional extraction failure",
        )
        result = ExtractionResult(
            message_id="wrong-request" if mode == "stale" else message_id,
            status="failed" if mode == "failed" else "ok",
            extraction=None if mode == "failed" else Extraction(entities=entities),
            failure=failure if mode == "failed" else None,
            raw_output="RAW_EXTRACTOR_DIAGNOSTIC_MUST_NOT_LEAK",
            input_sha256=text_sha256(text),
            extractor_sha256=identity.digest(),
        )
        write_artifact(
            extraction_path,
            ExtractionArtifact(
                text=text,
                identity=identity,
                result=result,
                diagnostics={"private": "DIAGNOSTICS_MUST_NOT_LEAK"},
            ),
        )
        if mode == "malformed":
            extraction_path.write_bytes(b'{"broken":')
    inputs = prepare_inputs(
        dataset,
        message_id,
        root=tmp_path,
        baseline_path="baseline.json",
        assisted_path="assisted.json",
        extraction_path=extraction_path.name,
        identity=identity,
        binding=binding,
    )
    return dataset, inputs


def packet(dataset, inputs, tmp_path, binding, condition="B"):
    sealed = inputs.baseline if condition == "baseline" else inputs.assisted
    assert sealed is not None
    return verify_input(
        sealed,
        dataset=dataset,
        message_id=dataset.requests[0].message_id,
        root=tmp_path,
        binding=binding,
    )


def execute(dataset, inputs, tmp_path, binding, condition="B", raw=None, failure=None):
    if raw is None:
        raw = json.dumps(nondraft())
    provider = ScriptedProvider(raw, failure)
    result = execute_condition(
        dataset,
        dataset.requests[0].message_id,
        condition,
        inputs,
        provider,
        root=tmp_path,
        run_id="fictional-run",
        binding=binding,
    )
    assert result.provider_calls == len(provider.calls)
    assert result.row.attempt == 1
    assert "unmeasured" in result.validation_scope.lower()
    return result, provider


def alias_for(prepared_packet, kind, internal_id):
    return next(
        alias
        for alias, target in prepared_packet.aliases.items()
        if target.kind == kind and target.internal_id == internal_id
    )


def draft_envelope(prepared_packet, draft="Ren, supper starts at three pm."):
    contact = alias_for(prepared_packet, "contact", "descriptive-ren-private")
    contact_record = alias_for(
        prepared_packet, "record", "descriptive-contact-record-private"
    )
    event = alias_for(prepared_packet, "event", "descriptive-supper-private")
    event_record = alias_for(
        prepared_packet, "record", "descriptive-event-record-private"
    )
    return {
        "decision": {
            "action": "draft",
            "draft": draft,
            "clarification": None,
            "reason": None,
            "selection": {
                "recipient_contact_ids": [contact],
                "selected_event_id": event,
                "source_record_ids": [contact_record, event_record],
                "facts": [
                    {
                        "field": "name",
                        "value": "Ren",
                        "source_record_ids": [contact_record],
                    },
                    {
                        "field": "title",
                        "value": "supper",
                        "source_record_ids": [event_record],
                    },
                    {
                        "field": "start",
                        "value": START,
                        "source_record_ids": [event_record],
                    },
                ],
            },
        },
        "mentions": [
            {"surface": "Ren", "rendering_kind": "literal", "fact_index": 0},
            {"surface": "supper", "rendering_kind": "literal", "fact_index": 1},
            {"surface": "three pm", "rendering_kind": "clock", "fact_index": 2},
        ],
    }


def test_original_unicode_opaque_prompts_and_actual_shared_assisted_file(
    tmp_path, binding
):
    dataset, inputs = prepare(tmp_path, binding)
    sealed = inputs.assisted
    assert sealed is not None and type(sealed.data) is bytes
    saved = (tmp_path / sealed.path).read_bytes()
    assert saved == sealed.data
    assert hashlib.sha256(saved).hexdigest() == sealed.sha256
    before = packet(dataset, inputs, tmp_path, binding)
    assert (
        packet(dataset, inputs, tmp_path, binding, "baseline").aliases == before.aliases
    )
    assert before.extraction_artifact_path == "fictional-extraction.json"
    extraction_bytes = (tmp_path / before.extraction_artifact_path).read_bytes()
    assert (
        before.extraction_artifact_sha256
        == hashlib.sha256(extraction_bytes).hexdigest()
    )

    # Deliberately construct A+B before B: no sibling prompt/output can be reused.
    ab, _ = execute(dataset, inputs, tmp_path, binding, "A+B")
    b, _ = execute(dataset, inputs, tmp_path, binding, "B")
    baseline, _ = execute(dataset, inputs, tmp_path, binding, "baseline")
    assert ab.row.status == b.row.status == baseline.row.status == "ok"
    assert b.row.preprocessing_sha256 == ab.row.preprocessing_sha256 == sealed.sha256
    assert (
        b.row.extraction_result == ab.row.extraction_result == before.extraction_result
    )
    assert inputs.assisted is sealed  # One actual saved artifact, not equivalents.
    assert (tmp_path / sealed.path).read_bytes() == saved
    assert b.prompt.common == ab.prompt.common
    assert b.prompt.system == ab.prompt.system == baseline.prompt.system
    assert b.prompt.entities is baseline.prompt.entities is None
    assert ab.prompt.entities is not None
    assert b.prompt.user == b.prompt.common
    assert (
        ab.prompt.user
        == b.prompt.common + b"\nENTITY ANNOTATIONS\n" + ab.prompt.entities
    )
    for result in (baseline, b, ab):
        assert result.row.request.text == NOISY
        # Decode JSON string escaping if used; the request must remain exact.
        try:
            common = json.loads(result.prompt.common)
        except json.JSONDecodeError:
            assert NOISY.encode() in result.prompt.common
        else:
            assert NOISY in list(_strings(common))
        visible = result.prompt.system + result.prompt.user.decode()
        assert INSTRUCTION in visible
        for private in (
            "descriptive-",
            "PROVENANCE_MUST_NOT_LEAK",
            "RAW_EXTRACTOR_DIAGNOSTIC_MUST_NOT_LEAK",
            "DIAGNOSTICS_MUST_NOT_LEAK",
            "fictional-run",
        ):
            assert private not in visible
    assert dataset.requests[0].text.encode() == NOISY.encode()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


@pytest.mark.parametrize("mode", ["missing", "failed", "stale", "malformed"])
def test_bad_extraction_blocks_both_assisted_conditions_only(tmp_path, binding, mode):
    dataset, inputs = prepare(tmp_path, binding, mode=mode)
    assert inputs.assisted is None and inputs.assisted_failure is not None
    for condition in ("B", "A+B"):
        result, provider = execute(dataset, inputs, tmp_path, binding, condition)
        assert not provider.calls
        assert result.row.status == "unexecuted"
        assert result.row.failure is not None
        assert result.row.raw_output is result.row.decision is result.prompt is None
    baseline, provider = execute(dataset, inputs, tmp_path, binding, "baseline")
    assert baseline.row.status == "ok" and len(provider.calls) == 1
    assert baseline.row.extraction_result is None


@pytest.mark.parametrize("condition", CONDITIONS)
def test_successful_empty_extraction_is_not_failure(tmp_path, binding, condition):
    dataset, inputs = prepare(tmp_path, binding, mode="empty")
    assert inputs.assisted is not None and inputs.assisted_failure is None
    result, provider = execute(dataset, inputs, tmp_path, binding, condition)
    assert result.row.status == "ok" and len(provider.calls) == 1
    if condition != "baseline":
        assert result.row.extraction_result.extraction.entities == []


@pytest.mark.parametrize(
    "mutation", ["file", "request", "context", "policy", "extraction", "timezone"]
)
@pytest.mark.parametrize("condition", ("B", "A+B"))
def test_stale_or_mutated_inputs_fail_before_generation(
    tmp_path, binding, mutation, condition
):
    dataset, inputs = prepare(tmp_path, binding)
    sealed = inputs.assisted
    if mutation == "file":
        (tmp_path / sealed.path).write_bytes(sealed.data + b" ")
    elif mutation == "request":
        request = dataset.requests[0].model_copy(update={"text": NOISY + "!"})
        dataset = dataset.model_copy(update={"requests": [request]})
    elif mutation == "context":
        contact = dataset.contacts[0].model_copy(update={"name": "Different Ren"})
        dataset = dataset.model_copy(update={"contacts": [contact]})
    elif mutation == "policy":
        # Even a self-consistent digest cannot supply the expected policy identity.
        content = json.loads(sealed.data)
        content["policy_sha256"] = "0" * 64
        data = canonical_json(content)
        sealed = replace(sealed, data=data, sha256=hashlib.sha256(data).hexdigest())
        (tmp_path / sealed.path).write_bytes(data)
        inputs = replace(inputs, assisted=sealed)
    elif mutation == "extraction":
        path = tmp_path / "fictional-extraction.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        binding.tzif_path.write_bytes(binding.tzif_path.read_bytes() + b" ")
    with pytest.raises((ValueError, OSError)):
        verify_input(
            sealed,
            dataset=dataset,
            message_id=dataset.requests[0].message_id,
            root=tmp_path,
            binding=binding,
        )
    result, provider = execute(dataset, inputs, tmp_path, binding, condition)
    assert provider.calls == []
    assert result.row.status == "unexecuted" and result.row.failure is not None


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize(
    "invalid_payload",
    ["not bytes", None, bytearray(b"not bytes"), memoryview(b"not bytes")],
    ids=["text", "none", "mutable-bytearray", "memoryview"],
)
def test_invalid_typed_sealed_payload_retains_structured_zero_call_failure(
    tmp_path, binding, condition, invalid_payload
):
    dataset, inputs = prepare(tmp_path, binding)
    original = inputs.baseline if condition == "baseline" else inputs.assisted
    invalid = replace(original, data=invalid_payload)
    inputs = replace(
        inputs, **{"baseline" if condition == "baseline" else "assisted": invalid}
    )
    result, provider = execute(dataset, inputs, tmp_path, binding, condition)
    assert provider.calls == [] and result.provider_calls == 0
    assert result.row.status == "unexecuted"
    assert result.row.failure.stage == "preprocessing"
    assert result.row.failure.code == "invalid_output"
    assert result.row.preprocessing_sha256 is None  # No coercion or fabricated digest.
    assert result.row.raw_output is result.row.decision is result.prompt is None
    assert result.row.candidates is result.row.extraction_result is None
    assert (tmp_path / original.path).read_bytes() == original.data


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("fault", ["stale-hash", "mutated-bytes"])
def test_actual_bytes_failed_seal_still_retains_actual_payload_hash(
    tmp_path, binding, condition, fault
):
    dataset, inputs = prepare(tmp_path, binding)
    original = inputs.baseline if condition == "baseline" else inputs.assisted
    invalid = (
        replace(original, sha256="0" * 64)
        if fault == "stale-hash"
        else replace(original, data=original.data + b" ")
    )
    inputs = replace(
        inputs, **{"baseline" if condition == "baseline" else "assisted": invalid}
    )
    result, provider = execute(dataset, inputs, tmp_path, binding, condition)
    assert provider.calls == [] and result.provider_calls == 0
    assert result.row.status == "unexecuted"
    assert result.row.failure.stage == "preprocessing"
    assert result.row.failure.code == "invalid_output"
    assert "byte/hash mismatch" in result.row.failure.message
    assert result.row.preprocessing_sha256 == hashlib.sha256(invalid.data).hexdigest()
    assert result.row.raw_output is result.row.decision is result.prompt is None
    assert (tmp_path / original.path).read_bytes() == original.data


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":',
        '{"decision":{},"decision":{},"mentions":[]}',
        json.dumps(nondraft()).replace(
            '"action": "unsupported"', '"action": "draft", "action": "unsupported"'
        ),
        json.dumps(nondraft()).replace('"unsupported"', '"send"'),
        "```json\n" + json.dumps(nondraft()) + "\n```",
        json.dumps(nondraft()) + " trailing prose",
    ],
)
def test_malformed_outputs_are_retained_without_repair(tmp_path, binding, raw):
    dataset, inputs = prepare(tmp_path, binding)
    result, provider = execute(dataset, inputs, tmp_path, binding, raw=raw)
    assert len(provider.calls) == 1
    assert result.row.status == "failed"
    assert result.row.failure.stage == "validation"
    assert result.row.raw_output == raw and result.row.decision is None


def test_partial_provider_failure_keeps_raw_without_parsing_or_retry(tmp_path, binding):
    dataset, inputs = prepare(tmp_path, binding)
    failure = TechnicalFailure(
        stage="generation", code="truncated", message="fictional cutoff"
    )
    raw = '{"decision":{"action":"draft"'
    result, provider = execute(
        dataset, inputs, tmp_path, binding, raw=raw, failure=failure
    )
    assert len(provider.calls) == 1
    assert result.row.status == "failed" and result.row.failure == failure
    assert result.row.raw_output == raw and result.row.decision is None


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("action", ["unsupported", "clarify"])
def test_reason_only_unsupported_and_unselected_clarification(
    tmp_path, binding, condition, action
):
    dataset, inputs = prepare(tmp_path, binding)
    result, _ = execute(
        dataset, inputs, tmp_path, binding, condition, raw=json.dumps(nondraft(action))
    )
    assert result.row.status == "ok"
    assert result.row.decision.selection is result.row.decision.draft is None
    assert result.envelope.mentions == []
    assert "semantic_success" not in result.row.model_dump()


def test_previous_outputs_never_enter_later_prompts(tmp_path, binding):
    dataset, inputs = prepare(tmp_path, binding)
    sentinel = "UNIQUE_PREVIOUS_ASSISTANT_OUTPUT_879321"
    first, _ = execute(
        dataset,
        inputs,
        tmp_path,
        binding,
        "B",
        raw=json.dumps(nondraft(reason=sentinel)),
    )
    assert first.row.status == "ok"
    for condition in ("A+B", "baseline", "B"):
        later, _ = execute(dataset, inputs, tmp_path, binding, condition)
        assert sentinel not in later.prompt.system + later.prompt.user.decode()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_supported_draft_resolves_opaque_aliases_without_semantic_success(
    tmp_path, binding, condition
):
    dataset, inputs = prepare(tmp_path, binding)
    envelope = draft_envelope(packet(dataset, inputs, tmp_path, binding, condition))
    raw = json.dumps(envelope)
    result, _ = execute(dataset, inputs, tmp_path, binding, condition, raw=raw)
    assert result.row.status == "ok" and result.row.raw_output == raw
    assert result.row.decision.selection.recipient_contact_ids == [
        "descriptive-ren-private"
    ]
    assert (
        result.row.decision.selection.selected_event_id == "descriptive-supper-private"
    )
    assert result.row.decision.draft == envelope["decision"]["draft"]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_duplicate_name_invitation_author_can_be_selected_without_preemptive_veto(
    tmp_path, binding, condition
):
    dataset, inputs = prepare(tmp_path, binding, duplicate_name=True)
    saved = packet(dataset, inputs, tmp_path, binding, condition)
    assert set(saved.search.candidates.contact_ids) == {
        "descriptive-ren-private",
        "second-ren",
    }
    invitation = saved.search.evidence.history[0]
    assert invitation.sender_contact_id == "second-ren"
    assert invitation.recipient_contact_id == dataset.owner_id
    output = draft_envelope(saved)
    selection = output["decision"]["selection"]
    selection["recipient_contact_ids"] = [alias_for(saved, "contact", "second-ren")]
    inviter_record = alias_for(saved, "record", "inviter-record")
    invitation_record = alias_for(saved, "record", invitation.record_id)
    selection["source_record_ids"].extend([inviter_record, invitation_record])
    selection["facts"][0]["source_record_ids"] = [inviter_record]
    selection["facts"].append(
        {
            "field": "message_text",
            "value": invitation.text,
            "source_record_ids": [invitation_record],
        }
    )
    result, provider = execute(
        dataset, inputs, tmp_path, binding, condition, raw=json.dumps(output)
    )
    assert len(provider.calls) == 1 and result.row.status == "ok"
    assert result.row.decision.action == "draft"
    assert result.row.decision.selection.recipient_contact_ids == ["second-ren"]


@pytest.mark.parametrize(
    "bad_reference", ["unknown", "wrongkind", "internal", "outside"]
)
def test_alias_resolution_never_guesses_or_accepts_internal_ids(
    tmp_path, binding, bad_reference
):
    dataset, inputs = prepare(tmp_path, binding)
    prepared_packet = packet(dataset, inputs, tmp_path, binding)
    envelope = draft_envelope(prepared_packet)
    if bad_reference == "unknown":
        bad = "UNKNOWN_ALIAS_879321"
    elif bad_reference == "wrongkind":
        bad = alias_for(prepared_packet, "event", "descriptive-supper-private")
    elif bad_reference == "internal":
        bad = "descriptive-ren-private"
    else:
        # Owner is a real eligible alias but is not a candidate contact/source.
        bad = alias_for(prepared_packet, "owner", dataset.owner_id)
        envelope["decision"]["selection"]["source_record_ids"].append(bad)
    envelope["decision"]["selection"]["recipient_contact_ids"] = [bad]
    raw = json.dumps(envelope)
    result, provider = execute(dataset, inputs, tmp_path, binding, raw=raw)
    assert result.row.status == "failed" and result.row.failure.stage == "validation"
    assert result.row.raw_output == raw and len(provider.calls) == 1


@pytest.mark.parametrize(
    "request_text",
    [
        "Ask Ren: Could we move supper to four pm?",
        "Ask Ren: Would four pm work instead for supper?",
    ],
)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_current_and_proposed_multifact_envelope_same_policy(
    tmp_path, binding, request_text, condition
):
    dataset, inputs = prepare(tmp_path, binding, text=request_text)
    prepared_packet = packet(dataset, inputs, tmp_path, binding, condition)
    anchors_before = canonical_json([a.model_dump() for a in prepared_packet.anchors])
    proposal = next(a for a in prepared_packet.anchors if a.kind == "proposal_time")
    assert request_text[proposal.start : proposal.end] == proposal.text
    proposal_alias = alias_for(prepared_packet, "anchor", proposal.anchor_id)
    envelope = draft_envelope(
        prepared_packet,
        "Ren, supper starts at three pm. Could we move it to four pm? Bring the blue book.",
    )
    message_record = alias_for(
        prepared_packet, "record", "descriptive-message-record-private"
    )
    selection = envelope["decision"]["selection"]
    selection["source_record_ids"].append(message_record)
    selection["facts"].append(
        {
            "field": "message_text",
            "value": "Bring the blue book to supper.",
            "source_record_ids": [message_record],
        }
    )
    envelope["mentions"].extend(
        [
            {
                "surface": "four pm",
                "rendering_kind": "proposal_clock",
                "anchor_id": proposal_alias,
            },
            {"surface": "the blue book", "rendering_kind": "excerpt", "fact_index": 3},
        ]
    )
    raw = json.dumps(envelope)
    result, _ = execute(dataset, inputs, tmp_path, binding, condition, raw=raw)
    assert result.row.status == "ok" and result.row.raw_output == raw
    after = packet(dataset, inputs, tmp_path, binding, condition)
    assert canonical_json([a.model_dump() for a in after.anchors]) == anchors_before
    # Same anchor must not certify a statement of current schedule.
    envelope["decision"]["draft"] = (
        "Ren, supper now starts at four pm. Bring the blue book."
    )
    envelope["mentions"] = [
        m for m in envelope["mentions"] if m["surface"] != "three pm"
    ]
    rejected, _ = execute(
        dataset, inputs, tmp_path, binding, condition, raw=json.dumps(envelope)
    )
    assert rejected.row.status == "failed" and rejected.row.decision is None


def test_repeated_clock_surface_is_rejected_without_occurrence_guessing(
    tmp_path, binding
):
    dataset, inputs = prepare(tmp_path, binding)
    envelope = draft_envelope(
        packet(dataset, inputs, tmp_path, binding),
        "Ren, supper starts at three pm; yes, three pm.",
    )
    result, _ = execute(dataset, inputs, tmp_path, binding, raw=json.dumps(envelope))
    assert result.row.status == "failed" and result.row.failure.stage == "validation"


@pytest.mark.parametrize("action", ["unsupported", "clarify"])
def test_nondraft_cannot_retain_selection_or_mentions(tmp_path, binding, action):
    dataset, inputs = prepare(tmp_path, binding)
    draft = draft_envelope(packet(dataset, inputs, tmp_path, binding))
    envelope = nondraft(action)
    envelope["decision"]["selection"] = draft["decision"]["selection"]
    envelope["mentions"] = draft["mentions"]
    raw = json.dumps(envelope)
    result, _ = execute(dataset, inputs, tmp_path, binding, raw=raw)
    assert result.row.status == "failed" and result.row.raw_output == raw
    assert result.row.decision is None


@pytest.mark.parametrize("condition", CONDITIONS)
def test_single_compound_datetime_binding_covers_date_and_clock(
    tmp_path, binding, condition
):
    dataset, inputs = prepare(tmp_path, binding)
    envelope = draft_envelope(
        packet(dataset, inputs, tmp_path, binding, condition),
        "Ren, supper starts on October 27, 2026 at 3 pm.",
    )
    envelope["mentions"][2] = {
        "surface": "October 27, 2026 at 3 pm",
        "rendering_kind": "datetime",
        "fact_index": 2,
    }
    result, _ = execute(
        dataset, inputs, tmp_path, binding, condition, raw=json.dumps(envelope)
    )
    assert result.row.status == "ok"
    # Nested components cannot be added as overlapping bindings.
    envelope["mentions"].append(
        {"surface": "3 pm", "rendering_kind": "clock", "fact_index": 2}
    )
    rejected, _ = execute(
        dataset, inputs, tmp_path, binding, condition, raw=json.dumps(envelope)
    )
    assert rejected.row.status == "failed" and rejected.row.decision is None


@pytest.mark.parametrize("kind", ["contact", "event", "record"])
def test_real_eligible_alias_outside_packed_candidates_is_rejected(
    tmp_path, binding, kind
):
    dataset, inputs = prepare(tmp_path, binding, extra_context=True)
    saved = packet(dataset, inputs, tmp_path, binding)
    envelope = draft_envelope(saved)
    selection = envelope["decision"]["selection"]
    if kind == "contact":
        assert "unmatched-person" not in saved.search.candidates.contact_ids
        selection["recipient_contact_ids"] = [
            alias_for(saved, kind, "unmatched-person")
        ]
    elif kind == "event":
        assert "unmatched-event" not in saved.search.candidates.event_ids
        selection["selected_event_id"] = alias_for(saved, kind, "unmatched-event")
    else:
        assert (
            "unmatched-person-record" not in saved.search.candidates.source_record_ids
        )
        selection["source_record_ids"].append(
            alias_for(saved, kind, "unmatched-person-record")
        )
    result, _ = execute(dataset, inputs, tmp_path, binding, raw=json.dumps(envelope))
    assert result.row.status == "failed"
    assert "outside candidates" in result.row.failure.message


def test_execution_never_reruns_search_extraction_or_anchor_creation(
    tmp_path, binding, monkeypatch
):
    dataset, inputs = prepare(tmp_path, binding)

    def forbidden(*args, **kwargs):
        raise AssertionError("sealed preparation must not be recomputed")

    monkeypatch.setattr("messaging_nlp.retrieval.search", forbidden)
    monkeypatch.setattr("messaging_nlp.drafting.precompute_anchors", forbidden)
    monkeypatch.setattr("messaging_nlp.encoder.load_encoder", forbidden)
    for condition in CONDITIONS:
        result, _ = execute(dataset, inputs, tmp_path, binding, condition)
        assert result.row.status == "ok"


def test_swapped_condition_receipts_block_without_provider_calls(tmp_path, binding):
    dataset, inputs = prepare(tmp_path, binding)
    swapped = replace(inputs, baseline=inputs.assisted, assisted=inputs.baseline)
    for condition in CONDITIONS:
        result, provider = execute(dataset, swapped, tmp_path, binding, condition)
        assert result.row.status == "unexecuted"
        assert result.row.failure.stage == "preprocessing"
        assert not provider.calls


def test_exception_with_partial_output_is_preserved(tmp_path, binding):
    dataset, inputs = prepare(tmp_path, binding)

    class PartialError(RuntimeError):
        raw_output = '{"decision":'

    class Provider:
        calls = 0

        def generate(self, prompt, **limits):
            self.calls += 1
            raise PartialError("fictional process interruption")

    provider = Provider()
    result = execute_condition(
        dataset,
        dataset.requests[0].message_id,
        "B",
        inputs,
        provider,
        root=tmp_path,
        run_id="partial",
        binding=binding,
    )
    assert provider.calls == result.provider_calls == 1
    assert result.row.raw_output == PartialError.raw_output
    assert result.row.status == "failed"
    assert result.row.failure.stage == "generation"


@pytest.mark.parametrize("clock_field", ["timestamp", "available_at"])
def test_high_level_preparation_cannot_bypass_both_message_clocks(
    tmp_path, binding, clock_field
):
    data = fictional_dataset().model_dump()
    data["history"][0][clock_field] = "2026-10-26T10:00:01+00:00"
    dataset = PublicDataset.model_validate(data)
    with pytest.raises(ValueError, match="future"):
        prepare_inputs(
            dataset,
            dataset.requests[0].message_id,
            root=tmp_path,
            baseline_path="base.json",
            assisted_path="assisted.json",
            extraction_path="not-opened.json",
            identity=retained_identity(),
            binding=binding,
        )
    assert not (tmp_path / "base.json").exists()


def test_exclusive_preparation_keeps_prior_saved_bytes(tmp_path, binding):
    dataset, inputs = prepare(tmp_path, binding)
    before = (tmp_path / inputs.baseline.path).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_inputs(
            dataset,
            dataset.requests[0].message_id,
            root=tmp_path,
            baseline_path=inputs.baseline.path,
            assisted_path="new.json",
            extraction_path="fictional-extraction.json",
            identity=retained_identity(),
            binding=binding,
        )
    assert (tmp_path / inputs.baseline.path).read_bytes() == before


@pytest.mark.parametrize(
    "unsafe", ["../escape.json", "/absolute.json", "a/../b.json", "."]
)
def test_preparation_requires_explicit_project_relative_artifacts(
    tmp_path, binding, unsafe
):
    dataset = fictional_dataset()
    with pytest.raises(ValueError, match="project-relative"):
        prepare_inputs(
            dataset,
            dataset.requests[0].message_id,
            root=tmp_path,
            baseline_path=unsafe,
            assisted_path="assisted.json",
            extraction_path="extract.json",
            identity=retained_identity(),
            binding=binding,
        )
    assert not (tmp_path / "assisted.json").exists()


def test_artifact_symlink_cannot_leave_explicit_root(tmp_path, binding):
    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    (root / "redirect").symlink_to(external, target_is_directory=True)
    dataset = fictional_dataset()
    with pytest.raises(ValueError, match="leaves explicit project root"):
        prepare_inputs(
            dataset,
            dataset.requests[0].message_id,
            root=root,
            baseline_path="redirect/base.json",
            assisted_path="assist.json",
            extraction_path="extract.json",
            identity=retained_identity(),
            binding=binding,
        )
    assert not (external / "base.json").exists()


@pytest.mark.parametrize(
    "version_state", ["future-unlisted", "available-unlisted", "cancelled-listed"]
)
def test_high_level_versions_do_not_escape_snapshot_or_resurrect_cancelled_event(
    tmp_path, binding, version_state
):
    dataset, _ = prepare(tmp_path, binding, text="Ask Ren whether supper is cancelled")
    data = dataset.model_dump()
    later = dict(data["calendar"][0])
    later.update(record_id="cancelled-v2", version=2, status="cancelled")
    if version_state == "future-unlisted":
        later["available_at"] = "2026-10-26T10:00:01+00:00"
    data["calendar"].append(later)
    if version_state == "cancelled-listed":
        records = data["snapshots"][0]["record_ids"]
        records[records.index("descriptive-event-record-private")] = "cancelled-v2"
    dataset = PublicDataset.model_validate(data)
    root = tmp_path / "version-check"
    root.mkdir()
    (root / "extraction.json").write_bytes(
        (tmp_path / "fictional-extraction.json").read_bytes()
    )
    kwargs = dict(
        root=root,
        baseline_path="base.json",
        assisted_path="assist.json",
        extraction_path="extraction.json",
        identity=retained_identity(),
        binding=binding,
    )
    if version_state == "available-unlisted":
        with pytest.raises(ValueError, match="stale"):
            prepare_inputs(dataset, dataset.requests[0].message_id, **kwargs)
        assert not (root / "base.json").exists()
        return
    inputs = prepare_inputs(dataset, dataset.requests[0].message_id, **kwargs)
    saved = packet(dataset, inputs, root, binding)
    selected = saved.search.evidence.calendar
    assert len(selected) == 1
    expected_record = (
        "cancelled-v2"
        if version_state == "cancelled-listed"
        else "descriptive-event-record-private"
    )
    assert selected[0].record_id == expected_record
    assert selected[0].status == (
        "cancelled" if version_state == "cancelled-listed" else "scheduled"
    )
    if version_state == "cancelled-listed":
        output = nondraft()
        output["decision"].update(
            action="draft",
            draft="Is supper cancelled?",
            reason=None,
            selection={
                "recipient_contact_ids": [
                    alias_for(saved, "contact", "descriptive-ren-private")
                ],
                "selected_event_id": alias_for(
                    saved, "event", "descriptive-supper-private"
                ),
                "facts": [],
                "source_record_ids": [
                    alias_for(saved, "record", "descriptive-contact-record-private"),
                    alias_for(saved, "record", expected_record),
                ],
            },
        )
        result, provider = execute(
            dataset, inputs, root, binding, raw=json.dumps(output)
        )
        assert result.row.status == "ok" and len(provider.calls) == 1
