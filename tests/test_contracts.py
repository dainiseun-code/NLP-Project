"""Independent contract checks using invented requests and saved outcomes."""

import hashlib
import json
import unicodedata
from copy import deepcopy

import pytest
from pydantic import ValidationError

from messaging_nlp.contracts import (
    CandidateEvidence,
    Decision,
    Entity,
    Extraction,
    ExtractionResult,
    Request,
    RunRow,
    Selection,
)


def request_payload():
    return {
        "message_id": "fictional-request",
        "text": "  plz 📨 Zoe\u0308—tmrw??  ",
        "owner_id": "account",
        "request_time": "2026-06-02T12:00:00+00:00",
        "time_zone": "UTC",
        "context_snapshot_id": "snapshot",
    }


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_digest(value):
    # Independent spelling of the persisted canonical JSON identity.
    return digest(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def failure(stage="generation"):
    return {"stage": stage, "code": "timeout", "message": "fictional timeout"}


def extraction_payload():
    return {
        "message_id": "fictional-request",
        "status": "ok",
        "extraction": {"entities": []},
        "failure": None,
        "raw_output": "[]",
        "input_sha256": digest(request_payload()["text"]),
        "extractor_sha256": "b" * 64,
    }


def draft_payload():
    return {
        "action": "draft",
        "selection": {
            "recipient_contact_ids": ["zoe"],
            "selected_event_id": None,
            "facts": [],
            "source_record_ids": ["contact-zoe"],
        },
        "draft": "Hi Zoë, shall we meet tomorrow?",
        "clarification": None,
        "reason": None,
    }


def run_payload(condition="baseline", status="ok"):
    request = request_payload()
    return {
        "schema_version": 1,
        "run_id": "fictional-run",
        "condition": condition,
        "request": request,
        "request_sha256": request_digest(request),
        "context_sha256": "c" * 64,
        "preprocessing_sha256": "d" * 64 if status == "ok" else None,
        "extraction_result": extraction_payload() if condition != "baseline" else None,
        "candidates": {
            "contact_ids": ["zoe"],
            "event_ids": [],
            "source_record_ids": ["contact-zoe"],
        }
        if status == "ok"
        else None,
        "status": status,
        "decision": draft_payload() if status == "ok" else None,
        "failure": None if status == "ok" else failure(),
        "raw_output": None if status == "unexecuted" else "original raw bytes: !?",
        "attempt": 1,
    }


def test_request_preserves_all_six_original_fields():
    payload = request_payload()
    assert Request.model_validate_json(json.dumps(payload)).model_dump() == payload
    assert len(payload) == 6


@pytest.mark.parametrize(
    "field", ["gold", "prior_state", "scenario_id", "expected_action"]
)
def test_request_rejects_nonpublic_fields(field):
    with pytest.raises(ValidationError):
        Request.model_validate({**request_payload(), field: {}})


@pytest.mark.parametrize("field", list(request_payload()))
def test_request_requires_every_field(field):
    payload = request_payload()
    del payload[field]
    with pytest.raises(ValidationError):
        Request.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"text": 42},
        {"text": "  "},
        {"request_time": "2026-06-02T12:00:00"},
        {"time_zone": "Fictional/Nowhere"},
    ],
)
def test_request_rejects_invalid_types_and_clocks(change):
    with pytest.raises(ValidationError):
        Request.model_validate({**request_payload(), **change})


def test_exact_unicode_spans_use_original_codepoint_slices():
    text = "🛰 Zoe\u0308 / tmrw"
    extraction = Extraction(
        entities=[
            Entity(type="PERSON", text="Zoe\u0308", start=2, end=6),
            Entity(type="DATE", text="tmrw", start=9, end=13),
        ]
    )
    assert extraction.validate_text(text) is extraction
    normalized = unicodedata.normalize("NFC", text)
    with pytest.raises(ValueError):
        extraction.validate_text(normalized)


@pytest.mark.parametrize(
    "spans",
    [
        [(0, 2, "ab"), (1, 3, "bc")],
        [(2, 4, "cd"), (0, 2, "ab")],
        [(0, 2, "ab"), (0, 2, "ab")],
        [(0, 5, "abcde")],
        [(0, 2, "AB")],
    ],
)
def test_reject_overlap_unsorted_duplicate_outside_or_nonliteral_spans(spans):
    extraction = Extraction(
        entities=[
            Entity(type="TOPIC", start=start, end=end, text=text)
            for start, end, text in spans
        ]
    )
    with pytest.raises(ValueError):
        extraction.validate_text("abcd")


def test_adjacent_spans_and_empty_extraction_are_valid():
    Extraction(
        entities=[
            Entity(type="TOPIC", text="a", start=0, end=1),
            Entity(type="TOPIC", text="b", start=1, end=2),
        ]
    ).validate_text("ab")
    empty = ExtractionResult.model_validate(extraction_payload())
    failed = ExtractionResult.model_validate(
        {
            **extraction_payload(),
            "status": "failed",
            "extraction": None,
            "failure": failure("extraction"),
            "raw_output": None,
        }
    )
    assert empty.extraction.entities == []
    assert failed.extraction is None
    assert failed.failure.code == "timeout"


@pytest.mark.parametrize(
    "change",
    [
        {"extraction": None},
        {"failure": failure("extraction")},
        {"status": "failed"},
        {"status": "failed", "extraction": None, "failure": failure("runner")},
    ],
)
def test_incoherent_extraction_is_not_empty_success(change):
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({**extraction_payload(), **change})


@pytest.mark.parametrize("action", ["draft", "clarify", "unsupported"])
def test_coherent_actions_roundtrip(action):
    payload = (
        draft_payload()
        if action == "draft"
        else {
            "action": action,
            "selection": None,
            "draft": None,
            "clarification": "Which Zoë?" if action == "clarify" else None,
            "reason": "Ambiguous recipient" if action == "clarify" else "Cannot send",
        }
    )
    assert Decision.model_validate_json(json.dumps(payload)).model_dump() == payload


@pytest.mark.parametrize(
    "change",
    [
        {"action": "unsupported"},
        {"action": "clarify"},
        {"selection": None},
        {"draft": None},
        {"reason": "Cannot send"},
        {"clarification": "Who?"},
    ],
)
def test_incoherent_decision_rejected(change):
    with pytest.raises(ValidationError):
        Decision.model_validate({**draft_payload(), **change})


@pytest.mark.parametrize(
    "extra",
    [
        {"draft": "Fallback"},
        {"selection": draft_payload()["selection"]},
        {"clarification": "Who?"},
        {"reason": None},
    ],
)
def test_unsupported_is_reason_only(extra):
    payload = {
        "action": "unsupported",
        "selection": None,
        "draft": None,
        "clarification": None,
        "reason": "No sending capability",
    }
    with pytest.raises(ValidationError):
        Decision.model_validate({**payload, **extra})


def test_duplicate_candidates_recipients_and_unbound_fact_sources_rejected():
    with pytest.raises(ValidationError):
        CandidateEvidence(
            contact_ids=["zoe", "zoe"], event_ids=[], source_record_ids=[]
        )
    selection = draft_payload()["selection"]
    with pytest.raises(ValidationError):
        Selection.model_validate({**selection, "recipient_contact_ids": ["zoe", "zoe"]})
    with pytest.raises(ValidationError):
        Selection.model_validate(
            {
                **selection,
                "facts": [
                    {
                        "field": "name",
                        "value": "Zoë",
                        "source_record_ids": ["not-selected"],
                    },
                ],
            }
        )


def test_selection_allows_recipient_third_party_and_two_message_chain():
    facts = [
        {"field": "name", "value": "Zoë", "source_record_ids": ["contact-zoe"]},
        {"field": "name", "value": "Inez", "source_record_ids": ["contact-inez"]},
        {
            "field": "message_text",
            "value": "Inez offered me The Glass Orchard.",
            "source_record_ids": ["message-offer"],
        },
        {
            "field": "message_text",
            "value": "Zoë is reading my copy now.",
            "source_record_ids": ["message-reader"],
        },
    ]
    payload = {
        **draft_payload()["selection"],
        "source_record_ids": [
            "contact-zoe",
            "contact-inez",
            "message-offer",
            "message-reader",
        ],
        "facts": facts,
    }
    # Inez is a third party, not an additional selected recipient.
    assert Selection.model_validate(payload).model_dump() == payload


@pytest.mark.parametrize("field", ["name", "message_text"])
def test_same_literal_fact_from_distinct_sources_is_allowed(field):
    facts = [
        {"field": field, "value": "A shared literal", "source_record_ids": [source]}
        for source in ("source-one", "source-two")
    ]
    payload = {
        **draft_payload()["selection"],
        "source_record_ids": ["source-one", "source-two"],
        "facts": facts,
    }
    assert Selection.model_validate(payload).model_dump() == payload


@pytest.mark.parametrize("field", ["name", "message_text"])
def test_distinct_literal_facts_from_same_source_are_allowed(field):
    facts = [
        {"field": field, "value": value, "source_record_ids": ["source-one"]}
        for value in ("First literal", "Second literal")
    ]
    payload = {
        **draft_payload()["selection"],
        "source_record_ids": ["source-one"],
        "facts": facts,
    }
    assert Selection.model_validate(payload).model_dump() == payload


@pytest.mark.parametrize("field", ["name", "message_text"])
@pytest.mark.parametrize("reverse_sources", [False, True])
def test_exact_duplicate_facts_rejected_independent_of_source_order(
    field, reverse_sources
):
    sources = ["source-one", "source-two"]
    facts = [
        {"field": field, "value": "Same literal", "source_record_ids": sources},
        {
            "field": field,
            "value": "Same literal",
            "source_record_ids": sources[::-1] if reverse_sources else sources[:],
        },
    ]
    with pytest.raises(ValidationError):
        Selection.model_validate(
            {
                **draft_payload()["selection"],
                "source_record_ids": sources,
                "facts": facts,
            }
        )


@pytest.mark.parametrize(
    "field,first,second",
    [
        ("title", "Lantern workshop", "Map workshop"),
        ("start", "2026-06-04T10:00:00+00:00", "2026-06-04T11:00:00+00:00"),
        ("end", "2026-06-04T12:00:00+00:00", "2026-06-04T13:00:00+00:00"),
        ("location", "North hall", "South hall"),
    ],
)
@pytest.mark.parametrize("same_value", [False, True])
def test_selected_event_field_cannot_repeat_even_with_distinct_sources(
    field, first, second, same_value
):
    # Distinct sources must not circumvent the one-entry event-field rule.
    with pytest.raises(ValidationError):
        Selection.model_validate(
            {
                **draft_payload()["selection"],
                "selected_event_id": "fictional-event",
                "source_record_ids": ["source-one", "source-two"],
                "facts": [
                    {
                        "field": field,
                        "value": first,
                        "source_record_ids": ["source-one"],
                    },
                    {
                        "field": field,
                        "value": first if same_value else second,
                        "source_record_ids": ["source-two"],
                    },
                ],
            }
        )


@pytest.mark.parametrize("field", ["start", "end"])
def test_repeated_event_field_rejected_for_equivalent_instants(field):
    with pytest.raises(ValidationError):
        Selection.model_validate(
            {
                **draft_payload()["selection"],
                "selected_event_id": "fictional-event",
                "source_record_ids": ["fictional-event"],
                "facts": [
                    {
                        "field": field,
                        "value": value,
                        "source_record_ids": ["fictional-event"],
                    }
                    for value in (
                        "2026-06-04T10:00:00+00:00",
                        "2026-06-04T12:00:00+02:00",
                    )
                ],
            }
        )


@pytest.mark.parametrize("duplicate_in_fact", [False, True])
def test_duplicate_selected_sources_remain_rejected(duplicate_in_fact):
    with pytest.raises(ValidationError):
        Selection.model_validate(
            {
                **draft_payload()["selection"],
                "source_record_ids": ["contact-zoe"]
                if duplicate_in_fact
                else ["contact-zoe", "contact-zoe"],
                "facts": [
                    {
                        "field": "name",
                        "value": "Zoë",
                        "source_record_ids": ["contact-zoe", "contact-zoe"]
                        if duplicate_in_fact
                        else ["contact-zoe"],
                    }
                ],
            }
        )


@pytest.mark.parametrize("condition", ["baseline", "B", "A+B"])
@pytest.mark.parametrize("status", ["ok", "failed", "unexecuted"])
def test_saved_outcomes_preserve_success_failure_and_unexecuted(condition, status):
    payload = run_payload(condition, status)
    assert RunRow.model_validate_json(json.dumps(payload)).model_dump() == payload


@pytest.mark.parametrize("attempt", [0, 2, -1])
def test_saved_rows_do_not_authorize_retry(attempt):
    with pytest.raises(ValidationError):
        RunRow.model_validate({**run_payload(), "attempt": attempt})


@pytest.mark.parametrize("field", ["schema_version", "attempt"])
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_saved_counters_do_not_coerce_boolean_float_or_text(field, value):
    with pytest.raises(ValidationError):
        RunRow.model_validate_json(json.dumps({**run_payload(), field: value}))


@pytest.mark.parametrize(
    "change",
    [
        {"decision": None},
        {"failure": failure()},
        {"raw_output": None},
        {"candidates": None},
        {"preprocessing_sha256": None},
        {"extraction_result": extraction_payload()},
        {"request_sha256": "0" * 64},
    ],
)
def test_incoherent_success_and_wrong_request_binding_rejected(change):
    with pytest.raises(ValidationError):
        RunRow.model_validate({**run_payload(), **change})


@pytest.mark.parametrize("status", ["failed", "unexecuted"])
def test_failure_requires_detail_and_cannot_keep_decision(status):
    payload = run_payload(status=status)
    for change in ({"failure": None}, {"decision": draft_payload()}):
        with pytest.raises(ValidationError):
            RunRow.model_validate({**payload, **change})
    if status == "unexecuted":
        with pytest.raises(ValidationError):
            RunRow.model_validate({**payload, "raw_output": "not actually unexecuted"})


@pytest.mark.parametrize(
    "mutation", ["text", "message_id", "input_hash", "span", "missing"]
)
def test_assisted_rows_bind_exact_request_and_extraction(mutation):
    payload = deepcopy(run_payload("B"))
    if mutation == "text":
        payload["request"]["text"] += "!"
    elif mutation == "message_id":
        payload["extraction_result"]["message_id"] = "another-request"
    elif mutation == "input_hash":
        payload["extraction_result"]["input_sha256"] = "0" * 64
    elif mutation == "missing":
        payload["extraction_result"] = None
    else:
        payload["extraction_result"]["extraction"]["entities"] = [
            {"type": "PERSON", "text": "Zoë", "start": 7, "end": 10},
        ]
    with pytest.raises(ValidationError):
        RunRow.model_validate(payload)
