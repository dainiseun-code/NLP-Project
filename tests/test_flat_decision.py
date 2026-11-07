"""Fictional behavioral checks for the flat model-facing decision.

The model returns action/recipient/event/draft only. Python derives record
citations from eligible candidates. These are software checks, not model-quality
or semantic-success evidence.
"""

import hashlib
import json
from pathlib import Path
from zoneinfo import TZPATH

import pytest
import test_application as fiction

from messaging_nlp.contracts import ModelDecision
from messaging_nlp.temporal import TimezoneBinding


@pytest.fixture
def binding(tmp_path):
    source = next(
        (Path(p) / "UTC" for p in TZPATH if (Path(p) / "UTC").is_file()), None
    )
    assert source is not None, "fixture requires a system UTC TZif"
    content = source.read_bytes()
    path = tmp_path / "fictional-UTC.tzif"
    path.write_bytes(content)
    return TimezoneBinding("UTC", path, hashlib.sha256(content).hexdigest())


def flat(packet, draft, *, event=True, action="draft", clarification=None, reason=None):
    contact = fiction.alias_for(packet, "contact", "descriptive-ren-private")
    return {
        "action": action,
        "recipient_contact_ids": [contact] if action == "draft" else [],
        "selected_event_id": (
            fiction.alias_for(packet, "event", "descriptive-supper-private")
            if action == "draft" and event
            else None
        ),
        "draft": draft if action == "draft" else None,
        "clarification": clarification,
        "reason": reason,
    }


@pytest.fixture
def prepared(tmp_path, binding):
    dataset, inputs = fiction.prepare(tmp_path, binding)
    return dataset, inputs, tmp_path, binding


def test_flat_draft_derives_contact_and_event_records(prepared):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(packet, "Ren, supper starts at three pm.")
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "ok"
    decision = result.row.decision
    assert decision.action == "draft"
    assert decision.selection.recipient_contact_ids == ["descriptive-ren-private"]
    assert decision.selection.selected_event_id == "descriptive-supper-private"
    assert {fact.field for fact in decision.selection.facts} == {
        "name",
        "title",
        "start",
    }
    assert result.envelope.mentions == []
    assert result.row.raw_output == json.dumps(payload)


@pytest.mark.parametrize("action", ["clarify", "unsupported"])
def test_flat_non_draft_actions_are_accepted(prepared, action):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(
        packet,
        None,
        action=action,
        clarification="Which supper?" if action == "clarify" else None,
        reason="Fictional blocker",
    )
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "ok"
    assert result.row.decision.action == action
    assert result.row.decision.selection is None


def test_flat_wrong_date_is_rejected(prepared):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(packet, "Ren, supper is today at three pm.")
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "failed"
    assert "contradicts" in result.row.failure.message


def test_flat_wrong_clock_is_rejected(prepared):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(packet, "Ren, supper is tomorrow at four pm.")
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "failed"


def test_flat_correct_relative_date_is_accepted(prepared):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(packet, "Ren, supper is tomorrow at three pm.")
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "ok"


def test_flat_abbreviated_month_matches_selected_event(prepared):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(packet, "Ren, supper is Oct 27 at three pm.")
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "ok"


def test_flat_proposal_clock_is_not_a_current_assertion(prepared):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(packet, "Ren, could we move supper to four pm?")
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "ok"


def test_flat_temporal_claim_without_selected_event_is_rejected(prepared):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(packet, "Ren, see you tomorrow at three pm.", event=False)
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "failed"


def test_flat_unknown_alias_is_rejected(prepared):
    dataset, inputs, tmp_path, binding = prepared
    packet = fiction.packet(dataset, inputs, tmp_path, binding)
    payload = flat(packet, "Ren, supper starts at three pm.")
    payload["recipient_contact_ids"] = ["not-a-displayed-contact"]
    result, _ = fiction.execute(
        dataset, inputs, tmp_path, binding, raw=json.dumps(payload)
    )
    assert result.row.status == "failed"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "action": "draft",
            "recipient_contact_ids": ["x"],
            "selected_event_id": None,
            "draft": None,
            "clarification": None,
            "reason": None,
        },
        {
            "action": "draft",
            "recipient_contact_ids": [],
            "selected_event_id": None,
            "draft": "Hi",
            "clarification": None,
            "reason": None,
        },
    ],
)
def test_flat_action_payloads_stay_coherent(payload):
    with pytest.raises(Exception):
        ModelDecision.model_validate(payload)


def test_flat_non_draft_accepts_null_recipient_metadata():
    decision = ModelDecision.model_validate(
        {
            "action": "unsupported",
            "recipient_contact_ids": None,
            "selected_event_id": None,
            "draft": None,
            "clarification": None,
            "reason": "Cannot send immediately.",
        }
    )
    assert decision.recipient_contact_ids is None


def test_flat_contract_rejects_duplicate_recipients():
    with pytest.raises(Exception):
        ModelDecision.model_validate(
            {
                "action": "draft",
                "recipient_contact_ids": ["x", "x"],
                "selected_event_id": None,
                "draft": "Hi",
                "clarification": None,
                "reason": None,
            }
        )


@pytest.mark.parametrize(
    "missing", ["action", "recipient_contact_ids", "draft", "clarification", "reason"]
)
def test_flat_contract_requires_every_key(missing):
    payload = {
        "action": "draft",
        "recipient_contact_ids": ["x"],
        "selected_event_id": None,
        "draft": "Hi",
        "clarification": None,
        "reason": None,
    }
    payload.pop(missing)
    with pytest.raises(Exception):
        ModelDecision.model_validate(payload)
