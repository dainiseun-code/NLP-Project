"""Fictional public worlds: identity, temporal eligibility and trust boundaries."""

import builtins
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest

from messaging_nlp.context import load_public_dataset, load_request_bundle
from messaging_nlp.contracts import CandidateEvidence, Decision

EARLY = "2026-06-01T08:00:00+00:00"
CUTOFF = "2026-06-02T12:00:00+00:00"
LATE = "2026-06-03T08:00:00+00:00"
FILES = ("requests", "contacts", "calendar", "history", "snapshots")


@pytest.fixture
def world():
    return {
        "requests": [
            {
                "message_id": "request",
                "text": "ask Ren re jam?? 🪕",
                "owner_id": "account",
                "request_time": CUTOFF,
                "time_zone": "UTC",
                "context_snapshot_id": "snapshot",
            }
        ],
        "contacts": [
            {
                "record_id": "contact-v1",
                "contact_id": "ren",
                "version": 1,
                "name": "Ren",
                "aliases": [],
                "relationship": "friend",
                "available_at": EARLY,
                "provenance": "fictional",
            }
        ],
        "calendar": [
            {
                "record_id": "event-v1",
                "event_id": "jam",
                "version": 1,
                "title": "Evening jam",
                "participant_contact_ids": ["ren", "account"],
                "start": LATE,
                "end": None,
                "location": None,
                "status": "scheduled",
                "available_at": EARLY,
                "provenance": "fictional",
            }
        ],
        "history": [
            {
                "record_id": "message",
                "sender_contact_id": "ren",
                "recipient_contact_id": "account",
                "text": "Let's jam!",
                "timestamp": EARLY,
                "available_at": EARLY,
                "provenance": "fictional",
            }
        ],
        "snapshots": [
            {
                "context_snapshot_id": "snapshot",
                "timestamp": CUTOFF,
                "record_ids": ["contact-v1", "event-v1", "message"],
            }
        ],
    }


@pytest.fixture
def export(tmp_path):
    def write(world):
        root = tmp_path / "public"
        root.mkdir(exist_ok=True)
        for name in FILES:
            (root / f"{name}.jsonl").write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in world[name]
                ),
                encoding="utf-8",
            )
        return root

    return write


def bundle(export, world):
    return load_request_bundle(export(world), "request", owner_id="account")


def test_fictional_world_preserves_text_cancellation_and_unknown_facts(export, world):
    world["calendar"][0]["status"] = "cancelled"
    result = bundle(export, world)
    assert result.request.text == "ask Ren re jam?? 🪕"
    assert result.context.calendar[0].status == "cancelled"
    assert result.context.calendar[0].location is None
    assert result.context.calendar[0].end is None


@pytest.mark.parametrize(
    "table", ["requests", "contacts", "calendar", "history", "snapshots"]
)
def test_duplicate_ids_rejected(export, world, table):
    world[table].append(deepcopy(world[table][0]))
    with pytest.raises(ValueError):
        load_public_dataset(export(world), owner_id="account")


@pytest.mark.parametrize("table", ["calendar", "history"])
def test_record_id_namespace_is_global_across_files(export, world, table):
    world[table][0]["record_id"] = "contact-v1"
    with pytest.raises(ValueError):
        load_public_dataset(export(world), owner_id="account")


@pytest.mark.parametrize(
    "mutation", ["request_snapshot", "snapshot_member", "duplicate_member"]
)
def test_snapshot_foreign_keys_and_unique_members(export, world, mutation):
    if mutation == "request_snapshot":
        world["requests"][0]["context_snapshot_id"] = "missing"
    elif mutation == "snapshot_member":
        world["snapshots"][0]["record_ids"].append("missing")
    else:
        world["snapshots"][0]["record_ids"].append("message")
    with pytest.raises(ValueError):
        load_public_dataset(export(world), owner_id="account")


def test_owner_is_explicit_separate_namespace(export, world):
    root = export(world)
    with pytest.raises(TypeError):
        load_public_dataset(root)
    with pytest.raises(ValueError):
        load_public_dataset(root, owner_id="ren")
    with pytest.raises(ValueError):
        load_public_dataset(root, owner_id="unrelated-account")


@pytest.mark.parametrize(
    "table,field",
    [
        ("history", "sender_contact_id"),
        ("history", "recipient_contact_id"),
        ("calendar", "participant_contact_ids"),
    ],
)
def test_endpoint_and_participant_foreign_keys(export, world, table, field):
    world[table][0][field] = ["nonexistent"] if table == "calendar" else "nonexistent"
    with pytest.raises(ValueError):
        load_public_dataset(export(world), owner_id="account")


@pytest.mark.parametrize("field", ["sender_contact_id", "recipient_contact_id"])
def test_null_endpoint_remains_unknown_not_owner(export, world, field):
    world["history"][0][field] = None
    assert getattr(bundle(export, world).context.history[0], field) is None


def test_snapshot_membership_not_union_file_controls_evidence(export, world):
    extra = {**world["history"][0], "record_id": "unlisted", "text": "NOT EVIDENCE"}
    world["history"].append(extra)
    result = bundle(export, world)
    assert [row.record_id for row in result.context.history] == ["message"]
    with pytest.raises(ValueError):
        result.validate_candidates(
            CandidateEvidence(
                contact_ids=[],
                event_ids=[],
                source_record_ids=["unlisted"],
            )
        )


def test_snapshot_must_include_contact_referenced_by_eligible_records(export, world):
    world["snapshots"][0]["record_ids"].remove("contact-v1")
    with pytest.raises(ValueError):
        bundle(export, world)


@pytest.mark.parametrize(
    "table,field,error",
    [
        ("history", "timestamp", "future message timestamp in snapshot"),
        ("history", "available_at", "future information availability in snapshot"),
        ("contacts", "available_at", "future information availability in snapshot"),
        ("calendar", "available_at", "future information availability in snapshot"),
    ],
)
def test_future_listed_information_fails_closed(export, world, table, field, error):
    world[table][0][field] = LATE
    with pytest.raises(ValueError, match=f"^{error}$"):
        bundle(export, world)


@pytest.mark.parametrize(
    "snapshot_time,request_time", [(EARLY, CUTOFF), (CUTOFF, EARLY)]
)
def test_cutoff_is_earlier_of_snapshot_and_request(
    export, world, snapshot_time, request_time
):
    world["snapshots"][0]["timestamp"] = snapshot_time
    world["requests"][0]["request_time"] = request_time
    # All records are already known at EARLY, even when the snapshot is later.
    result = bundle(export, world)
    assert [row.record_id for row in result.context.history] == ["message"]


@pytest.mark.parametrize("snapshot_time,request_time", [(EARLY, LATE), (LATE, EARLY)])
@pytest.mark.parametrize(
    "field,error",
    [
        ("timestamp", "future message timestamp in snapshot"),
        ("available_at", "future information availability in snapshot"),
    ],
)
def test_listed_message_between_two_clocks_fails_closed(
    export, world, snapshot_time, request_time, field, error
):
    world["snapshots"][0]["timestamp"] = snapshot_time
    world["requests"][0]["request_time"] = request_time
    world["history"][0][field] = CUTOFF
    with pytest.raises(ValueError, match=f"^{error}$"):
        bundle(export, world)


@pytest.mark.parametrize("table", ["contacts", "calendar"])
def test_version_currentness_uses_request_cutoff_when_snapshot_is_later(
    export, world, table
):
    world["snapshots"][0]["timestamp"] = LATE
    world["requests"][0]["request_time"] = EARLY
    world[table].append(
        {
            **world[table][0],
            "record_id": "later-version",
            "version": 2,
            "available_at": CUTOFF,
        }
    )
    assert [row.version for row in getattr(bundle(export, world).context, table)] == [1]


def test_current_cancelled_version_never_resurrects_older_active_event(export, world):
    world["calendar"].append(
        {
            **world["calendar"][0],
            "record_id": "cancelled-v2",
            "version": 2,
            "available_at": CUTOFF,
            "status": "cancelled",
        }
    )
    members = world["snapshots"][0]["record_ids"]
    members[members.index("event-v1")] = "cancelled-v2"
    events = bundle(export, world).context.calendar
    assert [(row.record_id, row.status) for row in events] == [
        ("cancelled-v2", "cancelled")
    ]


@pytest.mark.parametrize(
    "table,id_field", [("contacts", "contact_id"), ("calendar", "event_id")]
)
@pytest.mark.parametrize("listed", [False, True])
def test_latest_available_version_searches_all_exports(
    export, world, table, id_field, listed
):
    newer = {
        **world[table][0],
        "record_id": f"{table}-v2",
        "version": 2,
        "available_at": CUTOFF,
    }
    world[table].append(newer)
    if listed:
        members = world["snapshots"][0]["record_ids"]
        members[members.index(world[table][0]["record_id"])] = newer["record_id"]
        result = bundle(export, world)
        rows = getattr(result.context, table)
        assert [(getattr(row, id_field), row.version) for row in rows] == [
            (newer[id_field], 2),
        ]
    else:
        with pytest.raises(ValueError):
            bundle(export, world)


@pytest.mark.parametrize("table", ["contacts", "calendar"])
def test_future_unlisted_newer_version_does_not_invalidate_early_v1(
    export, world, table
):
    world[table].append(
        {
            **world[table][0],
            "record_id": f"{table}-v2",
            "version": 2,
            "available_at": LATE,
        }
    )
    assert [row.version for row in getattr(bundle(export, world).context, table)] == [1]


@pytest.mark.parametrize("table", ["contacts", "calendar"])
def test_duplicate_logical_version_rejected_even_with_unique_record_ids(
    export, world, table
):
    world[table].append({**world[table][0], "record_id": "another-record"})
    with pytest.raises(ValueError):
        load_public_dataset(export(world), owner_id="account")


@pytest.mark.parametrize("table", ["contacts", "calendar"])
def test_backwards_version_availability_rejected(export, world, table):
    world[table][0]["available_at"] = CUTOFF
    world[table].append(
        {
            **world[table][0],
            "record_id": f"{table}-v2",
            "version": 2,
            "available_at": EARLY,
        }
    )
    with pytest.raises(
        ValueError, match="version availability contradicts version order"
    ):
        load_public_dataset(export(world), owner_id="account")


@pytest.mark.parametrize(
    "table,field",
    [
        ("requests", "prior_state"),
        ("contacts", "gold"),
        ("calendar", "expected_action"),
        ("history", "assistant_state"),
        ("snapshots", "gold"),
    ],
)
def test_no_gold_or_prior_state_in_public_schemas(export, world, table, field):
    world[table][0][field] = "secret"
    with pytest.raises(ValueError):
        load_public_dataset(export(world), owner_id="account")


def test_loader_opens_only_five_allowlisted_files_without_traversal(
    export, world, monkeypatch
):
    root = export(world)
    evaluator = root.parent / "evaluator"
    evaluator.mkdir()
    (evaluator / "gold.jsonl").write_text("not JSON: never open", encoding="utf-8")
    (root / "extra.jsonl").write_text("not JSON: never open", encoding="utf-8")
    opened = []
    original_builtin, original_io = builtins.open, io.open

    def guard(original):
        def checked(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                path = Path(file).resolve()
                assert path.parent == root
                assert path.name in {f"{name}.jsonl" for name in FILES}
                opened.append(path.name)
            return original(file, *args, **kwargs)

        return checked

    def no_traversal(*args, **kwargs):
        pytest.fail("public loading must not traverse directories")

    monkeypatch.setattr(builtins, "open", guard(original_builtin))
    monkeypatch.setattr(io, "open", guard(original_io))
    for method in ("glob", "rglob", "iterdir"):
        monkeypatch.setattr(Path, method, no_traversal)
    load_public_dataset(root, owner_id="account")
    assert sorted(opened) == sorted(f"{name}.jsonl" for name in FILES)


@pytest.mark.parametrize("name", FILES)
def test_public_file_symlink_cannot_escape_into_evaluator(export, world, name):
    root = export(world)
    evaluator = root.parent / "evaluator"
    evaluator.mkdir()
    target = evaluator / f"{name}.jsonl"
    path = root / f"{name}.jsonl"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError):
        load_public_dataset(root, owner_id="account")


def references():
    return CandidateEvidence(
        contact_ids=["ren"],
        event_ids=["jam"],
        source_record_ids=["contact-v1", "event-v1", "message"],
    )


def decision():
    return {
        "action": "draft",
        "draft": "Arbitrary wording is NOT semantically verified here.",
        "clarification": None,
        "reason": None,
        "selection": {
            "recipient_contact_ids": ["ren"],
            "selected_event_id": "jam",
            "source_record_ids": ["contact-v1", "event-v1"],
            "facts": [
                {
                    "field": "start",
                    "value": "2026-06-03T10:00:00+02:00",
                    "source_record_ids": ["event-v1"],
                }
            ],
        },
    }


def test_reference_validation_accepts_equivalent_instant_not_semantic_claim(
    export, world
):
    result = bundle(export, world)
    # This layer must not pretend to validate arbitrary prose in draft.
    result.validate_decision(Decision.model_validate(decision()), references())


@pytest.fixture
def fictional_loan_case(export, world):
    world["requests"][0]["text"] = (
        "Ask Neri when they will finish the book Osa offered to lend me."
    )
    world["contacts"] = [
        {
            **world["contacts"][0],
            "record_id": record_id,
            "contact_id": contact_id,
            "name": name,
        }
        for record_id, contact_id, name in [
            ("neri-contact", "neri", "Neri Vale"),
            ("osa-contact", "osa", "Osa Reed"),
        ]
    ]
    world["calendar"] = []
    world["history"] = [
        {
            **world["history"][0],
            "record_id": record_id,
            "sender_contact_id": sender,
            "text": text,
        }
        for record_id, sender, text in [
            (
                "loan-offer",
                "osa",
                "You can borrow my copy of The Copper Orchard after Neri finishes it.",
            ),
            (
                "holder-message",
                "neri",
                "I am reading Osa's copy of The Copper Orchard now.",
            ),
        ]
    ]
    source_ids = ["neri-contact", "osa-contact", "loan-offer", "holder-message"]
    world["snapshots"][0]["record_ids"] = source_ids.copy()
    payload = {
        "action": "draft",
        "draft": "Neri, when will you finish Osa's copy of The Copper Orchard?",
        "clarification": None,
        "reason": None,
        "selection": {
            "recipient_contact_ids": ["neri"],
            "selected_event_id": None,
            "source_record_ids": source_ids.copy(),
            "facts": [
                {
                    "field": field,
                    "value": row[value_field],
                    "source_record_ids": [row["record_id"]],
                }
                for table, field, value_field in [
                    ("contacts", "name", "name"),
                    ("history", "message_text", "text"),
                ]
                for row in world[table]
            ],
        },
    }
    candidates = CandidateEvidence(
        contact_ids=["neri"], event_ids=[], source_record_ids=source_ids
    )
    return bundle(export, world), payload, candidates


def test_literal_multifacts_allow_third_party_name_and_two_history_texts(
    fictional_loan_case,
):
    result, payload, candidates = fictional_loan_case
    selected = Decision.model_validate(payload)
    assert selected.selection.recipient_contact_ids == ["neri"]
    assert [fact.field for fact in selected.selection.facts] == [
        "name",
        "name",
        "message_text",
        "message_text",
    ]
    # Osa is a source-supported third party, not a second selected recipient.
    # Acceptance proves literal support only, not semantic interpretation of roles.
    result.validate_decision(selected, candidates)


@pytest.mark.parametrize(
    "fact_index,invented_value",
    [
        (1, "Osa Lake"),
        (3, "I finished Osa's copy of The Copper Orchard yesterday."),
    ],
)
def test_literal_multifacts_reject_changed_second_name_or_history_text(
    fictional_loan_case, fact_index, invented_value
):
    result, payload, candidates = fictional_loan_case
    result.validate_decision(Decision.model_validate(payload), candidates)
    payload["selection"]["facts"][fact_index]["value"] = invented_value
    changed = Decision.model_validate(payload)
    with pytest.raises(ValueError, match="^selected fact contradicts source$"):
        result.validate_decision(changed, candidates)


@pytest.mark.parametrize(
    "mutation",
    [
        "invented_time",
        "missing_location",
        "wrong_source",
        "recipient",
        "event",
        "missing_contact_source",
    ],
)
def test_reference_validation_rejects_unsupported_literals_and_ids(
    export, world, mutation
):
    payload = decision()
    selection = payload["selection"]
    if mutation == "invented_time":
        selection["facts"][0]["value"] = "2026-06-03T09:00:00+00:00"
    elif mutation == "missing_location":
        selection["facts"][0].update(field="location", value="Invented hall")
    elif mutation == "wrong_source":
        selection["facts"][0]["source_record_ids"] = ["contact-v1"]
    elif mutation == "recipient":
        selection["recipient_contact_ids"] = ["invented"]
    elif mutation == "event":
        selection["selected_event_id"] = "invented"
    else:
        selection["source_record_ids"] = ["event-v1"]
    with pytest.raises(ValueError):
        bundle(export, world).validate_decision(
            Decision.model_validate(payload), references()
        )
