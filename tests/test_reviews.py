"""Independent fictional review requirements; no judge, tokenizer or model calls."""

import json
from pathlib import Path

import pytest
from test_runner import deny_processes_and_optional_models, fixture  # noqa: F401

from messaging_nlp.io import canonical_json
from messaging_nlp.preparation import digest
from messaging_nlp.reviews import (
    JudgmentRecord,
    create_review_packet,
    input_digest,
    judge_assignment,
    load_review_binding,
    resolve_reviews,
)
from messaging_nlp.run_storage import ref, write_new


@pytest.fixture
def world(fixture):  # noqa: F811
    f = fixture
    result = f.runner._run_plan(f.path, dependencies=f.dependencies)
    assert result["status"] == "completed"
    root = f.root / "out"
    policy = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/evaluation-policy.json"
        ).read_bytes()
    )
    request = f.validated.dataset.requests[0]
    requirement = dict(
        message_id=request.message_id,
        scenario_id="SECRET_SCENARIO",
        family_id="SECRET_FAMILY",
        category_tags=["SECRET_CATEGORY"],
        human_review_status="pending_not_human_annotation",
        request_text_sha256=digest(request.text.encode()),
        context_snapshot_id=request.context_snapshot_id,
        expected_action="unsupported",
        recipient_contact_ids=[],
        required_meaning=["Explain the no-send limit."],
        forbidden_meaning=["Claim sending was performed."],
        support_alternatives=[[]],
        actual_blockers=[{"kind": "operation_unavailable", "detail": "Draft-only."}],
        normalized_facts=[],
        clarification_slot=None,
        unsupported_reason="sending_not_supported_draft_only",
    )
    path = root / "reviews/one"
    binding = create_review_packet(
        root, root / "rows/000000/receipt.json", requirement, policy, path
    )
    return root, path, binding, policy


def record(
    world, judge_id, *, verdict="pass", quote="Cannot send messages.", session=None
):
    root, path, binding, policy = world
    response = {
        "schema_version": "nlp3-judgment-v1",
        "judge_id": judge_id,
        "packet_sha256": binding.packet.sha256,
        "inputs_sha256": input_digest(binding),
        "dimensions": {
            key: {
                "verdict": verdict,
                "quotes": [quote],
                "omission": None,
                "source_ids": [],
                "rationale": "The explicit sending limit is stated.",
            }
            for key in policy["dimensions"]
        },
    }
    target = path / judge_id
    write_new(target / "response.json", canonical_json(response))
    native = {
        "model": policy["judges"][judge_id],
        "finish": "stop",
        "session_id": session or judge_id,
        "message_id": "fictional-message",
        "response_sha256": digest(canonical_json(response)),
    }
    write_new(target / "native.json", canonical_json(native))
    return JudgmentRecord(
        binding=ref(root, path / "binding.json"),
        response=ref(root, target / "response.json"),
        identity=ref(root, target / "native.json"),
    )


def test_packet_is_anonymous_exact_request_full_context_no_metadata(world):
    root, path, _, _ = world
    _, packet, inputs = load_review_binding(root, path / "binding.json")
    raw = (path / "packet.json").read_text()
    for forbidden in (
        "SECRET_",
        "condition",
        "extractor",
        "extraction",
        "human_review_status",
    ):
        assert forbidden not in raw
    assert packet["request"]["text"] == "Send a message now."
    assert packet["eligible_context"] == inputs["context"]
    assert packet["output"]["decision"]["reason"] == "Cannot send messages."


def test_no_missing_review_is_success(world):
    root, path, _, _ = world
    assert resolve_reviews(root, path / "binding.json", [])["status"] == "unmeasured"
    one = record(world, "judge-sol-v1")
    assert resolve_reviews(root, path / "binding.json", [one])["status"] == "unmeasured"


def test_judge_assignment_contains_only_safe_packet_bindings_and_schema(world):
    root, path, binding, _ = world
    task = judge_assignment(root, path / "binding.json", "judge-sol-v1")
    encoded = canonical_json(task).decode()
    assert task["response_bindings"]["packet_sha256"] == binding.packet.sha256
    assert task["response_bindings"]["inputs_sha256"] == input_digest(binding)
    for forbidden in ("SECRET_", "private/", "condition", "extractor", "preparation"):
        assert forbidden not in encoded
    with pytest.raises(ValueError, match="outside"):
        judge_assignment(root, path / "binding.json", "substitute")


def test_two_bound_independent_reviews_pass_not_syntax_alone(world):
    root, path, _, _ = world
    records = [record(world, j) for j in world[3]["judges"]]
    assert resolve_reviews(root, path / "binding.json", records)["status"] == "pass"


@pytest.mark.parametrize(
    "second,expected", [("fail", "unmeasured"), ("unresolved", "unmeasured")]
)
def test_disagreement_is_preserved_not_majority_or_automatic_pass(
    world, second, expected
):
    root, path, _, _ = world
    records = [
        record(world, "judge-sol-v1"),
        record(world, "judge-deepseek-v1", verdict=second),
    ]
    assert resolve_reviews(root, path / "binding.json", records)["status"] == expected


def test_agreed_incorrect_reason_fails(world):
    root, path, _, _ = world
    records = [record(world, j, verdict="fail") for j in world[3]["judges"]]
    assert resolve_reviews(root, path / "binding.json", records)["status"] == "fail"


@pytest.mark.parametrize(
    "fault", ["stale", "identity", "quote", "source", "dimension", "no_evidence"]
)
def test_supplied_invalid_judgment_fails_closed(world, fault):
    root, path, _, _ = world
    item = record(world, "judge-sol-v1")
    response = json.loads((root / item.response.path).read_bytes())
    native = json.loads((root / item.identity.path).read_bytes())
    first = next(iter(response["dimensions"].values()))
    if fault == "stale":
        response["packet_sha256"] = "0" * 64
    elif fault == "identity":
        native["model"]["id"] = "substitute"
    elif fault == "quote":
        first["quotes"] = ["Invented quote."]
    elif fault == "source":
        first["source_ids"] = ["descriptive-secret-record"]
    elif fault == "dimension":
        response["dimensions"].pop(next(iter(response["dimensions"])))
    else:
        first["quotes"] = []
    response_raw = canonical_json(response)
    (root / item.response.path).write_bytes(response_raw)
    native["response_sha256"] = digest(response_raw)
    (root / item.identity.path).write_bytes(canonical_json(native))
    changed = JudgmentRecord(
        binding=item.binding,
        response=ref(root, root / item.response.path),
        identity=ref(root, root / item.identity.path),
    )
    with pytest.raises(ValueError):
        resolve_reviews(root, path / "binding.json", [changed])


def test_packet_mutation_cannot_be_rebound_silently(world):
    root, path, _, _ = world
    (path / "packet.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_review_binding(root, path / "binding.json")


def test_duplicate_judge_or_same_session_not_independent(world):
    root, path, _, _ = world
    one = record(world, "judge-sol-v1", session="same")
    two = record(world, "judge-deepseek-v1", session="same")
    for records in ([one, one], [one, two]):
        with pytest.raises(ValueError, match="duplicate"):
            resolve_reviews(root, path / "binding.json", records)


def test_separate_written_adjudication_preserves_originals(world):
    root, path, binding, _ = world
    records = [
        record(world, "judge-sol-v1"),
        record(world, "judge-deepseek-v1", verdict="fail"),
    ]
    originals = [(root / r.response.path).read_bytes() for r in records]
    dims = json.loads(originals[0])["dimensions"]
    adj = dict(
        schema_version="nlp3-adjudication-v1",
        packet_sha256=binding.packet.sha256,
        original_judgments=[r.response.sha256 for r in records],
        adjudicator="nlp_orchestrator",
        reason="Exact response states the supported no-send limitation.",
        dimensions=dims,
    )
    write_new(path / "adjudication.json", canonical_json(adj))
    result = resolve_reviews(
        root, path / "binding.json", records, ref(root, path / "adjudication.json")
    )
    assert result["status"] == "pass"
    assert originals == [(root / r.response.path).read_bytes() for r in records]


def test_exclusive_packet_creation_preserves_old_evidence(world):
    root, path, _, policy = world
    original = (path / "packet.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_review_packet(root, root / "rows/000000/receipt.json", {}, policy, path)
    assert (path / "packet.json").read_bytes() == original


def test_full_eligible_records_and_structured_ids_are_aliased_not_prose():
    from test_prompts import (
        INTERNAL_IDS,
        REC_REN,
        REC_SUPPER,
        REN,
        REQUEST_TEXT,
        SUPPER,
        make_packet,
    )

    from messaging_nlp.reviews import _visible

    prep = make_packet()
    requirements = dict(
        expected_action="draft",
        recipient_contact_ids=[REN],
        required_meaning=["Ask Ren about supper."],
        forbidden_meaning=[],
        support_alternatives=[[REC_REN, REC_SUPPER]],
        actual_blockers=[],
        normalized_facts=[
            dict(kind="selected_event", event_id=SUPPER, source_record_ids=[REC_SUPPER])
        ],
        clarification_slot=None,
        unsupported_reason=None,
    )
    packet = _visible(
        dict(
            preparation=prep.model_dump(),
            requirements=requirements,
            policy={
                "dimensions": {},
                "evidence_rule": "quotes",
                "disagreement": "retain",
            },
            parsed={
                "decision": {
                    "selected_event_id": SUPPER,
                    "recipient_contact_ids": [REN],
                    "draft": "Ren, shall we move supper?",
                    "source_record_ids": [REC_REN, REC_SUPPER],
                }
            },
        ),
        "random",
    )
    encoded = canonical_json(packet).decode()
    assert not any(internal in encoded for internal in INTERNAL_IDS)
    assert packet["request"]["text"] == REQUEST_TEXT
    assert len(packet["eligible_context"]["calendar"]) == len(prep.context.calendar)
    assert packet["output"]["decision"]["draft"] == "Ren, shall we move supper?"


def test_review_destination_cannot_escape_via_dotdot_or_symlink(world, tmp_path):
    root, _, _, policy = world
    for destination in (root / ".." / "escape", root / "link" / "escape"):
        if "link" in destination.parts:
            (root / "link").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="retained run root"):
            create_review_packet(
                root, root / "rows/000000/receipt.json", {}, policy, destination
            )
