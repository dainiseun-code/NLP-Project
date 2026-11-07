"""The one persisted 4A/4B wrapper is strict and distinguishes observations."""

import pytest

from messaging_nlp.run_contracts import (
    ArtifactRef,
    DecodePolicy,
    ExecutionApproval,
    ResourceLimits,
    RowReceipt,
    TokenReceipt,
    WorkerEvent,
    WorkerRequest,
    load_row_receipt,
)


@pytest.mark.parametrize("path", ["../x", "/x", "x/../y", "x\\y", ".", "x//y"])
def test_artifact_refs_no_path_escape(path):
    with pytest.raises(ValueError):
        ArtifactRef(path=path, sha256="0" * 64, bytes=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", False),
        ("seed", 42.0),
        ("enable_thinking", 0),
        ("max_output_tokens", 255),
        ("call_timeout_seconds", 119),
    ],
)
def test_declared_decoding_cannot_drift_or_coerce(field, value):
    with pytest.raises(ValueError):
        DecodePolicy(**{field: value})


def test_resource_floor_and_rss_not_relaxed():
    with pytest.raises(ValueError):
        ResourceLimits(min_free_disk_bytes=15_000_000_000)
    assert ResourceLimits().min_free_disk_bytes == 16106127360
    assert ResourceLimits(min_free_disk_bytes=10737418240).min_free_disk_bytes == (
        10737418240
    )
    assert ResourceLimits().max_tree_rss_bytes == 3758096384


def test_truthy_number_is_not_execution_consent():
    with pytest.raises(ValueError):
        ExecutionApproval(
            plan_sha256="0" * 64,
            readiness_sha256="1" * 64,
            approved=1,
            approved_by="fictional",
        )


@pytest.mark.parametrize("count", [0, 2049, True, 3.0])
def test_token_receipt_requires_actual_bounded_count(count):
    with pytest.raises(ValueError):
        TokenReceipt(
            messages_sha256="0" * 64,
            template_sha256="0" * 64,
            model_receipt_sha256="0" * 64,
            token_ids_sha256="0" * 64,
            token_count=count,
            token_ids=[1],
        )


@pytest.mark.parametrize("model", [WorkerEvent, WorkerRequest])
def test_protocol_unknown_fields_and_versions_rejected(model):
    with pytest.raises(ValueError):
        model(kind="ready" if model is WorkerEvent else "load", protocol="old")
    with pytest.raises(ValueError):
        model(kind="ready" if model is WorkerEvent else "load", repair=True)


@pytest.fixture
def unexecuted_receipt(tmp_path):
    from run_fixtures import fake_plan

    from messaging_nlp.contracts import RunRow, TechnicalFailure
    from messaging_nlp.io import canonical_json
    from messaging_nlp.preparation import digest
    from messaging_nlp.run_storage import ref, write_new

    plan_path, plan = fake_plan(tmp_path)
    planned = plan.rows[1]
    request = planned.request
    row = RunRow(
        schema_version=1,
        run_id=plan.run_id,
        condition="baseline",
        request=request,
        request_sha256=digest(canonical_json(request.model_dump())),
        context_sha256="0" * 64,
        preprocessing_sha256=None,
        extraction_result=None,
        candidates=None,
        status="unexecuted",
        decision=None,
        failure=TechnicalFailure(
            stage="runner", code="interrupted", message="Fictional stop"
        ),
        raw_output=None,
        attempt=1,
    )
    write_new(tmp_path / "row.json", canonical_json(row.model_dump()))
    receipt = RowReceipt(
        run_id=plan.run_id,
        plan_sha256=digest(plan_path.read_bytes()),
        row_id=planned.row_id,
        ordinal=planned.ordinal,
        condition="baseline",
        message_id=request.message_id,
        disposition="unattempted_stop",
        provider_calls=0,
        dispatch_intent=False,
        dispatched=False,
        generation_started=False,
        generation_completed=False,
        finish_reason=None,
        stop_reason="fictional stop",
        reservation_id=None,
        startup_reservation_id=None,
        charged_seconds=0,
        charged_generations=0,
        elapsed_seconds=0.0,
        run_row=ref(tmp_path, tmp_path / "row.json"),
        resolved_envelope=None,
        raw_output=None,
        bindings={"plan": ref(tmp_path, plan_path)},
        logs={},
        cleanup=None,
    )
    write_new(tmp_path / "receipt.json", canonical_json(receipt.model_dump()))
    return tmp_path, receipt


def test_actual_wrapper_loads_exact_stage_three_runrow(unexecuted_receipt):
    root, _ = unexecuted_receipt
    receipt, row, envelope = load_row_receipt(root, root / "receipt.json")
    assert receipt.disposition == "unattempted_stop"
    assert row.status == "unexecuted"
    assert envelope is None


def test_missing_or_mutated_runrow_not_reconstructed(unexecuted_receipt):
    root, _ = unexecuted_receipt
    (root / "row.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_row_receipt(root, root / "receipt.json")


def test_wrapper_plan_identity_cannot_be_rebound(unexecuted_receipt):
    from messaging_nlp.io import canonical_json

    root, receipt = unexecuted_receipt
    data = receipt.model_dump() | {"plan_sha256": "1" * 64}
    (root / "receipt.json").write_bytes(canonical_json(data))
    with pytest.raises(ValueError, match="plan binding"):
        load_row_receipt(root, root / "receipt.json")


def test_rehashed_row_cannot_change_original_request(unexecuted_receipt):
    from messaging_nlp.io import canonical_json, loads_json
    from messaging_nlp.preparation import digest
    from messaging_nlp.run_storage import ref

    root, receipt = unexecuted_receipt
    data = loads_json((root / "row.json").read_text())
    data["request"]["text"] = "Changed noisy request"
    data["request_sha256"] = digest(canonical_json(data["request"]))
    (root / "row.json").write_bytes(canonical_json(data))
    altered = receipt.model_dump() | {
        "run_row": ref(root, root / "row.json").model_dump()
    }
    (root / "receipt.json").write_bytes(canonical_json(altered))
    with pytest.raises(ValueError, match="original request"):
        load_row_receipt(root, root / "receipt.json")


@pytest.mark.parametrize(
    "changes",
    [
        {"generation_started": True},
        {"dispatched": True},
        {"generation_completed": True},
        {"provider_calls": 1},
        {"attempt": True},
        {"elapsed_seconds": float("inf")},
    ],
)
def test_observed_states_never_fabricated(unexecuted_receipt, changes):
    _, receipt = unexecuted_receipt
    with pytest.raises(ValueError):
        RowReceipt.model_validate(receipt.model_dump() | changes)
