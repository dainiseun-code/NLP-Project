"""Primary-owned fake end-to-end outcomes, never research/model observations."""

import shutil

import pytest
from test_reproduction import bundle, make_run, write_evaluator

from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.preparation import digest
from messaging_nlp.reproduction import build_bundle, reproduce
from messaging_nlp.reviews import create_review_packet, input_digest
from messaging_nlp.run_contracts import load_row_receipt
from messaging_nlp.run_storage import ref, write_new


def fictional_reviews(run, evaluator):
    requirement = loads_json(evaluator["requirements"].read_text().splitlines()[0])
    policy = loads_json(evaluator["policy"].read_text())
    records = {}
    for receipt_path in sorted((run / "rows").glob("*/receipt.json")):
        receipt, row, _ = load_row_receipt(run, receipt_path)
        directory = run / "reviews" / receipt.row_id
        binding = create_review_packet(
            run, receipt_path, requirement, policy, directory
        )
        records[receipt.row_id] = []
        for judge_id, identity in policy["judges"].items():
            response = dict(
                schema_version="nlp3-judgment-v1",
                judge_id=judge_id,
                packet_sha256=binding.packet.sha256,
                inputs_sha256=input_digest(binding),
                dimensions={
                    key: dict(
                        verdict="pass",
                        quotes=[row.decision.reason],
                        omission=None,
                        source_ids=[],
                        rationale="Fictional explicit no-send limit.",
                    )
                    for key in policy["dimensions"]
                },
            )
            raw = canonical_json(response)
            target = directory / judge_id
            write_new(target / "response.json", raw)
            write_new(
                target / "native.json",
                canonical_json(
                    dict(
                        model=identity,
                        finish="stop",
                        session_id=f"{receipt.row_id}-{judge_id}",
                        message_id="fictional-identity-not-a-real-call",
                        response_sha256=digest(raw),
                    )
                ),
            )
            records[receipt.row_id].append(
                dict(
                    binding=ref(run, directory / "binding.json").model_dump(),
                    response=ref(run, target / "response.json").model_dump(),
                    identity=ref(run, target / "native.json").model_dump(),
                )
            )
    return records


def test_bound_blinded_reviews_runner_and_offline_rescore_relocate_without_weights(
    tmp_path, capsys
):
    from messaging_nlp.cli import main

    evaluator = write_evaluator(tmp_path / "evaluator")
    run, _ = make_run(tmp_path / "source", assisted=True, mode="application")
    records = fictional_reviews(run, evaluator)
    target = tmp_path / "bundle"
    build_bundle(
        run,
        target,
        requirements_path=evaluator["requirements"],
        roster_path=evaluator["roster"],
        gold_path=evaluator["gold"],
        policy_path=evaluator["policy"],
        review_records=records,
    )
    expected = reproduce(target / "manifest.json")
    assert all(row["success"] is True for row in expected["rows"])
    assert expected["summaries"]["overall"]["conditions"]["B"]["planned"] == 1
    assert expected["summaries"]["overall"]["conditions"]["B"]["successful"] == 1
    assert len(expected["extraction"]) == 1  # not duplicated across assisted arms
    assert expected["costs"]["qwen_by_condition"]["B"]["attempted_rows"] == 1
    relocated = tmp_path / "relocated"
    shutil.copytree(target, relocated)
    shutil.rmtree(tmp_path / "source")  # fictional copy only, not research/archive
    assert reproduce(relocated / "manifest.json") == expected
    assert main(["reproduce", "--bundle", str(relocated / "manifest.json")]) == 0
    assert loads_json(capsys.readouterr().out) == expected


def test_post_execution_requirement_change_is_not_a_new_frozen_expectation(tmp_path):
    evaluator = write_evaluator(tmp_path / "evaluator", expected_action="clarify")
    run, _ = make_run(tmp_path / "source", assisted=True, mode="application")
    bundle(run, evaluator, tmp_path / "bundle")
    with pytest.raises(ValueError, match="not frozen"):
        reproduce(tmp_path / "bundle/manifest.json")


def test_pending_commit_wins_even_when_receipt_file_bytes_exist(tmp_path):
    evaluator = write_evaluator(tmp_path / "evaluator")
    run, _ = make_run(tmp_path / "source", assisted=True, mode="application")
    receipt, row, _ = load_row_receipt(run, run / "rows/000000/receipt.json")
    status = loads_json((run / "status.json").read_text())
    status["pending_rows"] = [
        dict(
            row_id=receipt.row_id,
            ordinal=0,
            condition="B",
            message_id="fiction",
            attempt=1,
            provider_calls=1,
            dispatch_intent=True,
            dispatched=True,
            generation_started=True,
            generation_completed=True,
            finish_reason="stop",
            reservation_id=receipt.reservation_id,
            raw_output=row.raw_output,
            run_row_bytes_utf8=canonical_json(row.model_dump()).decode(),
            stop_reason="fsync failed after receipt bytes were visible",
            persistence_errors=["fictional fsync failure"],
            artifacts={},
            unavailable_artifacts=["receipt.json"],
        )
    ]
    (run / "status.json").write_bytes(canonical_json(status))
    bundle(run, evaluator, tmp_path / "bundle")
    result = reproduce(tmp_path / "bundle/manifest.json")
    assert result["rows"][0]["disposition"] == "pending_persistence"
    assert result["rows"][0]["success"] is None
    assert result["costs"]["unattributed_row_ids"] == [receipt.row_id]


def test_unindexed_inserted_artifact_fails_instead_of_becoming_output(tmp_path):
    evaluator = write_evaluator(tmp_path / "evaluator")
    run, _ = make_run(tmp_path / "source", assisted=False, mode="invalid_json")
    bundle(run, evaluator, tmp_path / "bundle")
    (tmp_path / "bundle/run/unindexed.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="inventory differs"):
        reproduce(tmp_path / "bundle/manifest.json")


def test_symlinked_evaluator_input_is_rejected_before_copy(tmp_path):
    evaluator = write_evaluator(tmp_path / "evaluator")
    run, _ = make_run(tmp_path / "source", assisted=False, mode="invalid_json")
    link = tmp_path / "linked-policy.json"
    link.symlink_to(evaluator["policy"])
    evaluator["policy"] = link
    with pytest.raises(ValueError, match="symlink"):
        bundle(run, evaluator, tmp_path / "bundle")


def test_different_installed_scorer_cannot_silently_change_frozen_results(
    tmp_path, monkeypatch
):
    import messaging_nlp.reproduction as consumer

    evaluator = write_evaluator(tmp_path / "evaluator")
    run, _ = make_run(tmp_path / "source", assisted=False, mode="invalid_json")
    bundle(run, evaluator, tmp_path / "bundle")
    wrong = tmp_path / "fictional-other-install"
    wrong.mkdir()
    (wrong / "__init__.py").write_bytes(b"# different installed scoring version")
    monkeypatch.setattr(consumer, "files", lambda package: wrong)
    with pytest.raises(ValueError, match="installed scoring source differs"):
        consumer.reproduce(tmp_path / "bundle/manifest.json")


@pytest.mark.parametrize("bad_refs", [None, "not-a-mapping"])
def test_malformed_budget_refs_fail_as_controlled_evidence_error(tmp_path, bad_refs):
    evaluator = write_evaluator(tmp_path / "evaluator")
    run, _ = make_run(tmp_path / "source", assisted=False, mode="invalid_json")
    path = run / "budget-final.json"
    budget = loads_json(path.read_text())
    budget["reservations"][0]["refs"] = bad_refs
    path.write_bytes(canonical_json(budget))
    bundle(run, evaluator, tmp_path / "bundle")
    with pytest.raises(ValueError, match="budget reservation/refs"):
        reproduce(tmp_path / "bundle/manifest.json")


def test_current_run_startup_overhead_and_provider_observations_are_separate(tmp_path):
    evaluator = write_evaluator(tmp_path / "evaluator")
    run, _ = make_run(tmp_path / "source", assisted=True, mode="application")
    bundle(run, evaluator, tmp_path / "bundle")
    result = reproduce(tmp_path / "bundle/manifest.json")
    overhead = result["costs"]["startup_overhead"]
    assert sorted(row["kind"] for row in overhead) == ["overhead", "startup"]
    assert len({row["reservation_id"] for row in overhead}) == 2
    cost = result["costs"]["qwen_by_condition"]["baseline"]
    assert cost["attempted_rows"] == cost["provider_calls"] == 1
    assert cost["generation_started_rows"] == cost["generation_completed_rows"] == 1


@pytest.mark.parametrize(
    "state",
    [
        "attempted_failure",
        "preparation_blocked",
        "missing_receipt",
        "orphan_adjudication",
    ],
)
def test_noncompleted_work_or_missing_judges_cannot_carry_outcome_review(
    tmp_path, state
):
    evaluator = write_evaluator(tmp_path / "evaluator")
    run, _ = make_run(tmp_path / "source", assisted=False, mode="invalid_json")
    row_id = "row-0" if state == "preparation_blocked" else "row-1"
    if state == "missing_receipt":
        (run / "rows/000001/receipt.json").unlink()
    reference = ref(run, run / "plan.json").model_dump()
    kwargs = (
        dict(adjudications={row_id: reference})
        if state == "orphan_adjudication"
        else dict(
            review_records={
                row_id: [
                    dict(binding=reference, response=reference, identity=reference)
                ]
            }
        )
    )
    build_bundle(
        run,
        tmp_path / "bundle",
        requirements_path=evaluator["requirements"],
        roster_path=evaluator["roster"],
        gold_path=evaluator["gold"],
        policy_path=evaluator["policy"],
        **kwargs,
    )
    with pytest.raises(ValueError, match="cannot carry|cannot replace"):
        reproduce(tmp_path / "bundle/manifest.json")
