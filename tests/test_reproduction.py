"""Offline reproduction bundle sealing and the model-free consumer (Stage 4B).

All fixtures are fictional and produced by the sealed 4A runner with a fake
child. No test opens a model, imports an optional runtime, reads research data,
calls a reviewer, or mutates Git.
"""

import shutil
from pathlib import Path

import pytest

from messaging_nlp.application_types import PreparationPacket
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.preparation import digest
from messaging_nlp.reproduction import (
    MANIFEST_SCHEMA_VERSION,
    ReproductionManifest,
    _crosscheck_binding_inputs,
    _read_bundle,
    build_bundle,
    reproduce,
)
from messaging_nlp.run_contracts import load_row_receipt

TEXT = "Send a note now."


def write_evaluator(root, *, expected_action="unsupported"):
    """Fictional roster/requirements/gold/policy matching fake_plan's request."""
    root.mkdir(parents=True, exist_ok=True)
    roster = [
        {
            "message_id": "fiction",
            "scenario_id": "scenario-1",
            "family_id": "family-1",
            "variant_type": "clean",
            "category_tags": [],
            "text_sha256": digest(TEXT.encode()),
            "condition_order": ["B", "baseline", "A+B"],
            "human_review_status": "pending_not_human_annotated",
        }
    ]
    requirements = [
        {
            "schema_version": "nlp3-task-requirement-v1",
            "message_id": "fiction",
            "scenario_id": "scenario-1",
            "family_id": "family-1",
            "variant_type": "clean",
            "category_tags": [],
            "human_review_status": "pending_not_human_annotation",
            "request_text_sha256": digest(TEXT.encode()),
            "context_snapshot_id": "snapshot",
            "expected_action": expected_action,
            "recipient_contact_ids": [],
            "required_meaning": [],
            "forbidden_meaning": [],
            "support_alternatives": [[]],
            "actual_blockers": [],
            "normalized_facts": [],
            "clarification_slot": None,
            "unsupported_reason": "fictional no-send",
        }
    ]
    policy = loads_json(
        (
            Path(__file__).resolve().parents[1] / "configs/evaluation-policy.json"
        ).read_text()
    )
    paths = {}
    for name, rows in (("roster", roster), ("requirements", requirements)):
        path = root / f"{name}.jsonl"
        path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
        paths[name] = path
    gold = root / "gold.jsonl"
    gold.write_bytes(b"")
    paths["gold"] = gold
    policy_path = root / "policy.json"
    policy_path.write_bytes(canonical_json(policy))
    paths["policy"] = policy_path
    return paths


def make_run(base, *, assisted, mode, expected_action="unsupported"):
    from run_fixtures import fake_plan
    from test_run_plan import permit
    from test_run_process import dependencies

    from messaging_nlp.runner import _run_plan

    write_evaluator(base / "eval", expected_action=expected_action)
    plan_path, plan = fake_plan(base, assisted=assisted)
    permit(plan_path, plan, approval=False)
    _run_plan(plan_path, approve_run=True, dependencies=dependencies(mode))
    return base / plan.output_path, plan


def minimal_run_root(base):
    """A directory with the minimum build_bundle requires (plan.json + frozen/)."""
    root = base / "run_root"
    (root / "frozen").mkdir(parents=True)
    (root / "plan.json").write_bytes(canonical_json({"schema_version": "dummy"}))
    return root


def bundle(run_root, evaluator, destination):
    return build_bundle(
        run_root,
        destination,
        requirements_path=evaluator["requirements"],
        roster_path=evaluator["roster"],
        gold_path=evaluator["gold"],
        policy_path=evaluator["policy"],
    )


@pytest.fixture
def evaluator(tmp_path):
    # Function-scoped: mutation tests rewrite these files and must not leak.
    return write_evaluator(tmp_path / "eval")


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    return make_run(
        tmp_path_factory.mktemp("completed"), assisted=True, mode="application"
    )


@pytest.fixture(scope="module")
def failed_run(tmp_path_factory):
    return make_run(
        tmp_path_factory.mktemp("failed"), assisted=False, mode="invalid_json"
    )


def ref_dict(run_root, relative, data):
    path = run_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "sha256": digest(data), "bytes": len(data)}


# --- build_bundle -----------------------------------------------------------


def test_build_bundle_seals_hash_bound_manifest(completed_run, evaluator, tmp_path):
    run_root, _ = completed_run
    destination = tmp_path / "bundle"
    manifest = bundle(run_root, evaluator, destination)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["run_root"] == "run"
    assert manifest["plan"]["path"] == "run/plan.json"
    assert manifest["requirements"]["path"] == "evaluator/requirements.jsonl"
    assert manifest["roster"]["path"] == "evaluator/roster.jsonl"
    assert manifest["gold"]["path"] == "evaluator/gold.jsonl"
    assert manifest["policy"]["path"] == "evaluator/policy.json"
    assert manifest["preparation_costs"] is None
    paths = [item["path"] for item in manifest["files"]]
    assert paths == sorted(paths) and len(paths) == len(set(paths))
    assert "run/plan.json" in paths and "run/status.json" in paths
    assert any(path.startswith("run/frozen/") for path in paths)
    assert any(
        path.startswith("run/rows/") and path.endswith("/receipt.json")
        for path in paths
    )
    for item in manifest["files"]:
        data = (destination / item["path"]).read_bytes()
        assert digest(data) == item["sha256"] and len(data) == item["bytes"]
    raw = (destination / "manifest.json").read_bytes()
    parsed = ReproductionManifest.model_validate(loads_json(raw.decode()))
    assert canonical_json(parsed.model_dump()) == raw


def test_build_bundle_requires_fresh_destination(evaluator, tmp_path):
    root = minimal_run_root(tmp_path)
    destination = tmp_path / "bundle"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        bundle(root, evaluator, destination)


def test_build_bundle_rejects_destination_inside_source(evaluator, tmp_path):
    root = minimal_run_root(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        bundle(root, evaluator, root / "bundle")


def test_build_bundle_rejects_symlink_in_run_root(evaluator, tmp_path):
    root = minimal_run_root(tmp_path)
    (root / "link.txt").symlink_to(root / "plan.json")
    with pytest.raises(ValueError, match="symlink"):
        bundle(root, evaluator, tmp_path / "bundle")


def test_build_bundle_rejects_symlinked_directory(evaluator, tmp_path):
    root = minimal_run_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.json").write_bytes(b"{}")
    (root / "linked-dir").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        bundle(root, evaluator, tmp_path / "bundle")


def test_build_bundle_rejects_live_model_file(evaluator, tmp_path):
    root = minimal_run_root(tmp_path)
    (root / "frozen" / "weights.safetensors").write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="live model"):
        bundle(root, evaluator, tmp_path / "bundle")


def test_build_bundle_remaps_review_records_and_adjudications(evaluator, tmp_path):
    root = minimal_run_root(tmp_path)
    binding = ref_dict(root, "reviews/row-0/binding.json", b'{"binding":true}')
    response = ref_dict(root, "reviews/row-0/response.json", b'{"response":true}')
    identity = ref_dict(root, "reviews/row-0/identity.json", b'{"identity":true}')
    adjudication = ref_dict(
        root, "reviews/row-0/adjudication.json", b'{"adjudication":true}'
    )
    manifest = build_bundle(
        root,
        tmp_path / "bundle",
        requirements_path=evaluator["requirements"],
        roster_path=evaluator["roster"],
        gold_path=evaluator["gold"],
        policy_path=evaluator["policy"],
        review_records={
            "row-0": [{"binding": binding, "response": response, "identity": identity}]
        },
        adjudications={"row-0": adjudication},
    )
    record = manifest["review_records"]["row-0"][0]
    assert record["binding"]["path"] == "run/reviews/row-0/binding.json"
    assert record["response"]["path"] == "run/reviews/row-0/response.json"
    assert record["identity"]["path"] == "run/reviews/row-0/identity.json"
    assert (
        manifest["adjudications"]["row-0"]["path"]
        == "run/reviews/row-0/adjudication.json"
    )
    paths = {item["path"] for item in manifest["files"]}
    assert record["binding"]["path"] in paths


def test_build_bundle_remap_rejects_foreign_reference(evaluator, tmp_path):
    root = minimal_run_root(tmp_path)
    foreign = ref_dict(tmp_path, "elsewhere.json", b'{"foreign":true}')
    with pytest.raises(ValueError, match="not a copied run artifact"):
        build_bundle(
            root,
            tmp_path / "bundle",
            requirements_path=evaluator["requirements"],
            roster_path=evaluator["roster"],
            gold_path=evaluator["gold"],
            policy_path=evaluator["policy"],
            review_records={
                "row-0": [
                    {
                        "binding": foreign,
                        "response": foreign,
                        "identity": foreign,
                    }
                ]
            },
        )


def test_build_bundle_passes_preparation_costs(evaluator, tmp_path):
    root = minimal_run_root(tmp_path)
    costs = tmp_path / "costs.json"
    costs.write_bytes(canonical_json({"preparation_seconds": 4}))
    manifest = build_bundle(
        root,
        tmp_path / "bundle",
        requirements_path=evaluator["requirements"],
        roster_path=evaluator["roster"],
        gold_path=evaluator["gold"],
        policy_path=evaluator["policy"],
        preparation_costs_path=costs,
    )
    assert manifest["preparation_costs"]["path"] == "evaluator/preparation-costs.json"


# --- reproduce validation and reconciliation --------------------------------


def test_reproduce_classifies_failed_run(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    manifest_path = tmp_path / "bundle"
    bundle(run_root, evaluator, manifest_path)
    result = reproduce(manifest_path / "manifest.json")
    assert result["status"] == "reproduced"
    rows = {row["row_id"]: row for row in result["rows"]}
    assert rows["row-0"]["disposition"] == "preparation_blocked"
    assert rows["row-1"]["disposition"] == "attempted_failure"
    assert rows["row-2"]["disposition"] == "preparation_blocked"
    assert rows["row-1"]["success"] is False
    assert rows["row-0"]["success"] is None and rows["row-2"]["success"] is None
    assert result["extraction"]["fiction"]["status"] == "unmeasured"
    assert result["coverage"]["fiction"]["B"]["status"] == "unmeasured"
    assert result["coverage"]["fiction"]["baseline"]["status"] == "measured"
    baseline = result["costs"]["qwen_by_condition"]["baseline"]
    assert baseline["attempted_rows"] == 1 and baseline["charged_generations"] == 1
    assert result["costs"]["qwen_by_condition"]["B"]["charged_generations"] == 0


def test_reproduce_detects_artifact_mutation(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    destination = tmp_path / "bundle"
    bundle(run_root, evaluator, destination)
    target = destination / "run" / "status.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="inventory hash mismatch"):
        reproduce(destination / "manifest.json")


def test_reproduce_rejects_escaping_bundle_path(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "a.txt").write_bytes(b"x")
    with pytest.raises(ValueError):
        _read_bundle(root, "../a.txt")
    with pytest.raises(ValueError):
        _read_bundle(root, "/etc/hosts")


def test_reproduce_rejects_symlink_in_bundle(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    destination = tmp_path / "bundle"
    bundle(run_root, evaluator, destination)
    requirements = destination / "evaluator" / "requirements.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"[]")
    requirements.unlink()
    requirements.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        reproduce(destination / "manifest.json")


def test_reproduce_rejects_missing_requirement(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    destination = tmp_path / "bundle"
    evaluator["requirements"].write_bytes(b"")
    bundle(run_root, evaluator, destination)
    with pytest.raises(ValueError, match="requirements"):
        reproduce(destination / "manifest.json")


def test_reproduce_rejects_request_hash_mismatch(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    destination = tmp_path / "bundle"
    wrong = [
        {
            "message_id": "fiction",
            "scenario_id": "scenario-1",
            "family_id": "family-1",
            "variant_type": "clean",
            "request_text_sha256": "0" * 64,
            "context_snapshot_id": "snapshot",
        }
    ]
    evaluator["requirements"].write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in wrong)
    )
    bundle(run_root, evaluator, destination)
    with pytest.raises(ValueError, match="request hash"):
        reproduce(destination / "manifest.json")


def test_reproduce_rejects_roster_order_mismatch(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    destination = tmp_path / "bundle"
    roster = [
        {
            "message_id": "fiction",
            "scenario_id": "scenario-1",
            "family_id": "family-1",
            "variant_type": "clean",
            "text_sha256": digest(TEXT.encode()),
            "condition_order": ["baseline", "B", "A+B"],
        }
    ]
    evaluator["roster"].write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in roster)
    )
    bundle(run_root, evaluator, destination)
    with pytest.raises(ValueError, match="roster order"):
        reproduce(destination / "manifest.json")


def test_reproduce_missing_receipt_and_pending(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    copied = tmp_path / "run"
    shutil.copytree(run_root, copied)
    (copied / "rows" / "000001" / "receipt.json").unlink()
    destination = tmp_path / "bundle"
    bundle(copied, evaluator, destination)
    result = reproduce(destination / "manifest.json")
    row = next(r for r in result["rows"] if r["row_id"] == "row-1")
    assert row["disposition"] == "missing_receipt" and row["success"] is None


def test_reproduce_pending_persistence(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    copied = tmp_path / "run"
    shutil.copytree(run_root, copied)
    (copied / "rows" / "000001" / "receipt.json").unlink()
    status_path = copied / "status.json"
    status = loads_json(status_path.read_bytes().decode())
    status["pending_rows"] = [
        {
            "row_id": "row-1",
            "ordinal": 1,
            "condition": "baseline",
            "message_id": "fiction",
            "attempt": 1,
            "provider_calls": 0,
            "dispatch_intent": False,
            "dispatched": False,
            "generation_started": False,
            "generation_completed": False,
            "finish_reason": None,
            "reservation_id": None,
            "raw_output": None,
            "run_row_bytes_utf8": None,
            "stop_reason": "fictional incomplete receipt",
            "persistence_errors": ["fictional"],
            "artifacts": {},
            "unavailable_artifacts": ["receipt.json"],
        }
    ]
    status_path.write_bytes(canonical_json(status))
    destination = tmp_path / "bundle"
    bundle(copied, evaluator, destination)
    result = reproduce(destination / "manifest.json")
    row = next(r for r in result["rows"] if r["row_id"] == "row-1")
    assert row["disposition"] == "pending_persistence" and row["success"] is None


def test_reproduce_rejects_foreign_receipt(failed_run, evaluator, tmp_path):
    run_root, _ = failed_run
    copied = tmp_path / "run"
    shutil.copytree(run_root, copied)
    foreign = copied / "rows" / "000999" / "receipt.json"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"{}")
    destination = tmp_path / "bundle"
    bundle(copied, evaluator, destination)
    with pytest.raises(ValueError, match="foreign receipt"):
        reproduce(destination / "manifest.json")


# --- completed-row scoring (lazy evaluation kernels) ------------------------


def test_reproduce_scores_completed_rows(completed_run, tmp_path):
    evaluator = write_evaluator(tmp_path / "eval")
    run_root, _ = completed_run
    destination = tmp_path / "bundle"
    bundle(run_root, evaluator, destination)
    result = reproduce(destination / "manifest.json")
    rows = {row["row_id"]: row for row in result["rows"]}
    for row in rows.values():
        assert row["disposition"] == "completed"
        assert row["technical_ok"] is True
        assert row["structural"] == {
            "action": "pass",
            "recipient": "not_applicable",
            "selected_event": "not_applicable",
            "event_start": "not_applicable",
        }
        assert row["semantic"]["status"] == "unmeasured"
        assert row["success"] is None  # semantic unmeasured, never credited
    assert result["extraction"]["fiction"]["status"] == "measured"
    assert result["extraction"]["fiction"]["tp"] == 0
    assert result["coverage"]["fiction"]["B"] == {
        "status": "measured",
        "complete": True,
        "best_fraction": 1.0,
    }


def test_reproduce_structural_failure_is_failure(tmp_path):
    evaluator = write_evaluator(tmp_path / "eval", expected_action="draft")
    run_root, _ = make_run(
        tmp_path / "run-source",
        assisted=True,
        mode="application",
        expected_action="draft",
    )
    destination = tmp_path / "bundle"
    bundle(run_root, evaluator, destination)
    result = reproduce(destination / "manifest.json")
    rows = {row["row_id"]: row for row in result["rows"]}
    assert all(row["structural"]["action"] == "fail" for row in rows.values())
    assert all(row["success"] is False for row in rows.values())


# --- binding crosscheck -----------------------------------------------------


def _completed_row_and_prep(run_root):
    receipt_path = sorted((run_root / "rows").glob("*/receipt.json"))[0]
    receipt, row, envelope = load_row_receipt(run_root, receipt_path)
    preparation = PreparationPacket.model_validate(
        loads_json(
            (run_root / receipt.bindings["preparation"].path).read_bytes().decode()
        )
    )
    return row, envelope, preparation


def test_crosscheck_binding_rejects_stale_inputs(completed_run):
    run_root, _ = completed_run
    row, envelope, preparation = _completed_row_and_prep(run_root)
    requirement = {
        "message_id": "fiction",
        "expected_action": "unsupported",
        "recipient_contact_ids": [],
        "support_alternatives": [[]],
    }
    policy = {
        "judges": {"judge-1": "gpt-6.1-sol", "judge-2": "deepseek-v4-pro"},
        "dimensions": {"expected_action": "rubric"},
    }
    inputs = {
        "request": row.request.model_dump(),
        "context": preparation.context.model_dump(),
        "preparation": preparation.model_dump(),
        "requirements": requirement,
        "policy": policy,
        "raw": row.raw_output,
        "parsed": envelope.model_dump(),
    }
    # Matching inputs pass the crosscheck.
    _crosscheck_binding_inputs(inputs, requirement, policy, row, envelope, preparation)
    # A stale requirement must raise rather than silently rescore.
    stale = dict(requirement, expected_action="clarify")
    with pytest.raises(ValueError, match="requirements"):
        _crosscheck_binding_inputs(inputs, stale, policy, row, envelope, preparation)
    stale_inputs = dict(inputs, raw="tampered raw output")
    with pytest.raises(ValueError, match="raw output"):
        _crosscheck_binding_inputs(
            stale_inputs, requirement, policy, row, envelope, preparation
        )
