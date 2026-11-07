"""Fictional revision authority over a tiny, already attempted study."""

import copy
import hashlib
import json
from pathlib import Path

from run_fixtures import fake_plan

from messaging_nlp import run_revision, stage5
from messaging_nlp.encoder_cache import retained_identity
from messaging_nlp.io import canonical_json
from messaging_nlp.run_budget import BudgetLedger
from messaging_nlp.run_contracts import FrozenFile
from messaging_nlp.run_storage import ref


def revision_fixture(root, monkeypatch):
    path, plan = fake_plan(root, assisted=True)
    policy_path = root / "configs/study-policy.json"
    policy = json.loads(policy_path.read_text())
    policy["data"]["public_root"] = "public"
    policy["data"]["roster"] = "roster.jsonl"
    policy["temporal"]["timezone_identity"]["tzif_path"] = "UTC.tzif"
    policy_path.write_bytes(canonical_json(policy))
    for name in (
        "configs/evaluation-policy.json",
        "docs/stage4b-evaluation-contract.md",
    ):
        (root / name).write_bytes(Path(name).read_bytes())
    spending = plan.spending.model_copy(
        update={
            "extraction_version": "retained-xlmr-" + retained_identity().digest()[:16],
        }
    )
    inventory = {item.path: item for item in plan.frozen_files}
    for name in (
        "configs/study-policy.json",
        "configs/evaluation-policy.json",
        "docs/stage4b-evaluation-contract.md",
    ):
        inventory[name] = FrozenFile(
            **ref(root, root / name).model_dump(), role="other"
        )
    plan = plan.model_copy(
        update={"spending": spending, "frozen_files": list(inventory.values())}
    )
    path.write_bytes(canonical_json(plan.model_dump()))
    budget = BudgetLedger(
        root / "runs/study-state/budget.json",
        root / "docs/stage1-budget-opening.json",
        stage5.OPENING_SHA256,
        reset_path=root / "docs/budget-usage-reset-20261027.json",
        reset_sha256=stage5.RESET_SHA256,
    )
    budget.register_versions(plan.spending)
    for row in plan.rows:
        entry = budget.reserve(
            "row",
            120,
            1,
            refs={
                "run_id": plan.run_id,
                "attempt_key": run_revision.attempt_key(
                    row.request.model_dump(), row.condition
                ),
            },
        )
        budget.settle(entry, "finished", 2, 1)
    store = root / "history"
    store.mkdir()
    (store / "budget.json").write_bytes(budget.path.read_bytes())
    (store / "status.json").write_bytes(
        canonical_json(
            dict(
                cleanup_verified=True,
                pending_rows=[],
                persistence_errors=[],
                rows_saved=3,
                plan_sha256=ref(root, path).sha256,
            )
        )
    )
    (store / "evidence.json").write_bytes(canonical_json({"files": []}))
    body = copy.deepcopy(json.loads(Path(run_revision.AUTHORITY_PATH).read_text()))
    body["predecessor_budget"] = ref(root, store / "budget.json").model_dump()
    body["predecessor_plan"] = ref(root, path).model_dump()
    body["predecessor_status"] = ref(root, store / "status.json").model_dump()
    body["protected_evidence"] = ref(root, store / "evidence.json").model_dump()
    body["scope"]["planned_rows"] = 3
    authority_path = root / "docs/revision.json"
    authority_path.write_bytes(canonical_json(body))
    sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    monkeypatch.setattr(run_revision, "AUTHORITY_SHA256", sha)
    authority = ref(root, authority_path)

    def open_budget(*, apply=False):
        return BudgetLedger(
            budget.path,
            root / "docs/stage1-budget-opening.json",
            stage5.OPENING_SHA256,
            reset_path=root / "docs/budget-usage-reset-20261027.json",
            reset_sha256=stage5.RESET_SHA256,
            revision_path=authority_path,
            revision_sha256=sha,
            apply_revision_reset=apply,
        )

    def row_refs(row=plan.rows[0]):
        return dict(
            run_id=body["run_id"],
            experiment_id=body["experiment_id"],
            revision_authority_sha256=sha,
            attempt_key=run_revision.attempt_key(
                row.request.model_dump(), row.condition
            ),
        )

    materialized = dict(
        record_budget=8,
        rows=[
            dict(
                index=0,
                message_id="fiction",
                baseline=plan.rows[1].preparation.model_dump(),
                shared=plan.rows[0].preparation.model_dump(),
                assisted_failure=None,
                prompts={
                    r.condition: dict(id=r.row_id, prompt=r.prompt.model_dump())
                    for r in plan.rows
                },
            )
        ],
        token_items=[
            dict(
                id=r.row_id,
                payload=dict(messages=json.loads((root / r.prompt.path).read_text())),
            )
            for r in plan.rows
        ],
    )
    tokenized = dict(
        status="completed",
        cleanup_verified=True,
        outcomes=[
            dict(
                id=r.row_id,
                token_receipt=r.token_receipt.model_dump(),
                token_count=3,
                disposition="completed",
                result_status="fits",
            )
            for r in plan.rows
        ],
    )
    return dict(
        plan=plan,
        path=path,
        budget=budget,
        open_budget=open_budget,
        row_refs=row_refs,
        authority=authority,
        materialized=materialized,
        tokenized=tokenized,
    )
