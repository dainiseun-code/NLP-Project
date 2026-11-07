"""Canonical ledger integration on fictional temporary copies, no model calls."""

import json

import pytest
from test_diagnostic_budget import (
    _pin_fifth_authority,
    _register_third_and_fourth,
    diagnostic_ledger,
    register_fifth,
)
from test_repair_authority import (
    fictional_authority,
    fictional_plan,
    register_refs,
    sealed_authority,
    slot_refs,
    write_plan,
)

from messaging_nlp import repair_authority as a
from messaging_nlp.run_budget import BudgetLedgerError


@pytest.fixture
def setup(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    budget = diagnostic_ledger(docs, monkeypatch)
    _register_third_and_fourth(budget)
    _pin_fifth_authority(docs, monkeypatch, budget)
    register_fifth(budget)
    before = json.loads(budget.path.read_text())
    body = sealed_authority(
        tmp_path,
        monkeypatch,
        fictional_authority(
            prior_assistant_versions_sha256=a.digest(
                a.canonical(before["prompt_versions"]["assistant_prompt"])
            )
        ),
    )
    return tmp_path, budget, body, before


def register(root, budget, body, number):
    plan = fictional_plan(body, number)
    _, sha = write_plan(root, number, plan)
    budget.register_versions(
        dict(
            assistant_version=plan["candidate_version"]["version_id"],
            assistant_version_refs=register_refs(plan, number, sha),
        )
    )
    return plan, sha


def test_five_candidates_and_exactly_one_hundred_additional_slots(setup):
    root, budget, body, before = setup
    for number in range(1, 6):
        plan, sha = register(root, budget, body, number)
        for i in range(20):
            identifier = budget.reserve(
                "generation",
                120,
                generations=1,
                refs=slot_refs(plan, number, sha, f"{i:02d}"),
            )
            budget.settle(identifier, "finished", 0, generations=1)
    snapshot = budget.snapshot()
    after = json.loads(budget.path.read_text())
    assert len(after["prompt_versions"]["assistant_prompt"]) == 10
    assert snapshot["caps"]["max_assistant_prompt_versions"] == 10
    assert snapshot["version_remaining"]["assistant_prompt"] == 0
    assert snapshot["usage"]["generations"] == 100
    assert after["opening"] == before["opening"] and after["epochs"] == before["epochs"]
    assert (
        after["prompt_versions"]["assistant_prompt"][:5]
        == before["prompt_versions"]["assistant_prompt"]
    )
    with pytest.raises(BudgetLedgerError):
        budget.reserve(
            "generation",
            120,
            generations=1,
            refs={**slot_refs(plan, 5, sha), "round": "6"},
        )


@pytest.mark.parametrize("status", ["reserved", "finished", "failed", "cancelled"])
def test_existing_slot_never_reopens_for_any_outcome(setup, status):
    root, budget, body, _ = setup
    plan, sha = register(root, budget, body, 1)
    refs = slot_refs(plan, 1, sha)
    identifier = budget.reserve("generation", 120, generations=1, refs=refs)
    if status != "reserved":
        budget.settle(
            identifier,
            status,
            0,
            generations=1 if status == "finished" else 0,
            never_launched=status == "cancelled",
        )
    with pytest.raises(BudgetLedgerError, match="already reserved"):
        budget.reserve("generation", 120, generations=1, refs=refs)


def test_changed_round_plan_blocks_registered_ledger_access(setup):
    root, budget, body, _ = setup
    plan, sha = register(root, budget, body, 1)
    path = root / a.DIRECTORY / "rounds/01/plan.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(BudgetLedgerError, match="hash-bound"):
        budget.snapshot()


def test_unapproved_sixth_identity_stays_exhausted(setup):
    _, budget, _, _ = setup
    with pytest.raises(BudgetLedgerError, match="exhausted"):
        budget.register_versions(
            dict(assistant_version="arbitrary", assistant_version_refs={})
        )


def test_diagnostic_series_scope_cannot_be_used_for_study_rows(setup):
    root, budget, body, _ = setup
    plan, sha = register(root, budget, body, 1)
    with pytest.raises(BudgetLedgerError, match="generation accounting"):
        budget.reserve("row", 120, generations=1, refs=slot_refs(plan, 1, sha))


def test_missing_repair_authority_does_not_reset_history(setup):
    root, budget, body, before = setup
    register(root, budget, body, 1)
    frozen = budget.path.read_bytes()
    (root / "docs" / a.AUTHORITY_NAME).unlink()
    with pytest.raises(BudgetLedgerError, match="repair series authority"):
        budget.snapshot()
    assert budget.path.read_bytes() == frozen
    assert json.loads(frozen)["opening"] == before["opening"]
