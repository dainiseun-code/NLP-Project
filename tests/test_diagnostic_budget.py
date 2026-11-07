"""Behavioral tests for the diagnostic extension to the budget ledger.

Derived from ``docs/stage6-controlled-diagnostic-contract.md`` and the
``BudgetLedger`` integration in ``src/messaging_nlp/run_budget.py``. Every ledger
lives under ``tmp_path``; the fictional diagnostic authority sits beside a copy
of the retained reset authority and its digest is bound to
``diagnostic_authority.AUTHORITY_SHA256`` via ``monkeypatch``. No model,
tokenizer, Torch/MLX import, subprocess, network use or canonical budget is
touched.

Under test:

* diagnostic generations reserve each of the twenty slots once with the exact
  ``diagnostic_id``/``run_id``/``diagnostic_slot``/``messages_sha256`` refs,
  bounded to ``0 < seconds <= 120`` and exactly one generation;
* a duplicate slot cannot be replayed under a new run id, a cancellation or a
  fresh boot/process;
* foreign slots, ids, prompts and malformed/absent authorities fail closed;
* the opening cap of four and old epochs/reservations stay unchanged, the
  snapshot version cap becomes five only after the exact fifth, extraction
  stays two, and no sixth version or arbitrary cap inflation is admitted;
* ordinary row retry/restart gates remain untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_diagnostic_authority import (
    AUTHORITY_NAME,
    DIAGNOSTIC_ID,
    FIFTH_ID,
    fictional_authority,
    slot_messages_sha256,
    write_authority,
)

from messaging_nlp import diagnostic_authority, run_budget

ROOT = Path(__file__).resolve().parents[1]
OPENING_PATH = ROOT / "docs" / "stage1-budget-opening.json"
RESET_PATH = ROOT / "docs" / "budget-usage-reset-20261027.json"
OPENING_SHA256 = "3f7a172dd44af30695f9171555f13070d7408bf1e2ba88d4c2e8e6d17af4f67d"
RESET_SHA256 = "4620b44ee99596ab087e690be848bde07da883ba60ee3acd90aa64b0b9673a9f"

STAGE3_MAX_SECONDS = 21600
STAGE3_MAX_GENERATIONS = 3000
GLOBAL_MAX_SECONDS = 108000
MAX_STORAGE_BYTES = 5368709120


def diagnostic_ledger(tmp_path, monkeypatch, *, authority_document=None):
    """A ledger whose reset authority lives beside the fictional diagnostic one."""
    reset = tmp_path / "reset-authority.json"
    reset.write_bytes(RESET_PATH.read_bytes())
    if authority_document is not None:
        _, sha = write_authority(tmp_path, authority_document)
        monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    return run_budget.BudgetLedger(
        tmp_path / "budget-ledger.json",
        OPENING_PATH,
        OPENING_SHA256,
        reset_path=reset,
        reset_sha256=RESET_SHA256,
    )


def diagnostic_refs(slot, messages_sha256=None):
    if messages_sha256 is None:
        messages_sha256 = slot_messages_sha256(int(slot))
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "run_id": DIAGNOSTIC_ID,
        "diagnostic_slot": slot,
        "messages_sha256": messages_sha256,
    }


def reserve_slot(budget, slot, seconds=120):
    return budget.reserve(
        "generation", seconds, generations=1, refs=diagnostic_refs(slot)
    )


def _register_third_and_fourth(budget):
    budget.register_versions(
        {
            "assistant_version": "assistant_prompt-third",
            "assistant_version_refs": {"template_identity_sha256": "3" * 64},
        }
    )
    budget.register_versions(
        {
            "assistant_version": "assistant_prompt-fourth",
            "assistant_version_refs": {"template_identity_sha256": "4" * 64},
        }
    )


def _pin_fifth_authority(tmp_path, monkeypatch, budget):
    """Bind the authority to the ledger's exact first four assistant versions."""
    entries = json.loads(budget.path.read_bytes())["prompt_versions"][
        "assistant_prompt"
    ]
    prior_sha = diagnostic_authority.digest(diagnostic_authority.canonical(entries[:4]))
    _, sha = write_authority(
        tmp_path,
        fictional_authority(
            prior_assistant_versions_sha256=prior_sha,
            fifth_version={
                "version_id": FIFTH_ID,
                "refs": {"template_identity_sha256": "f" * 64},
            },
        ),
    )
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)


def register_fifth(budget):
    budget.register_versions(
        {
            "assistant_version": FIFTH_ID,
            "assistant_version_refs": {"template_identity_sha256": "f" * 64},
        }
    )


# ---------------------------------------------------------------------------
# Diagnostic generation reservations
# ---------------------------------------------------------------------------


def test_diagnostic_generation_reserves_each_slot_once(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    identifiers = [reserve_slot(budget, f"{index:02d}") for index in range(20)]
    assert len(set(identifiers)) == 20
    snapshot = budget.snapshot()
    assert snapshot["usage"]["generations"] == 20
    assert snapshot["usage"]["seconds"] == 20 * 120
    document = json.loads(budget.path.read_bytes())
    diagnostic = [r for r in document["reservations"] if "diagnostic_id" in r["refs"]]
    assert len(diagnostic) == 20
    assert all(record["kind"] == "generation" for record in diagnostic)


def test_diagnostic_reservation_charges_120_seconds_and_one_generation(
    tmp_path, monkeypatch
):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    identifier = reserve_slot(budget, "00")
    assert budget.charge(identifier) == (120, 1, 0)


@pytest.mark.parametrize("slot", ["00", "07", "19"])
def test_diagnostic_reserve_accepts_any_frozen_slot(tmp_path, monkeypatch, slot):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    reserve_slot(budget, slot)


def test_diagnostic_reserve_rejects_a_21st_slot(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    with pytest.raises(run_budget.BudgetLedgerError, match="outside the twenty frozen"):
        reserve_slot(budget, "20")


def test_diagnostic_reserve_rejects_a_foreign_messages_sha256(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    refs = diagnostic_refs("00")
    refs["messages_sha256"] = "0" * 64
    with pytest.raises(run_budget.BudgetLedgerError, match="outside the twenty frozen"):
        budget.reserve("generation", 120, generations=1, refs=refs)


@pytest.mark.parametrize("field", ["diagnostic_id", "run_id"])
def test_diagnostic_reserve_rejects_mismatched_ids(tmp_path, monkeypatch, field):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    refs = diagnostic_refs("00")
    refs[field] = "some-other-id"
    with pytest.raises(run_budget.BudgetLedgerError, match="outside the twenty frozen"):
        budget.reserve("generation", 120, generations=1, refs=refs)


@pytest.mark.parametrize("seconds", [0, 121])
def test_diagnostic_reserve_rejects_seconds_outside_bounds(
    tmp_path, monkeypatch, seconds
):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    with pytest.raises(run_budget.BudgetLedgerError, match="outside the twenty frozen"):
        reserve_slot(budget, "00", seconds=seconds)


@pytest.mark.parametrize("generations", [0, 2])
def test_diagnostic_reserve_requires_exactly_one_generation(
    tmp_path, monkeypatch, generations
):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    with pytest.raises(run_budget.BudgetLedgerError, match="outside the twenty frozen"):
        budget.reserve(
            "generation", 120, generations=generations, refs=diagnostic_refs("00")
        )


def test_diagnostic_extension_refuses_non_generation_kinds(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    with pytest.raises(
        run_budget.BudgetLedgerError, match="generation accounting only"
    ):
        budget.reserve("row", 120, generations=1, refs=diagnostic_refs("00"))


# ---------------------------------------------------------------------------
# Once-only slots: no replay under a new run id, cancellation or boot
# ---------------------------------------------------------------------------


def test_duplicate_slot_is_rejected_while_reserved(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    reserve_slot(budget, "00")
    with pytest.raises(
        run_budget.BudgetLedgerError, match="already reserved; no retry"
    ):
        reserve_slot(budget, "00")


def test_duplicate_slot_is_rejected_even_after_cancellation(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    identifier = reserve_slot(budget, "00")
    budget.settle(identifier, "cancelled", 0, never_launched=True)
    with pytest.raises(
        run_budget.BudgetLedgerError, match="already reserved; no retry"
    ):
        reserve_slot(budget, "00")


def test_duplicate_slot_rejects_a_replay_under_a_different_run_id(
    tmp_path, monkeypatch
):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    reserve_slot(budget, "00")
    refs = diagnostic_refs("00")
    refs["run_id"] = "some-other-run-id"
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("generation", 120, generations=1, refs=refs)


def test_duplicate_slot_cannot_be_replayed_by_a_new_boot(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    reserve_slot(budget, "04")
    reopened = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    with pytest.raises(
        run_budget.BudgetLedgerError, match="already reserved; no retry"
    ):
        reserve_slot(reopened, "04")


def test_diagnostic_reservation_survives_reopen(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    identifier = reserve_slot(budget, "03")
    reopened = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    snapshot = reopened.snapshot()
    assert identifier in snapshot["unfinished_reservation_ids"]
    assert snapshot["usage"]["generations"] == 1
    assert snapshot["usage"]["seconds"] == 120


# ---------------------------------------------------------------------------
# Fail-closed authorities
# ---------------------------------------------------------------------------


def test_missing_authority_fails_closed(tmp_path, monkeypatch):
    budget = diagnostic_ledger(tmp_path, monkeypatch, authority_document=None)
    with pytest.raises(
        run_budget.BudgetLedgerError, match="invalid diagnostic authority"
    ):
        reserve_slot(budget, "00")


def test_altered_authority_fails_closed(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    (tmp_path / AUTHORITY_NAME).write_text(json.dumps({"forged": True}) + "\n")
    with pytest.raises(
        run_budget.BudgetLedgerError, match="invalid diagnostic authority"
    ):
        reserve_slot(budget, "00")


def test_diagnostic_extension_requires_the_retained_reset_authority(tmp_path):
    budget = run_budget.BudgetLedger(
        tmp_path / "ledger.json", OPENING_PATH, OPENING_SHA256
    )
    with pytest.raises(run_budget.BudgetLedgerError, match="retained reset authority"):
        reserve_slot(budget, "00")


# ---------------------------------------------------------------------------
# The exact fifth version: caps, extraction and unchanged history
# ---------------------------------------------------------------------------


def test_version_cap_is_five_only_after_the_exact_fifth(tmp_path, monkeypatch):
    budget = diagnostic_ledger(tmp_path, monkeypatch, authority_document=None)
    _register_third_and_fourth(budget)
    before = budget.snapshot()
    assert before["caps"]["max_assistant_prompt_versions"] == 4
    assert before["version_remaining"]["assistant_prompt"] == 0
    assert before["caps"]["max_extraction_versions"] == 2

    _pin_fifth_authority(tmp_path, monkeypatch, budget)
    register_fifth(budget)

    after = budget.snapshot()
    assert after["caps"]["max_assistant_prompt_versions"] == 5
    assert after["version_remaining"]["assistant_prompt"] == 0
    assert len(after["prompt_versions"]["assistant_prompt"]) == 5
    assert after["caps"]["max_extraction_versions"] == 2
    # The immutable opening cap and the version registry stay unchanged.
    document = json.loads(budget.path.read_bytes())
    assert document["opening"]["caps"]["max_assistant_prompt_versions"] == 4


def test_fifth_version_does_not_change_other_caps_or_usage(tmp_path, monkeypatch):
    budget = diagnostic_ledger(tmp_path, monkeypatch, authority_document=None)
    _register_third_and_fourth(budget)
    before = budget.snapshot()
    _pin_fifth_authority(tmp_path, monkeypatch, budget)
    register_fifth(budget)
    after = budget.snapshot()
    assert after["usage"] == before["usage"]
    assert after["remaining_generations"] == STAGE3_MAX_GENERATIONS
    assert after["remaining_seconds"] == STAGE3_MAX_SECONDS
    assert (
        after["remaining_storage_bytes"]
        == MAX_STORAGE_BYTES - before["storage_bytes_used"]
    )
    assert after["caps"]["stage3_max_generations"] == STAGE3_MAX_GENERATIONS
    assert after["caps"]["stage3_max_seconds"] == STAGE3_MAX_SECONDS
    assert after["caps"]["global_max_seconds"] == GLOBAL_MAX_SECONDS
    assert after["caps"]["max_storage_bytes"] == MAX_STORAGE_BYTES
    assert after["caps"]["max_extraction_versions"] == 2


def test_a_wrong_fifth_version_is_refused(tmp_path, monkeypatch):
    budget = diagnostic_ledger(tmp_path, monkeypatch, authority_document=None)
    _register_third_and_fourth(budget)
    _pin_fifth_authority(tmp_path, monkeypatch, budget)
    with pytest.raises(
        run_budget.BudgetLedgerError, match="only the explicitly authorized fifth"
    ):
        budget.register_versions(
            {
                "assistant_version": FIFTH_ID,
                "assistant_version_refs": {"template_identity_sha256": "e" * 64},
            }
        )


def test_an_altered_first_four_is_refused(tmp_path, monkeypatch):
    budget = diagnostic_ledger(tmp_path, monkeypatch, authority_document=None)
    _register_third_and_fourth(budget)
    # Pin to a digest that does not match the actual first four entries.
    _, sha = write_authority(
        tmp_path,
        fictional_authority(prior_assistant_versions_sha256="1" * 64),
    )
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(run_budget.BudgetLedgerError, match="preserve all four"):
        register_fifth(budget)


def test_a_sixth_version_is_refused(tmp_path, monkeypatch):
    budget = diagnostic_ledger(tmp_path, monkeypatch, authority_document=None)
    _register_third_and_fourth(budget)
    _pin_fifth_authority(tmp_path, monkeypatch, budget)
    register_fifth(budget)
    with pytest.raises(run_budget.BudgetLedgerError, match="exhausted"):
        budget.register_versions(
            {
                "assistant_version": "assistant_prompt-sixth",
                "assistant_version_refs": {"template_identity_sha256": "6" * 64},
            }
        )


def test_the_fifth_cannot_be_registered_out_of_order(tmp_path, monkeypatch):
    # With fewer than four prior versions there is no diagnostic allowance; a
    # fifth-named version is only an ordinary registration, never a cap bump.
    budget = diagnostic_ledger(tmp_path, monkeypatch, authority_document=None)
    budget.register_versions(
        {
            "assistant_version": "assistant_prompt-third",
            "assistant_version_refs": {"template_identity_sha256": "3" * 64},
        }
    )
    snapshot = budget.snapshot()
    assert snapshot["caps"]["max_assistant_prompt_versions"] == 4
    assert len(snapshot["prompt_versions"]["assistant_prompt"]) == 3


# ---------------------------------------------------------------------------
# Ordinary row gates remain untouched
# ---------------------------------------------------------------------------


def test_normal_row_retry_gates_remain_untouched(tmp_path, monkeypatch):
    budget = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    first = budget.reserve(
        "row",
        120,
        generations=1,
        refs={"run_id": "coord", "attempt_key": "fixed-request"},
    )
    budget.settle(first, "finished", 60, generations=1)
    reopened = diagnostic_ledger(
        tmp_path, monkeypatch, authority_document=fictional_authority()
    )
    with pytest.raises(run_budget.BudgetLedgerError, match="retries refused"):
        reopened.reserve(
            "row",
            120,
            generations=1,
            refs={"run_id": "new-run", "attempt_key": "fixed-request"},
        )
