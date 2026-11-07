"""Behavioral tests for the portable Stage 4A budget ledger.

Derived from ``docs/stage4a-runner-contract.md``, ``docs/budget-usage-reset-20261027.json``
and ``runs/stage4a-runner-20261027/interfaces.md`` *before* the implementation
was written. The real opening/reset authority documents are only read; every
ledger is created under ``tmp_path``. No model, tokenizer, Torch/MLX import,
subprocess or network use is involved.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from messaging_nlp import run_budget

ROOT = Path(__file__).resolve().parents[1]
OPENING_PATH = ROOT / "docs" / "stage1-budget-opening.json"
RESET_PATH = ROOT / "docs" / "budget-usage-reset-20261027.json"

# Frozen authorities (independent constants, not imported from the module).
OPENING_SHA256 = "3f7a172dd44af30695f9171555f13070d7408bf1e2ba88d4c2e8e6d17af4f67d"
RESET_SHA256 = "4620b44ee99596ab087e690be848bde07da883ba60ee3acd90aa64b0b9673a9f"
RESET_ID = "development-usage-reset-20261027T222217Z"

IMPORTED_RECORDS = 416
IMPORTED_SECONDS = 11075
IMPORTED_GENERATIONS = 2054
IMPORTED_STORAGE_BYTES = 78719238
STAGE3_MAX_SECONDS = 21600
STAGE3_MAX_GENERATIONS = 3000
GLOBAL_MAX_SECONDS = 108000
MAX_STORAGE_BYTES = 5368709120
MAX_ASSISTANT_VERSIONS = 4
MAX_EXTRACTION_VERSIONS = 2
STORAGE_REMAINING = MAX_STORAGE_BYTES - IMPORTED_STORAGE_BYTES


def opening():
    return json.loads(OPENING_PATH.read_bytes())


def reset_authority():
    return json.loads(RESET_PATH.read_bytes())


def charged(record):
    """Independent conservative charge rule (contract section 109-123)."""
    if record["status"] == "cancelled":
        return (0, 0, 0)
    if record["status"] == "finished":
        return (
            record["actual_seconds"],
            record["actual_generations"],
            record["actual_bytes"],
        )
    return (
        max(record["planned_seconds"], record["actual_seconds"]),
        max(record["planned_generations"], record["actual_generations"]),
        max(record["planned_bytes"], record["actual_bytes"]),
    )


def derived_imported_usage(records=None):
    records = opening()["reservations"] if records is None else records
    seconds = generations = storage = 0
    for record in records:
        one, two, three = charged(record)
        seconds += one
        generations += two
        storage += three
    return {
        "seconds": seconds,
        "generations": generations,
        "storage_bytes": storage,
    }


def ledger(
    tmp_path,
    *,
    reset=True,
    opening_path=None,
    opening_sha=None,
    reset_path=None,
    reset_sha=None,
    name="budget-ledger.json",
):
    kwargs = {}
    if reset:
        kwargs = {
            "reset_path": RESET_PATH if reset_path is None else reset_path,
            "reset_sha256": RESET_SHA256 if reset_sha is None else reset_sha,
        }
    return run_budget.BudgetLedger(
        tmp_path / name,
        OPENING_PATH if opening_path is None else opening_path,
        OPENING_SHA256 if opening_sha is None else opening_sha,
        **kwargs,
    )


@pytest.mark.parametrize("status", ["reserved", "failed", "finished"])
def test_attempt_identity_cannot_retry_under_new_run_or_reopened_epoch(
    tmp_path, status
):
    value = ledger(tmp_path)
    first = value.reserve(
        "row",
        120,
        generations=1,
        refs={"run_id": "old-run", "attempt_key": "fixed-request-condition"},
    )
    if status != "reserved":
        value.settle(first, status, 2, generations=1)
    reopened = ledger(tmp_path)
    with pytest.raises(ValueError, match="retries refused"):
        reopened.reserve(
            "row",
            120,
            generations=1,
            refs={"run_id": "new-run", "attempt_key": "fixed-request-condition"},
        )


def sealed(tmp_path, document, *, name="authority.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document, sort_keys=True, indent=1) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_document(path):
    return json.loads(Path(path).read_bytes())


def reservation(document, reservation_id):
    return next(
        record
        for record in document["reservations"]
        if record["reservation_id"] == reservation_id
    )


@pytest.mark.parametrize("status", ["finished", "failed"])
def test_final_storage_observed_bytes_include_own_ledger_growth_not_allocation(
    tmp_path, status
):
    value = ledger(tmp_path)
    before = value.path.stat().st_size
    identifier = value.reserve("storage", 0, storage_bytes=1_000_000)
    value.settle_storage(
        identifier, status, output_bytes=12345, ledger_start_bytes=before
    )
    expected = 12345 + max(0, value.path.stat().st_size - before)
    record = reservation(ledger_document(value.path), identifier)
    assert record["actual_bytes"] == expected < 1_000_000
    assert value.charge(identifier)[2] == (
        expected if status == "finished" else 1_000_000
    )
    with pytest.raises(ValueError, match="settled"):
        value.settle_storage(
            identifier, status, output_bytes=12345, ledger_start_bytes=before
        )


# ---------------------------------------------------------------------------
# Frozen authority documents
# ---------------------------------------------------------------------------


def test_frozen_authority_bytes_match_the_documented_hashes():
    assert hashlib.sha256(OPENING_PATH.read_bytes()).hexdigest() == OPENING_SHA256
    assert hashlib.sha256(RESET_PATH.read_bytes()).hexdigest() == RESET_SHA256
    authority = reset_authority()
    assert authority["reset_id"] == RESET_ID
    assert authority["source_opening"]["sha256"] == OPENING_SHA256


def test_opening_contains_the_expected_prior_usage():
    document = opening()
    assert len(document["reservations"]) == IMPORTED_RECORDS
    usage = derived_imported_usage()
    assert usage == {
        "seconds": IMPORTED_SECONDS,
        "generations": IMPORTED_GENERATIONS,
        "storage_bytes": IMPORTED_STORAGE_BYTES,
    }
    assert document["prior_global_seconds"] == IMPORTED_SECONDS
    assert document["prior_storage_bytes"] == IMPORTED_STORAGE_BYTES
    assert document["prior_usage_by_historical_account"]["stage3"] == {
        "seconds": IMPORTED_SECONDS,
        "generations": IMPORTED_GENERATIONS,
        "bytes": IMPORTED_STORAGE_BYTES,
    }
    assert len(document["registered_prompt_versions"]["assistant_prompt"]) == 2
    assert len(document["registered_prompt_versions"]["extraction"]) == 1


def test_constructor_rejects_a_substituted_opening(tmp_path):
    with pytest.raises(run_budget.BudgetLedgerError):
        run_budget.BudgetLedger(
            tmp_path / "ledger.json",
            OPENING_PATH,
            "0" * 64,
            reset_path=RESET_PATH,
            reset_sha256=RESET_SHA256,
        )
    assert not (tmp_path / "ledger.json").exists()


def test_constructor_rejects_an_opening_whose_records_do_not_match_totals(
    tmp_path, monkeypatch
):
    document = opening()
    document["reservations"][1]["actual_generations"] += 1
    path, sha = sealed(tmp_path, document, name="opening.json")
    monkeypatch.setattr(run_budget, "OPENING_SHA256", sha)
    with pytest.raises(run_budget.BudgetLedgerError):
        run_budget.BudgetLedger(
            tmp_path / "ledger.json",
            path,
            sha,
            reset_path=RESET_PATH,
            reset_sha256=RESET_SHA256,
        )
    assert not (tmp_path / "ledger.json").exists()


def test_constructor_rejects_opening_caps_that_disagree(tmp_path, monkeypatch):
    document = opening()
    document["caps"]["stage3_max_generations"] = 4000
    path, sha = sealed(tmp_path, document, name="opening.json")
    monkeypatch.setattr(run_budget, "OPENING_SHA256", sha)
    with pytest.raises(run_budget.BudgetLedgerError):
        run_budget.BudgetLedger(
            tmp_path / "ledger.json",
            path,
            sha,
            reset_path=RESET_PATH,
            reset_sha256=RESET_SHA256,
        )


def test_changed_opening_after_creation_fails_closed(tmp_path):
    copy = tmp_path / "opening.json"
    copy.write_bytes(OPENING_PATH.read_bytes())
    budget = ledger(tmp_path, opening_path=copy)
    budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    changed = json.loads(copy.read_bytes())
    changed["reservations"][0]["actual_bytes"] += 1
    copy.write_text(json.dumps(changed))
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.snapshot()
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 120, 1, refs={"run_id": "coord"})


# ---------------------------------------------------------------------------
# Imported records and the authorized reset epoch
# ---------------------------------------------------------------------------


def test_imported_records_are_preserved_exactly(tmp_path):
    budget = ledger(tmp_path)
    document = ledger_document(budget.path)
    assert document["imported_records"] == opening()["reservations"]
    assert len(document["imported_records"]) == IMPORTED_RECORDS
    snapshot = budget.snapshot()
    assert snapshot["imported"]["reservation_count"] == IMPORTED_RECORDS
    assert snapshot["imported"]["usage"] == derived_imported_usage()


def test_reset_epoch_zeroes_generations_and_seconds_only(tmp_path):
    snapshot = ledger(tmp_path).snapshot()
    assert snapshot["usage"]["seconds"] == 0
    assert snapshot["usage"]["generations"] == 0
    assert snapshot["usage"]["storage_bytes"] == IMPORTED_STORAGE_BYTES
    assert snapshot["remaining_seconds"] == STAGE3_MAX_SECONDS
    assert snapshot["remaining_generations"] == STAGE3_MAX_GENERATIONS
    assert snapshot["remaining_storage_bytes"] == STORAGE_REMAINING
    assert snapshot["remaining_global_seconds"] == GLOBAL_MAX_SECONDS
    assert snapshot["caps_exceeded"] is False


def test_reset_preserves_history_storage_and_versions(tmp_path):
    budget = ledger(tmp_path)
    snapshot = budget.snapshot()
    assert snapshot["epoch"]["mode"] == "authorized_reset"
    assert snapshot["epoch"]["reset_id"] == RESET_ID
    assert snapshot["epoch"]["reset_sha256"] == RESET_SHA256
    assert snapshot["history"][0]["usage"] == {
        "seconds": IMPORTED_SECONDS,
        "generations": IMPORTED_GENERATIONS,
        "storage_bytes": IMPORTED_STORAGE_BYTES,
    }
    assert snapshot["prompt_versions"] == opening()["registered_prompt_versions"]
    assert snapshot["caps"]["stage3_max_seconds"] == STAGE3_MAX_SECONDS
    assert snapshot["caps"]["stage3_max_generations"] == STAGE3_MAX_GENERATIONS
    assert snapshot["caps"]["global_max_seconds"] == GLOBAL_MAX_SECONDS
    assert snapshot["caps"]["max_storage_bytes"] == MAX_STORAGE_BYTES


def test_reset_epoch_is_durable_and_applied_once(tmp_path):
    first = ledger(tmp_path)
    document = ledger_document(first.path)
    epochs = [
        entry for entry in document["epochs"] if entry["mode"] == "authorized_reset"
    ]
    assert len(epochs) == 1
    before = first.snapshot()
    reopened = ledger(tmp_path)
    after = reopened.snapshot()
    assert after["usage"] == before["usage"]
    assert after["remaining_seconds"] == before["remaining_seconds"]
    document = ledger_document(first.path)
    assert len([e for e in document["epochs"] if e["mode"] == "authorized_reset"]) == 1
    assert len(document["epochs"]) == len(before["epochs"])


def test_charges_after_the_reset_survive_reopen(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    budget.settle(identifier, "finished", 61, 1)
    budget.reserve("storage", 0, storage_bytes=2500, refs={"run_id": "coord"})
    reread = ledger(tmp_path).snapshot()
    assert reread["usage"] == {
        "seconds": 61,
        "generations": 1,
        "storage_bytes": IMPORTED_STORAGE_BYTES + 2500,
    }
    assert reread["remaining_seconds"] == STAGE3_MAX_SECONDS - 61
    assert reread["remaining_generations"] == STAGE3_MAX_GENERATIONS - 1


def test_reopening_never_reapplies_the_reset(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    budget.settle(identifier, "failed", 10, 1)
    for _ in range(3):
        budget = ledger(tmp_path)
    document = ledger_document(budget.path)
    assert len([e for e in document["epochs"] if e["mode"] == "authorized_reset"]) == 1
    assert budget.snapshot()["usage"]["seconds"] == 120


def test_reset_requires_both_path_and_hash(tmp_path):
    with pytest.raises(run_budget.BudgetLedgerError):
        run_budget.BudgetLedger(
            tmp_path / "a.json",
            OPENING_PATH,
            OPENING_SHA256,
            reset_path=RESET_PATH,
        )
    with pytest.raises(run_budget.BudgetLedgerError):
        run_budget.BudgetLedger(
            tmp_path / "b.json",
            OPENING_PATH,
            OPENING_SHA256,
            reset_sha256=RESET_SHA256,
        )
    assert not (tmp_path / "a.json").exists()
    assert not (tmp_path / "b.json").exists()


def test_reset_hash_mismatch_fails_closed(tmp_path):
    with pytest.raises(run_budget.BudgetLedgerError):
        ledger(tmp_path, reset_sha="f" * 64)


def test_reset_id_mismatch_with_same_bytes_is_refused(tmp_path, monkeypatch):
    document = reset_authority()
    document["reset_id"] = "development-usage-reset-OTHER"
    path, sha = sealed(tmp_path, document, name="reset.json")
    monkeypatch.setattr(run_budget, "RESET_AUTHORITY_SHA256", sha)
    with pytest.raises(run_budget.BudgetLedgerError):
        ledger(tmp_path, reset_path=path, reset_sha=sha)


def test_reset_bound_to_another_opening_is_refused(tmp_path, monkeypatch):
    document = reset_authority()
    document["source_opening"]["sha256"] = "1" * 64
    path, sha = sealed(tmp_path, document, name="reset.json")
    monkeypatch.setattr(run_budget, "RESET_AUTHORITY_SHA256", sha)
    with pytest.raises(run_budget.BudgetLedgerError):
        ledger(tmp_path, reset_path=path, reset_sha=sha)


def test_reset_caps_mismatch_is_refused(tmp_path, monkeypatch):
    document = reset_authority()
    document["unchanged_caps"]["development_seconds"] = 21601
    path, sha = sealed(tmp_path, document, name="reset.json")
    monkeypatch.setattr(run_budget, "RESET_AUTHORITY_SHA256", sha)
    with pytest.raises(run_budget.BudgetLedgerError):
        ledger(tmp_path, reset_path=path, reset_sha=sha)


def test_reset_scope_must_preserve_storage_and_versions(tmp_path, monkeypatch):
    document = reset_authority()
    document["scope"]["reset_storage_usage"] = True
    path, sha = sealed(tmp_path, document, name="reset.json")
    monkeypatch.setattr(run_budget, "RESET_AUTHORITY_SHA256", sha)
    with pytest.raises(run_budget.BudgetLedgerError):
        ledger(tmp_path, reset_path=path, reset_sha=sha)


def test_reopen_with_changed_reset_bytes_for_same_id_is_refused(tmp_path):
    copy = tmp_path / "reset.json"
    copy.write_bytes(RESET_PATH.read_bytes())
    ledger(tmp_path, reset_path=copy)
    document = json.loads(copy.read_bytes())
    document["authorization"]["verbatim"] += " (edited)"
    copy.write_text(json.dumps(document))
    with pytest.raises(run_budget.BudgetLedgerError):
        ledger(tmp_path, reset_path=copy)


def test_historical_no_reset_mode_uses_the_imported_usage(tmp_path):
    budget = ledger(tmp_path, reset=False)
    snapshot = budget.snapshot()
    assert snapshot["usage"] == derived_imported_usage()
    assert snapshot["remaining_seconds"] == STAGE3_MAX_SECONDS - IMPORTED_SECONDS
    assert snapshot["remaining_generations"] == (
        STAGE3_MAX_GENERATIONS - IMPORTED_GENERATIONS
    )
    assert snapshot["epoch"]["mode"] == "opening_import"


def test_no_reset_ledger_cannot_revert_from_the_reset_epoch(tmp_path):
    ledger(tmp_path)
    with pytest.raises(run_budget.BudgetLedgerError):
        ledger(tmp_path, reset=False)


def test_epoch_transition_retains_prior_new_charges_as_history(tmp_path):
    old = ledger(tmp_path, reset=False)
    row = old.reserve("row", 240, 2, refs={"run_id": "coord"})
    old.settle(row, "finished", 200, 2)
    storage = old.reserve("storage", 0, storage_bytes=1000, refs={"run_id": "coord"})
    old.settle(storage, "finished", 0, 0, 1000)
    migrated = ledger(tmp_path)
    snapshot = migrated.snapshot()
    assert snapshot["usage"] == {
        "seconds": 0,
        "generations": 0,
        "storage_bytes": IMPORTED_STORAGE_BYTES + 1000,
    }
    assert snapshot["remaining_seconds"] == STAGE3_MAX_SECONDS
    retired = snapshot["history"]
    assert retired[-1]["index"] == 0
    assert retired[-1]["new_charges"] == {
        "seconds": 200,
        "generations": 2,
        "storage_bytes": 1000,
    }
    assert reservation(ledger_document(migrated.path), row)["actual_seconds"] == 200
    assert (
        len(
            [
                e
                for e in ledger_document(migrated.path)["epochs"]
                if e["mode"] == "authorized_reset"
            ]
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Reservations, settlement and caps
# ---------------------------------------------------------------------------


def test_reserve_returns_an_id_and_charges_the_planned_maximum(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    assert isinstance(identifier, str) and identifier
    assert budget.charge(identifier) == (120, 1, 0)
    snapshot = budget.snapshot()
    assert snapshot["usage"] == {
        "seconds": 120,
        "generations": 1,
        "storage_bytes": IMPORTED_STORAGE_BYTES,
    }
    assert snapshot["unfinished_reservation_ids"] == [identifier]
    assert reservation(ledger_document(budget.path), identifier)["status"] == "reserved"


@pytest.mark.parametrize("kind", ["startup", "overhead", "preparation"])
def test_seconds_only_kinds_accept_seconds_only(tmp_path, kind):
    budget = ledger(tmp_path)
    identifier = budget.reserve(kind, 30, refs={"run_id": "coord"})
    assert budget.charge(identifier) == (30, 0, 0)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve(kind, 30, 1, refs={"run_id": "coord"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve(kind, 30, 0, 10, refs={"run_id": "coord"})


def test_storage_kind_charges_bytes_only(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("storage", 0, storage_bytes=4096, refs={"run_id": "c"})
    assert budget.charge(identifier) == (0, 0, 4096)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("storage", 10, storage_bytes=4096)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("storage", 0, 1, 4096)


def test_unknown_training_profile_and_empty_reservations_are_refused(tmp_path):
    budget = ledger(tmp_path)
    for kind in ("training", "training_job", "profile", "generation_job", ""):
        with pytest.raises(run_budget.BudgetLedgerError):
            budget.reserve(kind, 60, refs={"run_id": "coord"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 0, 0)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", -1, 1)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 1.5, 1)
    assert budget.snapshot()["usage"]["generations"] == 0


def test_caps_are_enforced_and_stage4_is_not_borrowed(tmp_path):
    budget = ledger(tmp_path)
    budget.reserve("row", STAGE3_MAX_SECONDS, STAGE3_MAX_GENERATIONS)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 1, 1)
    # The unused stage4 allowance cannot be borrowed for stage3 work.
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("startup", 1)


def test_storage_cap_is_enforced(tmp_path):
    budget = ledger(tmp_path)
    budget.reserve("storage", 0, storage_bytes=STORAGE_REMAINING)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("storage", 0, storage_bytes=1)


def test_refused_reservation_does_not_write(tmp_path):
    budget = ledger(tmp_path)
    before = Path(budget.path).read_bytes()
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", STAGE3_MAX_SECONDS + 1, 1)
    assert Path(budget.path).read_bytes() == before
    assert budget.snapshot()["usage"]["seconds"] == 0


def test_settle_finished_charges_the_observed_actual(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    budget.settle(identifier, "finished", 45, 1)
    assert budget.charge(identifier) == (45, 1, 0)
    assert budget.snapshot()["remaining_seconds"] == STAGE3_MAX_SECONDS - 45


def test_finished_invalid_json_still_charges_the_execution(tmp_path):
    # Invalid JSON is a task failure, not a cancelled or refunded call.
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    budget.settle(identifier, "finished", 100, 1)
    charged_seconds, charged_generations, _ = budget.charge(identifier)
    assert charged_seconds == 100
    assert charged_generations == 1
    assert budget.charge(identifier) != (0, 0, 0)
    assert reservation(ledger_document(budget.path), identifier)["status"] == "finished"


def test_finished_work_must_report_its_generation(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.settle(identifier, "finished", 30, 0)
    assert budget.charge(identifier) == (120, 1, 0)


def test_failed_settlement_keeps_max_of_planned_and_actual(tmp_path):
    budget = ledger(tmp_path)
    first = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    second = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    budget.settle(first, "failed", 30, 1)
    budget.settle(second, "failed", 150, 2)
    assert budget.charge(first) == (120, 1, 0)
    assert budget.charge(second) == (150, 2, 0)


def test_reserved_crash_remains_charged_and_blocking(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    # The process "crashes": reopen without settling.
    reopened = ledger(tmp_path)
    assert reopened.charge(identifier) == (120, 1, 0)
    assert identifier in reopened.snapshot()["unfinished_reservation_ids"]
    assert reopened.snapshot()["usage"]["seconds"] == 120


def test_cancelled_requires_never_launched_proof(tmp_path):
    budget = ledger(tmp_path)
    unproven = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.settle(unproven, "cancelled", 0)
    proven = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    budget.settle(proven, "cancelled", 0, never_launched=True)
    assert budget.charge(proven) == (0, 0, 0)
    assert reservation(ledger_document(budget.path), proven)["status"] == "cancelled"
    contradictory = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.settle(contradictory, "cancelled", 5, never_launched=True)


def test_cancelled_storage_reservation_is_zero(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("storage", 0, storage_bytes=4096)
    budget.settle(identifier, "cancelled", 0, never_launched=True)
    assert budget.charge(identifier) == (0, 0, 0)


def test_replay_unknown_imported_and_crashed_status_are_refused(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    budget.settle(identifier, "finished", 10, 1)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.settle(identifier, "finished", 90, 1)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.settle("does-not-exist", "failed", 1, 0)
    imported = opening()["reservations"][0]["reservation_id"]
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.settle(imported, "failed", 1, 0)
    other = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.settle(other, "crashed", 10, 1)
    assert budget.charge(identifier) == (10, 1, 0)
    assert budget.charge(other) == (120, 1, 0)


def test_charge_of_imported_records_matches_the_conservative_rule(tmp_path):
    budget = ledger(tmp_path)
    records = opening()["reservations"]
    finished = records[0]
    failed = next(record for record in records if record["status"] == "failed")
    assert budget.charge(finished["reservation_id"]) == charged(finished)
    assert budget.charge(failed["reservation_id"]) == charged(failed)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.charge("unknown-reservation")


def test_unfinished_reservation_blocks_unrelated_work(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord-a"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 120, 1, refs={"run_id": "coord-b"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 120, 1)
    same = budget.reserve("row", 120, 1, refs={"run_id": "coord-a"})
    assert same != identifier
    budget.settle(identifier, "finished", 10, 1)
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 120, 1, refs={"run_id": "coord-b"})
    budget.settle(same, "failed", 10, 1)
    budget.reserve("row", 120, 1, refs={"run_id": "coord-b"})


def test_snapshot_is_an_independent_copy(tmp_path):
    budget = ledger(tmp_path)
    snapshot = budget.snapshot()
    snapshot["reservations"].append({"forged": True})
    snapshot["usage"]["seconds"] = 999
    fresh = budget.snapshot()
    assert fresh["reservations"] == []
    assert fresh["usage"]["seconds"] == 0


# ---------------------------------------------------------------------------
# Version registration
# ---------------------------------------------------------------------------


def _spending(**overrides):
    values = {
        "assistant_version": "assistant_prompt-shared-common",
        "assistant_version_refs": {
            "template_identity_sha256": "a" * 64,
            "assistant_prompt_sha256": "b" * 64,
        },
        "extraction_version": None,
        "extraction_version_refs": {},
    }
    values.update(overrides)
    return values


def test_register_versions_preserves_imported_and_is_idempotent(tmp_path):
    budget = ledger(tmp_path)
    first = budget.register_versions(_spending())
    revision = ledger_document(budget.path)["revision"]
    second = budget.register_versions(_spending())
    assert ledger_document(budget.path)["revision"] == revision
    assert first["assistant_prompt"]["version_id"] == "assistant_prompt-shared-common"
    assert first["assistant_prompt"] == second["assistant_prompt"]
    snapshot = budget.snapshot()
    assert len(snapshot["prompt_versions"]["assistant_prompt"]) == 3
    assert snapshot["version_remaining"] == {
        "assistant_prompt": MAX_ASSISTANT_VERSIONS - 3,
        "extraction": MAX_EXTRACTION_VERSIONS - 1,
    }
    imported = opening()["registered_prompt_versions"]
    assert (
        snapshot["prompt_versions"]["assistant_prompt"][:2]
        == (imported["assistant_prompt"])
    )
    assert snapshot["prompt_versions"]["extraction"] == imported["extraction"]


def test_shared_version_across_conditions_counts_once(tmp_path):
    budget = ledger(tmp_path)
    for condition in ("baseline", "B", "A+B"):
        budget.register_versions(_spending())
        # One shared prompt/template stays one version across conditions.
        assert len(budget.snapshot()["prompt_versions"]["assistant_prompt"]) == 3
    snapshot = budget.snapshot()
    assert len(snapshot["prompt_versions"]["assistant_prompt"]) == 3
    assert budget.snapshot()["version_remaining"]["assistant_prompt"] == (
        MAX_ASSISTANT_VERSIONS - 3
    )


def test_version_collision_is_refused(tmp_path):
    budget = ledger(tmp_path)
    budget.register_versions(_spending())
    before = Path(budget.path).read_bytes()
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.register_versions(
            _spending(assistant_version_refs={"template_identity_sha256": "c" * 64})
        )
    assert Path(budget.path).read_bytes() == before


def test_version_caps_include_imported_versions(tmp_path):
    budget = ledger(tmp_path)
    budget.register_versions(_spending(assistant_version="assistant_prompt-1"))
    budget.register_versions(_spending(assistant_version="assistant_prompt-2"))
    before = Path(budget.path).read_bytes()
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.register_versions(_spending(assistant_version="assistant_prompt-3"))
    assert Path(budget.path).read_bytes() == before
    budget.register_versions(
        _spending(
            assistant_version="assistant_prompt-1",
            extraction_version="extraction-1",
            extraction_version_refs={"template_identity_sha256": "d" * 64},
        )
    )
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.register_versions(
            _spending(
                assistant_version="assistant_prompt-1",
                extraction_version="extraction-2",
                extraction_version_refs={"template_identity_sha256": "e" * 64},
            )
        )


def test_version_registration_does_not_charge_or_reset(tmp_path):
    budget = ledger(tmp_path)
    before = budget.snapshot()
    budget.register_versions(
        _spending(
            assistant_version="assistant_prompt-x",
            extraction_version="extraction-x",
            extraction_version_refs={"template_identity_sha256": "f" * 64},
        )
    )
    after = budget.snapshot()
    assert after["usage"] == before["usage"]
    assert after["remaining_seconds"] == before["remaining_seconds"]
    assert after["epoch"] == before["epoch"]
    assert after["remaining_storage_bytes"] == before["remaining_storage_bytes"]
    assert after["version_remaining"] != before["version_remaining"]


def test_register_versions_accepts_objects_and_rejects_bad_shapes(tmp_path):
    budget = ledger(tmp_path)
    spending = SimpleNamespace(
        assistant_version="assistant_prompt-object",
        assistant_version_refs={"template_identity_sha256": "a" * 64},
        extraction_version=None,
    )
    result = budget.register_versions(spending)
    assert result["assistant_prompt"]["version_id"] == "assistant_prompt-object"
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.register_versions({"assistant_version": ""})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.register_versions(
            {"assistant_version": "v", "assistant_version_refs": {"k": ""}}
        )
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.register_versions(
            {"assistant_version": "v", "assistant_version_refs": {"k": 1}}
        )


# ---------------------------------------------------------------------------
# Durability, integrity, locking and concurrency
# ---------------------------------------------------------------------------


def test_ledger_revision_advances_once_per_mutation(tmp_path):
    budget = ledger(tmp_path)
    first = ledger_document(budget.path)["revision"]
    identifier = budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    second = ledger_document(budget.path)["revision"]
    assert second == first + 1
    budget.settle(identifier, "finished", 10, 1)
    assert ledger_document(budget.path)["revision"] == second + 1


def test_external_ledger_edits_fail_closed(tmp_path):
    budget = ledger(tmp_path)
    document = ledger_document(budget.path)
    document["imported_records"][0]["actual_bytes"] += 1
    Path(budget.path).write_text(json.dumps(document))
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.snapshot()
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 120, 1, refs={"run_id": "coord"})


def test_removed_imported_version_fails_closed(tmp_path):
    budget = ledger(tmp_path)
    document = ledger_document(budget.path)
    document["prompt_versions"]["assistant_prompt"].pop()
    Path(budget.path).write_text(json.dumps(document))
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.snapshot()


def test_changed_reset_authority_after_creation_fails_closed(tmp_path):
    copy = tmp_path / "reset.json"
    copy.write_bytes(RESET_PATH.read_bytes())
    budget = ledger(tmp_path, reset_path=copy)
    changed = json.loads(copy.read_bytes())
    changed["new_epoch_opening_usage"]["storage_bytes"] = 0
    copy.write_text(json.dumps(changed))
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.reserve("row", 120, 1, refs={"run_id": "coord"})
    with pytest.raises(run_budget.BudgetLedgerError):
        budget.snapshot()


def test_unfinished_legacy_reservation_blocks_after_epoch_transition(tmp_path):
    legacy = ledger(tmp_path, reset=False)
    identifier = legacy.reserve("row", 120, 1, refs={"run_id": "coord-a"})
    migrated = ledger(tmp_path)
    assert identifier in migrated.snapshot()["unfinished_reservation_ids"]
    with pytest.raises(run_budget.BudgetLedgerError):
        migrated.reserve("row", 120, 1, refs={"run_id": "coord-b"})
    assert len(migrated.snapshot()["reservations"]) == 1
    migrated.reserve("row", 120, 1, refs={"run_id": "coord-a"})


def test_temporary_files_do_not_survive_writes(tmp_path):
    budget = ledger(tmp_path)
    for _ in range(3):
        budget.reserve("row", 10, 1, refs={"run_id": "coord"})
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "budget-ledger.json",
        "budget-ledger.json.lock",
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_mutation_lock_lives_beside_the_ledger_and_git_paths_are_refused(tmp_path):
    budget = ledger(tmp_path)
    assert (tmp_path / "budget-ledger.json.lock").is_file()
    with pytest.raises(run_budget.BudgetLedgerError):
        run_budget.BudgetLedger(
            tmp_path / ".git" / "ledger.json",
            OPENING_PATH,
            OPENING_SHA256,
            reset_path=RESET_PATH,
            reset_sha256=RESET_SHA256,
        )
    with pytest.raises(run_budget.BudgetLedgerError):
        run_budget.BudgetLedger(
            OPENING_PATH,
            OPENING_PATH,
            OPENING_SHA256,
            reset_path=RESET_PATH,
            reset_sha256=RESET_SHA256,
        )
    assert not (OPENING_PATH.parent / (OPENING_PATH.name + ".lock")).exists()
    assert budget.snapshot()["remaining_seconds"] == STAGE3_MAX_SECONDS


def test_concurrent_reservations_cannot_overspend(tmp_path):
    budget = ledger(tmp_path)
    successes = []
    refusals = []
    errors = []
    barrier = threading.Barrier(2)

    def worker():
        local_ok = 0
        local_refused = 0
        try:
            barrier.wait(timeout=10)
            for _ in range(15):
                try:
                    budget.reserve("row", 1200, 1, refs={"run_id": "coord"})
                    local_ok += 1
                except run_budget.BudgetLedgerError:
                    local_refused += 1
        except BaseException as error:  # pragma: no cover - surfaced below
            errors.append(error)
        successes.append(local_ok)
        refusals.append(local_refused)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    assert sum(successes) == STAGE3_MAX_SECONDS // 1200
    snapshot = budget.snapshot()
    assert snapshot["usage"]["seconds"] == 21600
    assert snapshot["usage"]["generations"] == 18
    assert snapshot["remaining_seconds"] == 0
    assert snapshot["remaining_generations"] == STAGE3_MAX_GENERATIONS - 18
    assert sum(refusals) == 30 - 18


def test_caps_exceeded_is_visible_after_truthful_overrun(tmp_path):
    budget = ledger(tmp_path)
    identifier = budget.reserve("row", STAGE3_MAX_SECONDS, STAGE3_MAX_GENERATIONS)
    budget.settle(identifier, "failed", STAGE3_MAX_SECONDS + 10, STAGE3_MAX_GENERATIONS)
    snapshot = budget.snapshot()
    assert snapshot["usage"]["seconds"] == STAGE3_MAX_SECONDS + 10
    assert snapshot["remaining_seconds"] == -10
    assert snapshot["caps_exceeded"] is True


def test_module_is_standalone_and_has_no_archive_imports():
    source = (ROOT / "src" / "messaging_nlp" / "run_budget.py").read_text()
    assert "Old-version" not in source
    assert "run_contracts" not in source
