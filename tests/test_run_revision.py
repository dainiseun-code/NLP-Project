"""Independent authorization/reset boundaries, no model or research execution."""

import json
import threading

import pytest
from revision_fixtures import revision_fixture

from messaging_nlp.run_budget import BudgetLedgerError


def test_explicit_reset_preserves_history_storage_versions_and_caps(
    tmp_path, monkeypatch
):
    f = revision_fixture(tmp_path, monkeypatch)
    before = f["budget"].snapshot()
    raw = f["budget"].path.read_bytes()
    with pytest.raises(ValueError, match="not been applied"):
        f["open_budget"]()
    assert f["budget"].path.read_bytes() == raw
    budget = f["open_budget"](apply=True)
    after = budget.snapshot()
    assert after["usage"] == dict(
        seconds=0, generations=0, storage_bytes=before["usage"]["storage_bytes"]
    )
    assert after["reservations"] == before["reservations"]
    assert after["prompt_versions"] == before["prompt_versions"]
    assert after["caps"] == before["caps"]
    assert after["epochs"][:2] == before["epochs"]
    identifier = budget.reserve("row", 120, 1, refs=f["row_refs"]())
    budget.settle(identifier, "finished", 5, 1)
    reopened = f["open_budget"](apply=True)
    assert reopened.snapshot()["usage"]["seconds"] == 5
    assert len(reopened.snapshot()["epochs"]) == 3
    with pytest.raises(ValueError):
        f["budget"].snapshot()  # legacy authority may not spend the new epoch


@pytest.mark.parametrize("status", ["finished", "failed", "reserved"])
def test_only_authorized_scope_can_repeat_prior_keys_and_not_repeat_itself(
    tmp_path, monkeypatch, status
):
    f = revision_fixture(tmp_path, monkeypatch)
    budget = f["open_budget"](apply=True)
    refs = f["row_refs"]()
    for mutation in (
        {"run_id": "another"},
        {"experiment_id": "another"},
        {"attempt_key": "unlisted"},
        {"revision_authority_sha256": "0" * 64},
    ):
        with pytest.raises(ValueError, match="outside"):
            budget.reserve("row", 120, 1, refs=refs | mutation)
    entry = budget.reserve("row", 120, 1, refs=refs)
    if status != "reserved":
        budget.settle(entry, status, 4, 1)
    with pytest.raises(ValueError, match="retries refused"):
        f["open_budget"]().reserve("row", 120, 1, refs=refs)


def test_stale_predecessor_cannot_be_reset(tmp_path, monkeypatch):
    f = revision_fixture(tmp_path, monkeypatch)
    extra = f["budget"].reserve("preparation", 1)
    f["budget"].settle(extra, "finished", 1)
    raw = f["budget"].path.read_bytes()
    with pytest.raises(ValueError, match="predecessor"):
        f["open_budget"](apply=True)
    assert f["budget"].path.read_bytes() == raw


@pytest.mark.parametrize(
    "name",
    [
        "docs/revision.json",
        "history/budget.json",
        "history/status.json",
        "history/evidence.json",
        "plan.json",
    ],
)
def test_mutated_authority_or_predecessor_refused_without_ledger_write(
    tmp_path, monkeypatch, name
):
    f = revision_fixture(tmp_path, monkeypatch)
    original = f["budget"].path.read_bytes()
    with (tmp_path / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        f["open_budget"](apply=True)
    assert f["budget"].path.read_bytes() == original


def test_concurrent_reset_adds_exactly_one_epoch_and_preserves_later_cost(
    tmp_path, monkeypatch
):
    f = revision_fixture(tmp_path, monkeypatch)
    errors = []

    def apply():
        try:
            f["open_budget"](apply=True)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=apply) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert not errors
    assert len(f["open_budget"]().snapshot()["epochs"]) == 3


def test_old_history_not_rewritable_after_reset(tmp_path, monkeypatch):
    from messaging_nlp.run_budget import _integrity

    f = revision_fixture(tmp_path, monkeypatch)
    f["open_budget"](apply=True)
    path = f["budget"].path
    document = json.loads(path.read_bytes())
    document["reservations"][0]["actual_seconds"] += 1
    document["integrity_sha256"] = _integrity(document)
    path.write_text(json.dumps(document))
    with pytest.raises(BudgetLedgerError, match="history"):
        f["open_budget"]()


def test_new_charge_cannot_be_moved_to_a_prior_epoch(tmp_path, monkeypatch):
    from messaging_nlp.run_budget import _integrity

    f = revision_fixture(tmp_path, monkeypatch)
    budget = f["open_budget"](apply=True)
    identifier = budget.reserve("preparation", 1)
    budget.settle(identifier, "finished", 1)
    path = budget.path
    document = json.loads(path.read_bytes())
    document["reservations"][-1]["epoch"] = 1
    document["integrity_sha256"] = _integrity(document)
    path.write_text(json.dumps(document))
    with pytest.raises(BudgetLedgerError, match="moved new charges"):
        f["open_budget"]()
