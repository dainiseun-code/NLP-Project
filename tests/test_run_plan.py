"""Requirements-first frozen identity, permission and collision checks."""

from pathlib import Path

import pytest
from run_fixtures import fake_plan

from messaging_nlp.io import canonical_json
from messaging_nlp.preparation import digest
from messaging_nlp.run_contracts import ExecutionApproval, ReadinessReceipt
from messaging_nlp.run_storage import write_new


@pytest.fixture
def sealed(tmp_path):
    return fake_plan(tmp_path)


def validated(path):
    from messaging_nlp.run_plan import load_and_validate

    return load_and_validate(path, model_checker=lambda root, model: {"fake": True})


def permit(path, plan, boot="darwin-kern-boottime:100:0", *, approval=True):
    from messaging_nlp.run_budget import BudgetLedger

    ledger_path = path.parent / plan.ledger_path
    BudgetLedger(
        ledger_path,
        path.parent / plan.opening.path,
        plan.opening.sha256,
        reset_path=path.parent / plan.usage_reset.path,
        reset_sha256=plan.usage_reset.sha256,
    )
    ready = ReadinessReceipt(
        plan_sha256=digest(path.read_bytes()),
        boot_identity=boot,
        ledger_sha256=digest(ledger_path.read_bytes()),
    )
    raw = canonical_json(ready.model_dump())
    write_new(path.parent / plan.readiness_path, raw)
    if not approval:
        return
    approval = ExecutionApproval(
        plan_sha256=ready.plan_sha256,
        readiness_sha256=digest(raw),
        approved_by="fictional-user",
    )
    write_new(path.parent / plan.approval_path, canonical_json(approval.model_dump()))


def test_order_and_blocked_rows_preserved(sealed):
    path, plan = sealed
    value = validated(path)
    assert [r.condition for r in value.plan.rows] == ["B", "baseline", "A+B"]
    assert value.prepared["fiction"].assisted is None
    assert value.prompts["row-1"].user
    assert value.plan == plan


def test_actual_shared_assisted_bytes_and_prompt_only_entities_differ(tmp_path):
    path, plan = fake_plan(tmp_path, assisted=True)
    value = validated(path)
    assert value.prepared["fiction"].assisted is not None
    assert plan.rows[0].preparation == plan.rows[2].preparation
    assert value.prompts["row-0"].common == value.prompts["row-2"].common
    assert value.prompts["row-0"].entities is None
    assert value.prompts["row-2"].entities == b'{"entities":[]}'


@pytest.mark.parametrize("boot", [None, "", "junk", "darwin-kern-boottime:100:0"])
def test_missing_or_unchanged_boot_stops_before_authorization(sealed, boot):
    from messaging_nlp.run_plan import authorization

    path, plan = sealed
    permit(path, plan)
    with pytest.raises(ValueError, match="boot|restart"):
        authorization(validated(path), boot)


def test_changed_boot_alone_not_approval(sealed):
    from messaging_nlp.run_plan import authorization

    path, plan = sealed
    permit(path, plan)
    (path.parent / plan.approval_path).unlink()
    with pytest.raises(ValueError, match="approval"):
        authorization(validated(path), "darwin-kern-boottime:200:0")


def test_dry_run_no_state_or_approval_needed(sealed):
    from messaging_nlp.run_plan import authorization

    path, plan = sealed
    value = authorization(validated(path), "darwin-kern-boottime:100:0", dry_run=True)
    assert value["status"] == "restart_and_approval_required"
    assert not (path.parent / plan.ledger_path).exists()
    assert not (path.parent / plan.output_path).exists()


def test_changed_boot_and_bound_approval(sealed):
    from messaging_nlp.run_plan import authorization

    path, plan = sealed
    permit(path, plan)
    assert (
        authorization(validated(path), "darwin-kern-boottime:200:0")["status"]
        == "authorized"
    )


def test_deleted_preparation_ledger_cannot_reopen_reset(sealed):
    from messaging_nlp.run_plan import authorization

    path, plan = sealed
    permit(path, plan)
    (path.parent / plan.ledger_path).unlink()
    with pytest.raises(ValueError, match="ledger is missing"):
        authorization(validated(path), "darwin-kern-boottime:200:0")


def test_dry_run_does_not_hide_stale_ledger_as_restart_only(sealed):
    from messaging_nlp.run_plan import authorization

    path, plan = sealed
    permit(path, plan)
    (path.parent / plan.ledger_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="ledger changed"):
        authorization(validated(path), "darwin-kern-boottime:100:0", dry_run=True)


@pytest.mark.parametrize(
    "target",
    [
        "public/requests.jsonl",
        "prep/base.json",
        "prompts/base.json",
        "prompts/base-tokens.json",
        "roster.jsonl",
        "uv.lock",
        "src/messaging_nlp/application.py",
        "docs/budget-usage-reset-20261027.json",
    ],
)
def test_mutation_fails_closed(sealed, target):
    path, _ = sealed
    with (path.parent / target).open("ab") as out:
        out.write(b" ")
    with pytest.raises(ValueError, match="identity|frozen|hash"):
        validated(path)


@pytest.mark.parametrize(
    "output",
    [
        "public",
        "public/requests.jsonl/child",
        "src",
        "models",
        "prep/base.json",
        "runs/study-state/budget.json/child",
        "runs/study-state",
        ".",
    ],
)
def test_output_disjoint_both_directions(sealed, output):
    path, plan = sealed
    data = plan.model_dump()
    data["output_path"] = output
    path.write_bytes(canonical_json(data))
    with pytest.raises(ValueError):
        validated(path)


def test_collision_never_overwrites(sealed):
    from messaging_nlp.run_plan import create_output

    path, plan = sealed
    value = validated(path)
    destination = path.parent / plan.output_path
    destination.mkdir(parents=True)
    (destination / "sentinel").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        create_output(value)
    assert (destination / "sentinel").read_bytes() == b"keep"


def test_approval_cannot_overlap_frozen_input(sealed):
    path, plan = sealed
    data = plan.model_dump()
    data["approval_path"] = "public/requests.jsonl"
    path.write_bytes(canonical_json(data))
    with pytest.raises(ValueError, match="overlap"):
        validated(path)


def test_output_retains_exact_all_frozen_bytes(sealed):
    from messaging_nlp.run_plan import create_output

    path, _ = sealed
    value = validated(path)
    output = create_output(value)
    assert isinstance(output, Path)
    assert (output / "plan.json").read_bytes() == path.read_bytes()
    for name, raw in value.frozen_bytes.items():
        assert (output / "frozen" / name).read_bytes() == raw


def test_consent_writes_bound_immutable_approval_only_after_restart(sealed):
    from messaging_nlp.run_plan import approve_execution, authorization

    path, plan = sealed
    permit(path, plan, approval=False)
    value = validated(path)
    before = (path.parent / plan.ledger_path).read_bytes()
    assert (
        approve_execution(value, "darwin-kern-boottime:200:0")["status"] == "authorized"
    )
    approval_path = path.parent / plan.approval_path
    raw = approval_path.read_bytes()
    approval = ExecutionApproval.model_validate_json(raw)
    assert approval.approved_by == "terminal_user"
    assert approval.plan_sha256 == digest(path.read_bytes())
    assert approval.readiness_sha256 == digest(
        (path.parent / plan.readiness_path).read_bytes()
    )
    assert authorization(value, "darwin-kern-boottime:200:0")["status"] == "authorized"
    with pytest.raises(FileExistsError):
        approve_execution(value, "darwin-kern-boottime:200:0")
    assert approval_path.read_bytes() == raw
    assert (path.parent / plan.ledger_path).read_bytes() == before
    assert not (path.parent / plan.output_path).exists()
    assert not (path.parent / plan.mutex_path).exists()


@pytest.mark.parametrize(
    "boot",
    [None, "", "junk", "darwin-kern-boottime:100:0", "darwin-kern-boottime:0200:0"],
)
def test_consent_invalid_or_unchanged_boot_creates_nothing(sealed, boot):
    from messaging_nlp.run_plan import approve_execution

    path, plan = sealed
    permit(path, plan, approval=False)
    value = validated(path)
    before = (path.parent / plan.ledger_path).read_bytes()
    with pytest.raises(ValueError, match="boot|restart"):
        approve_execution(value, boot)
    assert not (path.parent / plan.approval_path).exists()
    assert not (path.parent / plan.output_path).exists()
    assert (path.parent / plan.ledger_path).read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing-readiness",
        "bad-readiness",
        "stale-plan",
        "invalid-ready-boot",
        "missing-ledger",
        "changed-ledger",
        "changed-plan",
        "changed-frozen",
    ],
)
def test_consent_authenticates_inputs_before_any_approval_write(sealed, fault):
    from messaging_nlp.run_plan import approve_execution

    path, plan = sealed
    permit(path, plan, approval=False)
    value = validated(path)
    ready_path = path.parent / plan.readiness_path
    ledger_path = path.parent / plan.ledger_path
    if fault == "missing-readiness":
        ready_path.unlink()
    elif fault == "bad-readiness":
        ready_path.write_bytes(b"{}")
    elif fault in ("stale-plan", "invalid-ready-boot"):
        data = ReadinessReceipt.model_validate_json(
            ready_path.read_bytes()
        ).model_dump()
        data["plan_sha256" if fault == "stale-plan" else "boot_identity"] = (
            "0" * 64 if fault == "stale-plan" else "junk"
        )
        ready_path.write_bytes(canonical_json(data))
    elif fault == "missing-ledger":
        ledger_path.unlink()
    elif fault == "changed-ledger":
        ledger_path.write_bytes(b"changed")
    elif fault == "changed-plan":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        (path.parent / "prep/base.json").write_bytes(b"changed")
    before = ledger_path.read_bytes() if ledger_path.exists() else None
    with pytest.raises((ValueError, OSError)):
        approve_execution(value, "darwin-kern-boottime:200:0")
    assert not (path.parent / plan.approval_path).exists()
    assert not (path.parent / plan.output_path).exists()
    assert (ledger_path.read_bytes() if ledger_path.exists() else None) == before


def test_consent_output_collision_does_not_create_approval(sealed):
    from messaging_nlp.run_plan import approve_execution

    path, plan = sealed
    permit(path, plan, approval=False)
    (path.parent / plan.output_path).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        approve_execution(validated(path), "darwin-kern-boottime:200:0")
    assert not (path.parent / plan.approval_path).exists()


@pytest.mark.parametrize("target", ["readiness_path", "ledger_path", "plan"])
def test_consent_rechecks_bindings_after_exclusive_write(sealed, monkeypatch, target):
    from messaging_nlp import run_plan

    path, plan = sealed
    permit(path, plan, approval=False)
    value = validated(path)
    real_write = run_plan.write_new

    def racing_write(destination, raw):
        real_write(destination, raw)
        changed = path if target == "plan" else path.parent / getattr(plan, target)
        changed.write_bytes(changed.read_bytes() + b" ")

    monkeypatch.setattr(run_plan, "write_new", racing_write)
    with pytest.raises(ValueError):
        run_plan.approve_execution(value, "darwin-kern-boottime:200:0")
    # Failed race evidence is immutable too; no rollback/deletion of consent.
    assert (path.parent / plan.approval_path).exists()
    assert not (path.parent / plan.output_path).exists()


@pytest.mark.parametrize("field", ["plan_sha256", "ledger_path"])
def test_consent_does_not_trust_mutated_validated_identity(sealed, field):
    from dataclasses import replace

    from messaging_nlp.run_plan import approve_execution

    path, plan = sealed
    permit(path, plan, approval=False)
    value = validated(path)
    if field == "plan_sha256":
        value = replace(value, plan_sha256="0" * 64)
    else:
        value = replace(
            value,
            plan=plan.model_copy(update={"ledger_path": "runs/other-ledger.json"}),
        )
    with pytest.raises(ValueError):
        approve_execution(value, "darwin-kern-boottime:200:0")
    assert not (path.parent / plan.approval_path).exists()


@pytest.mark.parametrize("field", ["plan_sha256", "readiness_sha256"])
def test_consent_preserves_stale_existing_approval_and_authorization_refuses(
    sealed, field
):
    from messaging_nlp.run_plan import approve_execution, authorization

    path, plan = sealed
    permit(path, plan)
    approval_path = path.parent / plan.approval_path
    data = ExecutionApproval.model_validate_json(
        approval_path.read_bytes()
    ).model_dump()
    data[field] = "0" * 64
    raw = canonical_json(data)
    approval_path.write_bytes(raw)
    value = validated(path)
    with pytest.raises(FileExistsError):
        approve_execution(value, "darwin-kern-boottime:200:0")
    with pytest.raises(ValueError, match="stale execution approval"):
        authorization(value, "darwin-kern-boottime:200:0")
    assert approval_path.read_bytes() == raw


@pytest.mark.parametrize(
    "boot", [None, "darwin-kern-boottime:100:0", "darwin-kern-boottime:200:0"]
)
@pytest.mark.parametrize("approval", [False, True])
def test_consent_dry_statuses_are_never_execution_authorization(sealed, boot, approval):
    from messaging_nlp.run_plan import authorization

    path, plan = sealed
    permit(path, plan, approval=approval)
    ledger = (path.parent / plan.ledger_path).read_bytes()
    result = authorization(validated(path), boot, dry_run=True)
    changed = boot == "darwin-kern-boottime:200:0"
    expected = (
        ("ready" if approval else "approval_required")
        if changed
        else ("restart_required" if approval else "restart_and_approval_required")
    )
    assert result["status"] == expected
    assert result["execution_authorized"] is False
    assert (path.parent / plan.approval_path).exists() == approval
    assert (path.parent / plan.ledger_path).read_bytes() == ledger
    assert not (path.parent / plan.output_path).exists()


def test_consent_dry_invalid_current_boot_does_not_report_ready(sealed):
    from messaging_nlp.run_plan import authorization

    path, plan = sealed
    permit(path, plan)
    with pytest.raises(ValueError, match="boot"):
        authorization(validated(path), "junk", dry_run=True)


def test_consent_exclusive_write_collision_preserves_other_approval(
    sealed, monkeypatch
):
    from messaging_nlp import run_plan

    path, plan = sealed
    permit(path, plan, approval=False)
    value = validated(path)
    real_write = run_plan.write_new
    other = b"immutable competing consent evidence"

    def racing_write(destination, raw):
        real_write(destination, other)
        real_write(destination, raw)

    monkeypatch.setattr(run_plan, "write_new", racing_write)
    with pytest.raises(FileExistsError):
        run_plan.approve_execution(value, "darwin-kern-boottime:200:0")
    assert (path.parent / plan.approval_path).read_bytes() == other
    assert not (path.parent / plan.output_path).exists()


@pytest.mark.parametrize("target", ["readiness_path", "ledger_path"])
def test_consent_rechecks_readiness_and_ledger_before_write(
    sealed, monkeypatch, target
):
    from messaging_nlp import run_plan

    path, plan = sealed
    permit(path, plan, approval=False)
    value = validated(path)
    real_recheck = run_plan._recheck_bound_inputs
    calls = 0

    def racing_recheck(validated_value):
        nonlocal calls
        real_recheck(validated_value)
        calls += 1
        if calls == 2:
            changed = path.parent / getattr(plan, target)
            changed.write_bytes(changed.read_bytes() + b" ")

    monkeypatch.setattr(run_plan, "_recheck_bound_inputs", racing_recheck)
    with pytest.raises(ValueError):
        run_plan.approve_execution(value, "darwin-kern-boottime:200:0")
    assert not (path.parent / plan.approval_path).exists()
    assert not (path.parent / plan.output_path).exists()
