"""Lead-only serial fake subprocess lease; never starts an optional runtime."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from run_fixtures import fake_plan
from test_run_plan import permit
from test_run_supervision import FictionalSensors

from messaging_nlp.run_contracts import load_row_receipt
from messaging_nlp.run_supervision import ChildSession
from messaging_nlp.runner import Dependencies, _run_plan

FIXTURE = Path(__file__).parent / "fixtures/fake_qwen_child.py"
CURRENT_BOOT = "darwin-kern-boottime:200:0"


def dependencies(mode):
    def child(command, output, sensors, limits, *, env=None):
        assert command[-3:] == ["-I", "-m", "messaging_nlp.qwen_worker"]
        return ChildSession(
            [sys.executable, "-I", str(FIXTURE), mode], output, sensors, limits, env=env
        )

    return Dependencies(
        boot=lambda: CURRENT_BOOT,
        sensors_factory=lambda root: FictionalSensors(),
        child_factory=child,
        model_checker=lambda root, model: {"fictional": True},
    )


def test_process_progress_journal_failure_stops_promptly_with_named_component(
    tmp_path, monkeypatch
):
    from messaging_nlp import run_progress
    from messaging_nlp.run_progress import RunProgress

    path, plan = fake_plan(tmp_path, assisted=True)
    permit(path, plan, approval=False)
    original = run_progress.append_bytes

    def fail_prefill(journal, raw):
        if journal.name == "progress.jsonl" and json.loads(raw)["phase"] == "prefill":
            raise OSError("fictional progress journal failure")
        return original(journal, raw)

    monkeypatch.setattr(run_progress, "append_bytes", fail_prefill)
    with RunProgress(interval_seconds=0.01, console_fd=None) as reporter:
        result = _run_plan(
            path,
            approve_run=True,
            dependencies=dependencies("progress_application"),
            reporter=reporter,
        )
    assert result["status"] == "failed" and result["cleanup_verified"]
    assert "progress_persistence" in result["stop_reason"]
    output = tmp_path / plan.output_path
    receipts = [
        load_row_receipt(output, p)[0]
        for p in sorted((output / "rows").glob("*/receipt.json"))
    ]
    assert [r.disposition for r in receipts] == [
        "attempted_failure",
        "unattempted_stop",
        "unattempted_stop",
    ]
    assert receipts[0].generation_started and not receipts[0].generation_completed
    assert receipts[0].charged_seconds >= 120


def test_process_explicit_terminal_consent_runs_once_and_keeps_immutable_approval(
    tmp_path,
):
    path, plan = fake_plan(tmp_path, assisted=True)
    permit(path, plan, approval=False)
    approval = tmp_path / plan.approval_path
    assert not approval.exists()
    result = _run_plan(path, approve_run=True, dependencies=dependencies("application"))
    assert result["status"] == "completed" and result["cleanup_verified"]
    approval_bytes = approval.read_bytes()
    bound = json.loads(approval_bytes)
    assert bound["plan_sha256"] == result["plan_sha256"]
    assert bound["approved_by"] == "terminal_user"
    output = tmp_path / plan.output_path
    requests = (output / "worker/requests.jsonl").read_bytes()
    ledger_bytes = (tmp_path / plan.ledger_path).read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        _run_plan(path, approve_run=True, dependencies=dependencies("application"))
    with pytest.raises((ValueError, FileExistsError)):
        _run_plan(path, dependencies=dependencies("application"))
    assert approval.read_bytes() == approval_bytes
    assert (output / "worker/requests.jsonl").read_bytes() == requests
    assert (tmp_path / plan.ledger_path).read_bytes() == ledger_bytes
    assert (
        len(
            [
                json.loads(line)
                for line in requests.splitlines()
                if json.loads(line)["kind"] == "generate"
            ]
        )
        == 3
    )


def test_process_runtime_ready_preparation_drift_blocks_before_any_generation(tmp_path):
    path, plan = fake_plan(tmp_path, assisted=True)
    permit(path, plan)
    original_deps = dependencies("application")
    factory = original_deps.child_factory

    def child(*args, **kwargs):
        value = factory(*args, **kwargs)
        wait = value.wait_event

        def wait_event(deadline):
            event = wait(deadline)
            if event.kind == "ready":
                preparation = tmp_path / plan.rows[0].preparation.path
                preparation.write_bytes(preparation.read_bytes() + b" ")
            return event

        value.wait_event = wait_event
        return value

    result = _run_plan(path, dependencies=replace(original_deps, child_factory=child))
    assert result["status"] == "failed" and result["cleanup_verified"]
    output = tmp_path / plan.output_path
    rows = [
        load_row_receipt(output, p)
        for p in sorted((output / "rows").glob("*/receipt.json"))
    ]
    assert [r.disposition for r, _, _ in rows] == [
        "preparation_blocked",
        "unattempted_stop",
        "unattempted_stop",
    ]
    assert all(r.provider_calls == 0 and r.reservation_id is None for r, _, _ in rows)
    assert rows[0][1].failure.stage == "preprocessing"
    assert "preparation rejected" in rows[0][1].failure.message
    requests = [
        json.loads(line)
        for line in (output / "worker/requests.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in requests] == ["load"]
    budget = json.loads((tmp_path / plan.ledger_path).read_bytes())
    assert not any(r["kind"] == "row" for r in budget["reservations"])
    assert any(
        r["kind"] == "startup" and r["actual_seconds"] > 0
        for r in budget["reservations"]
    )


@pytest.mark.parametrize(
    "gate",
    [
        "missing_boot",
        "unchanged_boot",
        "invalid_boot",
        "missing_readiness",
        "stale_readiness",
        "missing_ledger",
        "stale_ledger",
        "stale_plan",
        "missing_consent",
    ],
)
def test_consent_gate_failure_never_writes_approval_or_dispatches(tmp_path, gate):
    from messaging_nlp.io import canonical_json

    path, plan = fake_plan(tmp_path, assisted=True)
    permit(path, plan, approval=False)
    boot = CURRENT_BOOT
    if gate == "missing_boot":
        boot = None
    elif gate == "unchanged_boot":
        boot = "darwin-kern-boottime:100:0"
    elif gate == "invalid_boot":
        boot = "invalid-os-boot"
    elif gate == "missing_readiness":
        (tmp_path / plan.readiness_path).unlink()
    elif gate == "stale_readiness":
        ready = json.loads((tmp_path / plan.readiness_path).read_bytes())
        ready["plan_sha256"] = "0" * 64
        (tmp_path / plan.readiness_path).write_bytes(canonical_json(ready))
    elif gate == "missing_ledger":
        (tmp_path / plan.ledger_path).unlink()
    elif gate == "stale_ledger":
        with (tmp_path / plan.ledger_path).open("ab") as handle:
            handle.write(b" ")
    elif gate == "stale_plan":
        with path.open("ab") as handle:
            handle.write(b" ")
    ledger_path = tmp_path / plan.ledger_path
    before = ledger_path.read_bytes() if ledger_path.exists() else None
    attempts = []

    def forbidden_child(*args, **kwargs):
        attempts.append("child")
        raise AssertionError("gate failure launched a child")

    deps = replace(
        dependencies("application"), boot=lambda: boot, child_factory=forbidden_child
    )
    if gate == "missing_consent":
        assert _run_plan(path, dependencies=deps)["status"] == "failed"
    else:
        with pytest.raises((ValueError, OSError)):
            _run_plan(path, approve_run=True, dependencies=deps)
    assert attempts == []
    assert not (tmp_path / plan.approval_path).exists()
    assert (ledger_path.read_bytes() if ledger_path.exists() else None) == before


def test_dry_run_and_explicit_consent_are_mutually_exclusive_before_effects(tmp_path):
    path, plan = fake_plan(tmp_path)
    permit(path, plan, approval=False)
    ledger_path = tmp_path / plan.ledger_path
    before = ledger_path.read_bytes()
    with pytest.raises(ValueError):
        _run_plan(
            path,
            dry_run=True,
            approve_run=True,
            dependencies=dependencies("application"),
        )
    assert not (tmp_path / plan.approval_path).exists()
    assert not (tmp_path / plan.output_path).exists()
    assert ledger_path.read_bytes() == before


@pytest.mark.parametrize(
    "mode", ["application", "invalid_json", "application_length", "crash"]
)
def test_process_actual_coordinator_persists_all_rows_and_exact_wrappers(
    tmp_path, mode
):
    path, plan = fake_plan(tmp_path, assisted=True)
    permit(path, plan)
    result = _run_plan(path, dependencies=dependencies(mode))
    output = tmp_path / plan.output_path
    rows = [
        load_row_receipt(output, p)
        for p in sorted((output / "rows").glob("*/receipt.json"))
    ]
    assert len(rows) == result["rows_saved"] == result["planned_rows"] == 3
    assert [r.condition for r, _, _ in rows] == ["B", "baseline", "A+B"]
    assert result["cleanup_verified"]
    assert not (tmp_path / plan.mutex_path).exists()
    requests = [
        json.loads(line)
        for line in (output / "worker/requests.jsonl").read_text().splitlines()
    ]
    assert [r["kind"] for r in requests].count("load") == 1
    if mode == "crash":
        assert result["status"] == "failed"
        assert [r.disposition for r, _, _ in rows] == [
            "attempted_failure",
            "unattempted_stop",
            "unattempted_stop",
        ]
        assert rows[0][1].raw_output == '{"fictional":'
        assert rows[0][0].charged_seconds >= 120
        assert [r["kind"] for r in requests].count("generate") == 1
    else:
        assert result["status"] == "completed"
        assert all(
            r.generation_completed and r.charged_generations == 1 for r, _, _ in rows
        )
        assert [r["kind"] for r in requests].count("generate") == 3
        if mode == "application":
            assert all(row.status == "ok" and env is not None for _, row, env in rows)
        else:
            assert all(row.status == "failed" for _, row, _ in rows)
            if mode == "application_length":
                assert all(row.failure.code == "truncated" for _, row, _ in rows)
                assert all(json.loads(row.raw_output)["decision"] for _, row, _ in rows)
    # Earlier row snapshots stay authentic after later global logs have appended.
    for receipt, _, _ in rows:
        assert receipt.logs
    assert (output / "worker/stdout.log").read_bytes()
    assert (output / "worker/stderr.log").read_bytes()
    # Finished storage settles to observed retained bytes, not the allocation;
    # include the settlement's own canonical-ledger growth without double count.
    ledger_raw = (tmp_path / plan.ledger_path).read_bytes()
    ledger_document = json.loads(ledger_raw)
    storage = next(r for r in ledger_document["reservations"] if r["kind"] == "storage")
    output_size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    growth = max(
        0, len(ledger_raw) - (output / "budget-active-opening.json").stat().st_size
    )
    assert storage["actual_bytes"] == output_size + growth
    assert storage["actual_bytes"] < plan.spending.storage_bytes
    assert storage["status"] == ("failed" if mode == "crash" else "finished")


def test_process_standalone_private_fake_command_has_no_session_dependency(tmp_path):
    path, plan = fake_plan(tmp_path, assisted=True)
    permit(path, plan)
    script = """
import importlib.abc,json,sys,time
from pathlib import Path
class DenyModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','mlx','mlx_lm','transformers','tokenizers','sentencepiece'}:
   raise AssertionError('actual optional runtime import: '+fullname)
sys.meta_path.insert(0,DenyModels())
from messaging_nlp.runner import Dependencies,_run_plan
from messaging_nlp.run_contracts import ResourceSample
from messaging_nlp.run_supervision import ChildSession
class Sensors:
 def sample(self,pid):
  return ResourceSample(monotonic_seconds=time.monotonic(),pressure_level=1,
   swap_used_bytes=0,free_disk_bytes=10**12,tree_rss_bytes=0,tree_phys_footprint_bytes=0)
def child(command,output,sensors,limits,*,env=None):
 return ChildSession([sys.executable,'-I',sys.argv[2],'application'],output,sensors,limits,env=env)
deps=Dependencies(boot=lambda:'darwin-kern-boottime:200:0',sensors_factory=lambda root:Sensors(),
 child_factory=child,model_checker=lambda root,model:{'fictional':True})
result=_run_plan(Path(sys.argv[1]),dependencies=deps)
print(json.dumps(result))
raise SystemExit(0 if result['status']=='completed' else 1)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(path), str(FIXTURE.resolve())],
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["cleanup_verified"]
    assert len(list((tmp_path / plan.output_path / "rows").glob("*/receipt.json"))) == 3


def test_process_budget_atomic_cross_process_charges_survive_reopen(tmp_path):
    from messaging_nlp.run_budget import BudgetLedger

    path, plan = fake_plan(tmp_path)
    permit(path, plan)
    script = """
import sys
from messaging_nlp.run_budget import BudgetLedger
l=BudgetLedger(sys.argv[1],sys.argv[2],sys.argv[3],reset_path=sys.argv[4],reset_sha256=sys.argv[5])
r=l.reserve('row',120,generations=1,refs={'run_id':'shared-fictional-coordinator'})
l.settle(r,'finished',2,generations=1)
"""
    command = [
        sys.executable,
        "-I",
        "-c",
        script,
        str(tmp_path / plan.ledger_path),
        str(tmp_path / plan.opening.path),
        plan.opening.sha256,
        str(tmp_path / plan.usage_reset.path),
        plan.usage_reset.sha256,
    ]
    children = []
    try:
        for _ in range(2):
            children.append(
                subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            )
        for child in children:
            stdout, stderr = child.communicate(timeout=10)
            assert child.returncode == 0, stdout + stderr
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    ledger = BudgetLedger(
        tmp_path / plan.ledger_path,
        tmp_path / plan.opening.path,
        plan.opening.sha256,
        reset_path=tmp_path / plan.usage_reset.path,
        reset_sha256=plan.usage_reset.sha256,
    )
    assert ledger.snapshot()["usage"]["generations"] == 2
    assert ledger.snapshot()["usage"]["seconds"] == 4
    assert not ledger.snapshot()["unfinished_reservation_ids"]
