"""Independent once-only preparation tests with injected children, no model calls."""

from pathlib import Path

import pytest
from run_fixtures import fake_plan
from test_run_supervision import FictionalSensors

from messaging_nlp import stage5
from messaging_nlp.contracts import Extraction, ExtractionResult
from messaging_nlp.encoder_cache import (
    ExtractionArtifact,
    retained_identity,
    text_sha256,
    write_artifact,
)
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.run_contracts import CleanupReceipt, WorkerEvent
from messaging_nlp.run_storage import ref
from messaging_nlp.temporal import TimezoneBinding


@pytest.fixture
def job(tmp_path):
    fake_plan(tmp_path)
    sources = stage5.preparation_sources(tmp_path, "public")
    children = []
    behavior = dict(fault=None, cleanup=True)

    class Child:
        failure = None

        def __init__(self, command, output, sensors, limits):
            assert command[-3:] == ["-I", "-m", "messaging_nlp.preparation_worker"]
            self.events, self.sent = [], []
            children.append(self)

        def start(self):
            pass

        def send(self, request, deadline):
            self.sent.append(request)
            if request.kind == "load":
                self.events = [
                    WorkerEvent(
                        kind="ready",
                        payload=dict(
                            mode="encoder",
                            worker_source_sha256=stage5._source_map(sources),
                            encoder_sha256=retained_identity().digest(),
                        ),
                    )
                ]
                return
            registered = [
                loads_json(line)
                for line in (tmp_path / stage5.REGISTRY).read_text().splitlines()
            ]
            entry = next(
                r
                for r in registered
                if r["attempt_key"] == stage5._attempt_key(request.payload)
            )
            assert (tmp_path / entry["destination"] / "intent.json").is_file()
            artifact = ExtractionArtifact(
                text=request.payload["text"],
                identity=retained_identity(),
                result=ExtractionResult(
                    message_id=request.payload["message_id"],
                    status="ok",
                    extraction=Extraction(entities=[]),
                    failure=None,
                    raw_output="fictional",
                    input_sha256=text_sha256(request.payload["text"]),
                    extractor_sha256=retained_identity().digest(),
                ),
                diagnostics={"forward_performed": True},
            )
            data = artifact.model_dump()
            if behavior["fault"] == "bad_identity":
                data["result"]["input_sha256"] = "0" * 64
            self.events = [
                WorkerEvent(
                    kind="generation_started",
                    row_id=request.row_id,
                    payload=dict(operation="encoder_extraction", qwen_generation=False),
                )
            ]
            if behavior["fault"] != "timeout":
                self.events += [
                    WorkerEvent(
                        kind="fragment",
                        row_id=request.row_id,
                        payload={
                            "text": canonical_json(
                                dict(mode="encoder", artifact=data)
                            ).decode()
                        },
                    ),
                    WorkerEvent(
                        kind="terminal",
                        row_id=request.row_id,
                        payload=dict(
                            finish_reason="stop",
                            generation_tokens=0,
                            input_tokens=3,
                            elapsed_seconds=0.01,
                        ),
                    ),
                ]

        def wait_event(self, deadline):
            if not self.events:
                raise TimeoutError("fictional stopped after start acknowledgement")
            return self.events.pop(0)

        def close(self):
            return CleanupReceipt(verified=behavior["cleanup"], returncode=0)

    deps = stage5._Dependencies(
        boot=lambda: "darwin-kern-boottime:100:0",
        child=Child,
        sensors=lambda root: FictionalSensors(),
    )
    items = [
        dict(
            id=f"case-{i}",
            payload=dict(message_id=f"fiction-{i}", text=f"Noisy 📨 request {i}"),
        )
        for i in range(2)
    ]

    def run(name="job"):
        return stage5._run_job(
            tmp_path,
            tmp_path / name,
            mode="encoder",
            items=items,
            sources=sources,
            max_seconds=30,
            _dependencies=deps,
        )

    return tmp_path, items, children, behavior, run


def test_once_only_inputs_empty_success_shared_artifacts_and_zero_generations(job):
    root, items, children, _, run = job
    result = run()
    assert result["status"] == "completed" and result["cleanup_verified"]
    assert result["qwen_generations"] == 0
    assert [o["disposition"] for o in result["outcomes"]] == ["completed", "completed"]
    assert all(o["result_status"] == "ok" for o in result["outcomes"])
    assert len([r for r in children[0].sent if r.kind == "generate"]) == len(items)
    assert not (root / stage5.MUTEX_PATH).exists()
    ledger = loads_json((root / stage5.LEDGER_PATH).read_text())
    assert all(r["actual_generations"] == 0 for r in ledger["reservations"])
    again = run("different-destination")
    assert again["status"] == "failed" and "already attempted" in again["errors"][0]
    assert len(children) == 1  # refuse cross-destination replay BEFORE loading


@pytest.mark.parametrize("fault", ["timeout", "bad_identity"])
def test_failed_work_preserves_start_and_never_retries_remaining_requests(job, fault):
    root, _, children, behavior, run = job
    behavior["fault"] = fault
    result = run()
    assert result["status"] == "failed"
    assert result["outcomes"][0]["disposition"] == "attempted_unresolved"
    assert result["outcomes"][0]["operation_started"]
    assert result["outcomes"][1]["disposition"] == "unattempted"
    assert len([r for r in children[0].sent if r.kind == "generate"]) == 1
    assert len((root / stage5.REGISTRY).read_text().splitlines()) == 1
    assert not (root / stage5.MUTEX_PATH).exists()


def test_failed_outcome_commit_is_not_promoted_by_visible_artifact_bytes(
    job, monkeypatch
):
    root, _, _, _, run = job
    original = stage5.write_new

    def fail_after_write(path, data):
        original(path, data)
        if Path(path).name == "outcome.json":
            raise OSError("original injected outcome-fsync failure")

    monkeypatch.setattr(stage5, "write_new", fail_after_write)
    result = run()
    assert result["status"] == "failed"
    assert "original injected" in result["errors"][0]
    assert result["outcomes"][0]["disposition"] == "attempted_unresolved"
    assert (root / "job/items/000000/outcome.json").exists()
    assert not (root / stage5.MUTEX_PATH).exists()


def test_unresolved_cleanup_retains_lock_and_blocks_any_following_job(job):
    root, _, _, behavior, run = job
    behavior["cleanup"] = False
    result = run()
    assert result["status"] == "failed" and not result["cleanup_verified"]
    assert (root / stage5.MUTEX_PATH).is_dir()
    with pytest.raises(RuntimeError, match="mutex exists"):
        run("another-job")


def test_ledger_failure_cannot_skip_model_cleanup_or_mutex_disposition(
    job, monkeypatch
):
    root, _, _, _, run = job
    monkeypatch.setattr(
        stage5.BudgetLedger,
        "settle",
        lambda *a, **k: (_ for _ in ()).throw(OSError("original ledger failure")),
    )
    result = run()
    assert result["status"] == "failed" and result["cleanup_verified"]
    assert "original ledger failure" in result["errors"][0]
    assert not (root / stage5.MUTEX_PATH).exists()


def test_single_selected_request_cannot_appear_twice_under_different_item_ids(job):
    _, items, children, _, run = job
    items[1]["payload"] = dict(items[0]["payload"])
    with pytest.raises(ValueError, match="duplicate encoder"):
        run()
    assert not children


def test_storage_settlement_failure_cannot_leave_durable_completed_status(
    job, monkeypatch
):
    root, _, _, _, run = job
    monkeypatch.setattr(
        stage5.BudgetLedger,
        "settle_storage",
        lambda *a, **k: (_ for _ in ()).throw(OSError("storage settlement failed")),
    )
    result = run()
    retained = loads_json((root / "job/status.json").read_text())
    assert result["status"] == retained["status"] == "failed"
    assert "storage settlement failed" in retained["errors"][0]
    assert not (root / stage5.MUTEX_PATH).exists()


def test_mutex_disposition_failure_does_not_replace_the_original_error(
    job, monkeypatch
):
    _, _, _, behavior, run = job
    behavior["fault"] = "timeout"
    monkeypatch.setattr(
        stage5.ModelMutex,
        "release",
        lambda *a, **k: (_ for _ in ()).throw(OSError("secondary release fault")),
    )
    result = run()
    assert result["status"] == "failed"
    assert "stopped after start" in result["errors"][0]
    assert "secondary release fault" in result["errors"][-1]


def test_materialization_shared_actual_bytes_and_separate_costs_are_model_free(
    tmp_path, monkeypatch
):
    fake_plan(tmp_path)
    request = loads_json((tmp_path / "public/requests.jsonl").read_text().strip())
    identity = retained_identity()
    artifact = ExtractionArtifact(
        text=request["text"],
        identity=identity,
        result=ExtractionResult(
            message_id=request["message_id"],
            status="ok",
            extraction=Extraction(entities=[]),
            failure=None,
            raw_output="fictional",
            input_sha256=text_sha256(request["text"]),
            extractor_sha256=identity.digest(),
        ),
        diagnostics={},
    )
    path = tmp_path / "fictional-extraction.json"
    write_artifact(path, artifact)
    status = dict(
        status="completed",
        cleanup_verified=True,
        outcomes=[
            dict(
                payload=dict(message_id=request["message_id"], text=request["text"]),
                artifact=ref(tmp_path, path).model_dump(),
            )
        ],
    )
    binding = TimezoneBinding(
        "UTC", tmp_path / "UTC.tzif", ref(tmp_path, tmp_path / "UTC.tzif").sha256
    )
    monkeypatch.setattr(stage5, "boot_identity", lambda: "darwin-kern-boottime:100:0")
    monkeypatch.setattr(stage5, "MacSensors", lambda root: FictionalSensors())
    with stage5.accounted_preparation(
        tmp_path, tmp_path / "materialized", phase="materialization"
    ):
        result = stage5.materialize(
            tmp_path,
            tmp_path / "materialized",
            status,
            public_root="public",
            owner_id="owner",
            binding=binding,
        )
    case = result["rows"][0]
    assert result["record_budget"] == 8
    assert case["shared"] is not None and len(result["token_items"]) == 3
    assert all(value >= 0 for value in case["timings"].values())
    assert "baseline_search_and_save_seconds" in case["timings"]
    assert "shared_assisted_read_search_and_save_seconds" in case["timings"]
    b = loads_json((tmp_path / case["prompts"]["B"]["prompt"]["path"]).read_text())
    ab = loads_json((tmp_path / case["prompts"]["A+B"]["prompt"]["path"]).read_text())
    assert ab[1]["content"].startswith(b[1]["content"] + "\nENTITY ANNOTATIONS\n")
    assert (tmp_path / "materialized/accounting.json").is_file()
    assert not (tmp_path / stage5.MUTEX_PATH).exists()


def test_materialization_cannot_accidentally_omit_preparation_accounting(job):
    root, _, _, _, _ = job
    with pytest.raises(ValueError, match="active preparation accounting"):
        stage5.materialize(
            root,
            root / "unaccounted",
            dict(status="completed", cleanup_verified=True),
            public_root="public",
            owner_id="owner",
            binding=None,
        )
    assert not (root / "unaccounted").exists()


def test_freeze_rejects_record_budget_drift_before_creating_readiness(job):
    root, _, _, _, _ = job
    with pytest.raises(ValueError, match="record budget differs"):
        stage5.freeze(
            root,
            root / "freeze-drift",
            materialized={"record_budget": 7},
            tokenized={},
            phase_directories=[],
            model=None,
            run_id="unused",
            output_path="outputs/unused",
            readiness_path="ready.json",
            approval_path="approve.json",
        )
    assert not (root / "ready.json").exists()
