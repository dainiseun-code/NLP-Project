"""Real supervision boundary with fictional preparation, no study model/tokenizer."""

import sys
from pathlib import Path

from run_fixtures import fake_plan
from test_run_supervision import FictionalSensors

from messaging_nlp import stage5
from messaging_nlp.run_storage import ref
from messaging_nlp.run_supervision import ChildSession
from messaging_nlp.temporal import TimezoneBinding


def test_serial_fake_encoder_materialization_and_tokenizer_use_actual_supervision(
    tmp_path, monkeypatch
):
    from messaging_nlp.io import loads_json

    _, plan = fake_plan(tmp_path)
    source = Path(__file__).parent / "fixtures/fake_preparation_child.py"

    def child(command, output, sensors, limits):
        return ChildSession(
            [sys.executable, "-I", str(source)], output, sensors, limits
        )

    dependencies = stage5._Dependencies(
        boot=lambda: "darwin-kern-boottime:100:0",
        child=child,
        sensors=lambda root: FictionalSensors(),
    )
    monkeypatch.setattr(stage5, "boot_identity", dependencies.boot)
    monkeypatch.setattr(stage5, "MacSensors", lambda root: FictionalSensors())
    sources = stage5.preparation_sources(tmp_path, "public")
    request = plan.rows[0].request
    encoded = stage5._run_job(
        tmp_path,
        tmp_path / "encoder-job",
        mode="encoder",
        items=[
            dict(
                id="case-0",
                payload=dict(message_id=request.message_id, text=request.text),
            )
        ],
        sources=sources,
        max_seconds=60,
        _dependencies=dependencies,
    )
    assert encoded["status"] == "completed" and encoded["cleanup_verified"]
    binding = TimezoneBinding(
        "UTC", tmp_path / "UTC.tzif", ref(tmp_path, tmp_path / "UTC.tzif").sha256
    )
    with stage5.accounted_preparation(
        tmp_path, tmp_path / "materialized", phase="materialize"
    ):
        prepared = stage5.materialize(
            tmp_path,
            tmp_path / "materialized",
            encoded,
            public_root="public",
            owner_id="owner",
            binding=binding,
        )
    tokenized = stage5._run_job(
        tmp_path,
        tmp_path / "tokenizer-job",
        mode="tokenizer",
        items=prepared["token_items"],
        sources=sources,
        model=plan.model,
        max_seconds=60,
        _dependencies=dependencies,
    )
    assert tokenized["status"] == "completed" and tokenized["cleanup_verified"]
    assert len(tokenized["outcomes"]) == 3
    for outcome in tokenized["outcomes"]:
        assert outcome["result_status"] == "fits"
        assert (
            loads_json((tmp_path / outcome["token_receipt"]["path"]).read_text())[
                "token_count"
            ]
            == 3
        )
    assert (tmp_path / "encoder-job/worker/stdout.log").exists()
    assert (tmp_path / "tokenizer-job/worker/resources.jsonl").stat().st_size > 0
    assert not (tmp_path / stage5.MUTEX_PATH).exists()

    # Finish this fictional chain through freeze, actual4A receipts and offline
    # review replay. No retained model is available to this software fixture.
    from test_offline_integration import fictional_reviews
    from test_reproduction import write_evaluator
    from test_run_process import dependencies as fake_generation

    from messaging_nlp.io import canonical_json
    from messaging_nlp.reproduction import build_bundle, reproduce
    from messaging_nlp.runner import _run_plan

    evaluator = write_evaluator(tmp_path / "eval")
    (tmp_path / "docs/stage4b-evaluation-contract.md").write_bytes(
        Path("docs/stage4b-evaluation-contract.md").read_bytes()
    )
    (tmp_path / "configs/evaluation-policy.json").write_bytes(
        evaluator["policy"].read_bytes()
    )
    policy = loads_json((tmp_path / "configs/study-policy.json").read_text())
    policy["data"]["public_root"] = "public"
    policy["data"]["roster"] = "roster.jsonl"
    policy["temporal"]["timezone_identity"]["tzif_path"] = "UTC.tzif"
    (tmp_path / "configs/study-policy.json").write_bytes(canonical_json(policy))
    frozen = stage5.freeze(
        tmp_path,
        tmp_path / "freeze",
        materialized=prepared,
        tokenized=tokenized,
        phase_directories=[
            tmp_path / name for name in ("encoder-job", "materialized", "tokenizer-job")
        ],
        model=plan.model,
        run_id="prepared-fictional-run",
        output_path="outputs/final",
        readiness_path="runs/ready.json",
        approval_path="runs/approve.json",
        _model_checker=lambda root, model: {},
        _fixture=dict(owner_id="owner", timezone_name="UTC", data_root="eval"),
    )
    assert frozen["qwen_generation"] is False
    assert not (tmp_path / "runs/approve.json").exists()
    ready = loads_json((tmp_path / "runs/ready.json").read_text())
    assert ready["boot_identity"] == "darwin-kern-boottime:100:0"
    generated = _run_plan(
        tmp_path / frozen["plan"]["path"],
        approve_run=True,
        dependencies=fake_generation("application"),
    )
    assert generated["status"] == "completed"
    run = tmp_path / "outputs/final"
    records = fictional_reviews(run, evaluator)
    build_bundle(
        run,
        tmp_path / "offline-bundle",
        requirements_path=evaluator["requirements"],
        roster_path=evaluator["roster"],
        gold_path=evaluator["gold"],
        policy_path=evaluator["policy"],
        review_records=records,
    )
    replay = reproduce(tmp_path / "offline-bundle/manifest.json")
    assert all(row["success"] is True for row in replay["rows"])
    assert len(replay["extraction"]) == 1
