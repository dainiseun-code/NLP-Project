"""Fictional sealed run builder. Never opens a retained model or research data."""

from pathlib import Path
from zoneinfo import TZPATH

from messaging_nlp.application_types import PreparationPacket
from messaging_nlp.context import PublicDataset
from messaging_nlp.contracts import Extraction, ExtractionResult
from messaging_nlp.encoder_cache import (
    ExtractionArtifact,
    retained_identity,
    write_artifact,
)
from messaging_nlp.io import canonical_json
from messaging_nlp.preparation import digest, policy_identity, prepare_inputs
from messaging_nlp.prompts import build_prompt
from messaging_nlp.run_contracts import DecodePolicy, RunPlan, TokenReceipt
from messaging_nlp.run_storage import ref, write_new
from messaging_nlp.temporal import TimezoneBinding

ROOT = Path(__file__).resolve().parents[1]


def fake_plan(tmp_path, *, assisted=False):
    root = tmp_path
    request = dict(
        message_id="fiction",
        text="Send a note now.",
        owner_id="owner",
        request_time="2026-10-27T12:00:00Z",
        time_zone="UTC",
        context_snapshot_id="snapshot",
    )
    dataset = PublicDataset.model_validate(
        dict(
            owner_id="owner",
            requests=[request],
            snapshots=[
                dict(
                    context_snapshot_id="snapshot",
                    timestamp=request["request_time"],
                    record_ids=[],
                )
            ],
            contacts=[],
            calendar=[],
            history=[],
        )
    )
    for name in ("requests", "snapshots", "contacts", "calendar", "history"):
        write_new(
            root / f"public/{name}.jsonl",
            b"".join(
                canonical_json(r.model_dump()) + b"\n" for r in getattr(dataset, name)
            ),
        )
    tz = next(Path(p) / "UTC" for p in TZPATH if (Path(p) / "UTC").is_file())
    write_new(root / "UTC.tzif", tz.read_bytes())
    timezone = ref(root, root / "UTC.tzif")
    binding = TimezoneBinding("UTC", root / "UTC.tzif", timezone.sha256)
    identity = retained_identity()  # Identity only, never opens retained assets.
    if assisted:
        (root / "prep").mkdir(parents=True)
        result = ExtractionResult(
            message_id="fiction",
            status="ok",
            extraction=Extraction(entities=[]),
            failure=None,
            raw_output="fictional",
            input_sha256=digest(request["text"].encode()),
            extractor_sha256=identity.digest(),
        )
        write_artifact(
            root / "prep/extraction.json",
            ExtractionArtifact(
                text=request["text"], identity=identity, result=result, diagnostics={}
            ),
        )
    # Missing fictional extraction deliberately blocks assisted rows by default.
    prepared = prepare_inputs(
        dataset,
        "fiction",
        root=root,
        baseline_path="prep/base.json",
        assisted_path="prep/shared.json",
        extraction_path="prep/extraction.json",
        identity=identity,
        binding=binding,
    )
    packet = PreparationPacket.model_validate_json(prepared.baseline.data)
    prompt = build_prompt(packet, "baseline")
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user.decode()},
    ]
    write_new(root / "prompts/base.json", canonical_json(messages))
    template = digest(b"fictional-template")
    write_new(root / "models/fictional/chat_template.jinja", b"fictional-template")
    model_receipt = root / "model-receipt.json"
    write_new(model_receipt, canonical_json({"fictional": True}))
    token = TokenReceipt(
        messages_sha256=digest(canonical_json(messages)),
        template_sha256=template,
        model_receipt_sha256=ref(root, model_receipt).sha256,
        token_ids_sha256=digest(canonical_json([1, 2, 3])),
        token_count=3,
        token_ids=[1, 2, 3],
    )
    write_new(root / "prompts/base-tokens.json", canonical_json(token.model_dump()))
    if assisted:
        for condition, name in (("B", "b"), ("A+B", "ab")):
            packet = PreparationPacket.model_validate_json(prepared.assisted.data)
            built = build_prompt(packet, condition)
            chat = [
                {"role": "system", "content": built.system},
                {"role": "user", "content": built.user.decode()},
            ]
            write_new(root / f"prompts/{name}.json", canonical_json(chat))
            receipt = token.model_copy(
                update={"messages_sha256": digest(canonical_json(chat))}
            )
            write_new(
                root / f"prompts/{name}-tokens.json",
                canonical_json(receipt.model_dump()),
            )
    write_new(
        root / "roster.jsonl",
        canonical_json(
            dict(
                message_id="fiction",
                scenario_id="scenario-1",
                family_id="family-1",
                variant_type="clean",
                category_tags=[],
                text_sha256=digest(request["text"].encode()),
                human_review_status="pending_not_human_annotated",
                condition_order=["B", "baseline", "A+B"],
            )
        )
        + b"\n",
    )
    # Software manifests, not copied runtime dependencies or imported source trees.
    for name in (
        "pyproject.toml",
        "uv.lock",
        "environments/qwen/pyproject.toml",
        "environments/qwen/uv.lock",
        "environments/encoder/pyproject.toml",
        "environments/encoder/uv.lock",
        "configs/study-policy.json",
    ):
        write_new(root / name, (ROOT / name).read_bytes())
    for source in (ROOT / "src/messaging_nlp").glob("*.py"):
        write_new(root / "src/messaging_nlp" / source.name, source.read_bytes())
    for name in (
        "docs/stage1-budget-opening.json",
        "docs/budget-usage-reset-20261027.json",
    ):
        write_new(root / name, (ROOT / name).read_bytes())
    rows = []
    for ordinal, condition in enumerate(["B", "baseline", "A+B"]):
        ready = condition == "baseline" or assisted
        prompt_name = {"baseline": "base", "B": "b", "A+B": "ab"}[condition]
        rows.append(
            dict(
                row_id=f"row-{ordinal}",
                ordinal=ordinal,
                condition=condition,
                request=request,
                preparation=ref(
                    root,
                    root
                    / (
                        "prep/base.json"
                        if condition == "baseline"
                        else "prep/shared.json"
                    ),
                ).model_dump()
                if ready
                else None,
                preparation_failure=None
                if ready
                else prepared.assisted_failure.model_dump(),
                prompt=ref(root, root / f"prompts/{prompt_name}.json").model_dump()
                if ready
                else None,
                token_receipt=ref(
                    root, root / f"prompts/{prompt_name}-tokens.json"
                ).model_dump()
                if ready
                else None,
            )
        )
    frozen = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        role = (
            "source"
            if name.startswith("src/")
            else "lock"
            if name.endswith("uv.lock")
            else "environment"
            if name.endswith("pyproject.toml")
            else "other"
        )
        frozen.append(dict(**ref(root, path).model_dump(), role=role))
    plan = RunPlan(
        run_id="fictional-run",
        output_path="outputs/run",
        readiness_path="ready.json",
        approval_path="approval.json",
        opening=ref(root, root / "docs/stage1-budget-opening.json"),
        usage_reset=ref(root, root / "docs/budget-usage-reset-20261027.json"),
        public_root="public",
        owner_id="owner",
        timezone_name="UTC",
        timezone=timezone,
        roster=ref(root, root / "roster.jsonl"),
        frozen_files=frozen,
        model=dict(
            path="models/fictional",
            receipt=ref(root, model_receipt),
            template_sha256=template,
        ),
        spending=dict(
            startup_seconds=5,
            overhead_seconds=15,
            max_run_seconds=180,
            storage_bytes=2_000_000,
            assistant_version="fictional-common-v1",
            assistant_version_refs={
                "template_sha256": template,
                "prompt_policy_sha256": policy_identity(),
                "decoding_sha256": digest(canonical_json(DecodePolicy().model_dump())),
            },
            extraction_version="fictional-extraction-v1" if assisted else None,
            extraction_version_refs={"extractor_sha256": identity.digest()}
            if assisted
            else {},
        ),
        rows=rows,
    )
    path = root / "plan.json"
    write_new(path, canonical_json(plan.model_dump()))
    return path, plan
