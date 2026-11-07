"""Eligible, once-only preparation and exclusive, immutable byte receipts.

Only saved extraction artifacts are consumed. No encoder is imported or invoked.
The caller supplies the explicit artifact root and trusted timezone/encoder identity.
"""

from __future__ import annotations

import hashlib
import os
import time
from importlib.resources import files
from pathlib import Path

from messaging_nlp.application_types import (
    AliasTarget,
    PreparationPacket,
    PreparedInputs,
    RequestAnchor,
    SealedInput,
)
from messaging_nlp.context import PublicBundle, PublicDataset
from messaging_nlp.contracts import TechnicalFailure
from messaging_nlp.encoder_cache import (
    EncoderIdentity,
    ExtractionArtifact,
    portable_path,
    read_artifact,
)
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.limits import MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS
from messaging_nlp.temporal import TimezoneBinding, request_zone


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_path(root: Path, relative: str) -> Path:
    """Reject absolute/traversing paths and symlinks escaping the explicit root."""
    portable_path(relative)
    base = Path(root).resolve(strict=True)
    path = base / relative
    if not path.resolve().is_relative_to(base):
        raise ValueError("artifact path leaves explicit project root")
    return path


def policy_identity() -> str:
    # Independently expected code-owned policies, never copied from a saved packet.
    from messaging_nlp.drafting import RENDERING_POLICY
    from messaging_nlp.prompts import PROMPT_POLICY

    return digest(
        canonical_json(
            {
                "preparation": "eligible-sealed-once-v1",
                "search": "authored-associated-original-channel-v1",
                "rendering": RENDERING_POLICY,
                "prompt": PROMPT_POLICY,
                "limits": {"input": MAX_INPUT_TOKENS, "output": MAX_OUTPUT_TOKENS},
                "implementation": {
                    name: digest(files("messaging_nlp").joinpath(name).read_bytes())
                    for name in (
                        "application_types.py",
                        "contracts.py",
                        "retrieval.py",
                        "role_hints.py",
                        "preparation.py",
                        "prompts.py",
                        "temporal_language.py",
                        "drafting.py",
                        "application.py",
                        "limits.py",
                    )
                },
            }
        )
    )


def make_aliases(
    bundle: PublicBundle, anchors: list[RequestAnchor]
) -> dict[str, AliasTarget]:
    """Order depends only on eligible opaque identity, not condition or category."""
    targets = {("owner", bundle.request.owner_id)}
    targets.update(("contact", row.contact_id) for row in bundle.context.contacts)
    targets.update(("event", row.event_id) for row in bundle.context.calendar)
    targets.update(
        ("record", row.record_id)
        for rows in (
            bundle.context.contacts,
            bundle.context.calendar,
            bundle.context.history,
        )
        for row in rows
    )
    targets.update(("anchor", anchor.anchor_id) for anchor in anchors)
    ordered = sorted(targets, key=lambda item: (digest(canonical_json(item)), item))
    return {
        f"x{index:04d}": AliasTarget(kind=kind, internal_id=internal)
        for index, (kind, internal) in enumerate(ordered, 1)
    }


def _save(root: Path, relative: str, packet: PreparationPacket) -> SealedInput:
    data = canonical_json(packet.model_dump())
    path = artifact_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Recheck following directory creation, including any preexisting symlinks.
    path = artifact_path(root, relative)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return SealedInput(relative, data, digest(data))


def prepare_inputs(
    dataset: PublicDataset,
    message_id: str,
    *,
    root: Path,
    baseline_path: str,
    assisted_path: str,
    extraction_path: str,
    identity: EncoderIdentity,
    binding: TimezoneBinding,
    max_records: int = 40,
    timings: dict[str, float] | None = None,
) -> PreparedInputs:
    """Search baseline once, assisted once; save the one shared assisted packet.

    Baseline eligibility/preparation errors raise. Assisted errors are explicit
    blocking failures, never substitutions. Destinations must be new and distinct.
    """
    from messaging_nlp.drafting import precompute_anchors
    from messaging_nlp.retrieval import search

    if timings is not None and type(timings) is not dict:
        raise ValueError("preparation timings require a plain caller-owned dictionary")

    # Revalidate mutable containers in the strict frozen Pydantic dataset.
    dataset = PublicDataset.model_validate(dataset.model_dump())
    bundle = dataset.bundle(message_id)
    request_zone(bundle.request, binding=binding)
    paths = [
        artifact_path(root, value).resolve()
        for value in (baseline_path, assisted_path, extraction_path)
    ]
    if len(set(paths)) != 3:
        raise ValueError("preparation/extraction paths must be distinct")
    anchors = precompute_anchors(bundle.request, binding=binding)
    aliases = make_aliases(bundle, anchors)
    common = dict(
        request=bundle.request,
        context=bundle.context,
        request_sha256=digest(canonical_json(bundle.request.model_dump())),
        context_sha256=digest(canonical_json(bundle.context.model_dump())),
        policy_sha256=policy_identity(),
        timezone_sha256=binding.tzif_sha256,
        anchors=anchors,
        aliases=aliases,
    )
    baseline_started = time.perf_counter()
    baseline = _save(
        root,
        baseline_path,
        PreparationPacket(
            **common,
            search=search(bundle, binding=binding, max_records=max_records),
            extraction_result=None,
            extraction_artifact_path=None,
            extraction_artifact_bytes=None,
            extraction_artifact_sha256=None,
        ),
    )
    if timings is not None:
        timings["baseline_search_and_save_seconds"] = (
            time.perf_counter() - baseline_started
        )
    assisted_started = time.perf_counter()
    try:
        path = artifact_path(root, extraction_path)
        actual_bytes = path.read_bytes()
        artifact = read_artifact(
            path, message_id=message_id, text=bundle.request.text, identity=identity
        )
        if path.read_bytes() != actual_bytes:
            raise ValueError("extraction artifact changed during preparation")
        captured = ExtractionArtifact.model_validate(
            loads_json(actual_bytes.decode("utf-8"))
        )
        if captured != artifact:
            raise ValueError("captured extraction bytes differ from validated artifact")
        if artifact.result.status != "ok":
            return PreparedInputs(baseline, None, artifact.result.failure)
        assisted = _save(
            root,
            assisted_path,
            PreparationPacket(
                **common,
                search=search(
                    bundle,
                    artifact.result.extraction,
                    binding=binding,
                    max_records=max_records,
                ),
                extraction_result=artifact.result,
                extraction_artifact_path=extraction_path,
                extraction_artifact_bytes=actual_bytes.decode("utf-8"),
                extraction_artifact_sha256=digest(actual_bytes),
            ),
        )
        return PreparedInputs(baseline, assisted, None)
    except Exception as error:
        return PreparedInputs(
            baseline,
            None,
            TechnicalFailure(
                stage="preprocessing",
                code="invalid_output",
                message=f"assisted preparation blocked: {type(error).__name__}: {error}",
            ),
        )
    finally:
        if timings is not None:
            timings["shared_assisted_read_search_and_save_seconds"] = (
                time.perf_counter() - assisted_started
            )


def verify_input(
    sealed: SealedInput,
    *,
    dataset: PublicDataset,
    message_id: str,
    root: Path,
    binding: TimezoneBinding,
) -> PreparationPacket:
    """Authenticate captured/saved bytes against current external inputs/policy.

    No search, anchor creation or extraction is rerun at execution time.
    """
    if type(sealed.data) is not bytes or digest(sealed.data) != sealed.sha256:
        raise ValueError("sealed preparation byte/hash mismatch")
    if artifact_path(root, sealed.path).read_bytes() != sealed.data:
        raise ValueError("saved preparation differs from sealed bytes")
    packet = PreparationPacket.model_validate(loads_json(sealed.data.decode("utf-8")))
    if canonical_json(packet.model_dump()) != sealed.data:
        raise ValueError("preparation bytes are not canonical")
    dataset = PublicDataset.model_validate(dataset.model_dump())
    bundle = dataset.bundle(message_id)
    request_zone(bundle.request, binding=binding)
    if (
        packet.request.model_dump() != bundle.request.model_dump()
        or packet.request_sha256 != digest(canonical_json(bundle.request.model_dump()))
    ):
        raise ValueError("stale preparation request binding")
    if (
        packet.context.model_dump() != bundle.context.model_dump()
        or packet.context_sha256 != digest(canonical_json(bundle.context.model_dump()))
    ):
        raise ValueError("stale preparation context binding")
    if (
        packet.policy_sha256 != policy_identity()
        or packet.timezone_sha256 != binding.tzif_sha256
    ):
        raise ValueError("stale preparation policy or timezone identity")
    if packet.aliases != make_aliases(bundle, packet.anchors):
        raise ValueError("invalid sealed alias map")
    bundle.validate_candidates(packet.search.candidates)
    records = {
        row.record_id: row.model_dump()
        for rows in (
            bundle.context.contacts,
            bundle.context.calendar,
            bundle.context.history,
        )
        for row in rows
    }
    for rows in (
        packet.search.evidence.contacts,
        packet.search.evidence.calendar,
        packet.search.evidence.history,
    ):
        for row in rows:
            if records.get(row.record_id) != row.model_dump():
                raise ValueError("sealed evidence differs from eligible source")
    if packet.extraction_result is None:
        if any(
            value is not None
            for value in (
                packet.extraction_artifact_path,
                packet.extraction_artifact_bytes,
                packet.extraction_artifact_sha256,
            )
        ):
            raise ValueError("baseline cannot carry extraction artifact")
    else:
        if (
            packet.extraction_result.status != "ok"
            or packet.extraction_artifact_path is None
            or packet.extraction_artifact_bytes is None
        ):
            raise ValueError(
                "assisted preparation requires successful saved extraction"
            )
        raw = packet.extraction_artifact_bytes.encode("utf-8")
        if (
            digest(raw) != packet.extraction_artifact_sha256
            or artifact_path(root, packet.extraction_artifact_path).read_bytes() != raw
        ):
            raise ValueError("saved extraction differs from shared sealed bytes")
        artifact = ExtractionArtifact.model_validate(loads_json(raw.decode("utf-8")))
        if (
            artifact.text != bundle.request.text
            or artifact.result.message_id != message_id
            or artifact.result != packet.extraction_result
        ):
            raise ValueError("sealed extraction request/result mismatch")
    return packet
