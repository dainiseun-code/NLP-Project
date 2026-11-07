"""Single-turn injected generation. One call, strict output, no repair or effects."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from messaging_nlp.application_types import (
    ApplicationResult,
    GenerationOutput,
    OutputEnvelope,
    PreparationPacket,
    PreparedInputs,
    Prompt,
)
from messaging_nlp.context import PublicDataset
from messaging_nlp.contracts import (
    Condition,
    Decision,
    ModelDecision,
    RunRow,
    SelectedFact,
    Selection,
    TechnicalFailure,
)
from messaging_nlp.drafting import validate_envelope
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.limits import MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS
from messaging_nlp.preparation import digest, verify_input
from messaging_nlp.prompts import build_prompt
from messaging_nlp.role_hints import single_addressed_contact
from messaging_nlp.temporal import TimezoneBinding


class GenerationProvider(Protocol):
    def generate(
        self, prompt: Prompt, *, max_input_tokens: int, max_output_tokens: int
    ) -> GenerationOutput:
        """Return raw (including partial) output; never send or modify messages."""
        ...


def _resolve_alias(packet: PreparationPacket, value: str, kind: str) -> str:
    target = packet.aliases.get(value)
    if target is None or target.kind != kind:
        raise ValueError(f"unknown or wrong-kind {kind} alias: {value!r}")
    return target.internal_id


def _derive_decision(
    model: ModelDecision,
    *,
    recipients: list[str],
    event_id: str | None,
    packet: PreparationPacket,
) -> Decision:
    """Bind a flat model answer to eligible contact/event records.

    The model still chooses the recipient and event; Python only adds the
    record citations and literal facts needed by the existing saved/evaluated
    contract. It never chooses or rewrites either selection.
    """
    if model.action != "draft":
        return Decision(
            action=model.action,
            selection=None,
            draft=None,
            clarification=model.clarification,
            reason=model.reason,
        )
    contacts = {row.contact_id: row for row in packet.context.contacts}
    events = {row.event_id: row for row in packet.context.calendar}
    facts: list[SelectedFact] = []
    sources: set[str] = set()
    for contact_id in recipients:
        contact = contacts.get(contact_id)
        if contact is None:
            raise ValueError(
                f"selected recipient has no eligible contact record: {contact_id}"
            )
        facts.append(
            SelectedFact(
                field="name",
                value=contact.name,
                source_record_ids=[contact.record_id],
            )
        )
        sources.add(contact.record_id)
    if event_id is not None:
        event = events.get(event_id)
        if event is None:
            raise ValueError(f"selected event has no eligible record: {event_id}")
        sources.add(event.record_id)
        for field in ("title", "start", "end", "location"):
            value = getattr(event, field)
            if value is not None:
                facts.append(
                    SelectedFact(
                        field=field,
                        value=value,
                        source_record_ids=[event.record_id],
                    )
                )
    selection = Selection(
        recipient_contact_ids=recipients,
        selected_event_id=event_id,
        facts=facts,
        source_record_ids=sorted(sources),
    )
    return Decision(
        action="draft",
        selection=selection,
        draft=model.draft,
        clarification=None,
        reason=None,
    )


def _resolve_legacy(packet: PreparationPacket, payload: dict) -> OutputEnvelope:
    """Read the pre-flat envelope retained by older evidence and tests."""
    aliased = OutputEnvelope.model_validate(payload)
    data = aliased.model_dump()

    def resolve(value: str, kind: str) -> str:
        return _resolve_alias(packet, value, kind)

    selection = data["decision"]["selection"]
    if selection is not None:
        selection["recipient_contact_ids"] = [
            resolve(value, "contact") for value in selection["recipient_contact_ids"]
        ]
        if selection["selected_event_id"] is not None:
            selection["selected_event_id"] = resolve(
                selection["selected_event_id"], "event"
            )
        selection["source_record_ids"] = [
            resolve(value, "record") for value in selection["source_record_ids"]
        ]
        for fact in selection["facts"]:
            fact["source_record_ids"] = [
                resolve(value, "record") for value in fact["source_record_ids"]
            ]
    for mention in data["mentions"]:
        if mention["anchor_id"] is not None:
            mention["anchor_id"] = resolve(mention["anchor_id"], "anchor")
    return OutputEnvelope.model_validate(data)


def resolve_envelope(
    raw: str, packet: PreparationPacket, *, bind_addressed: bool = True
) -> OutputEnvelope:
    """Strict JSON/schema then typed opaque-alias resolution; no guessing.

    The flat object is the model-facing contract. A legacy ``decision`` plus
    ``mentions`` object is still read so retained evidence and older tests
    remain reproducible; that path is not requested by the current prompt.

    When exactly one contact is syntactically addressed and no identity is
    unresolved, the deterministic addressee binding replaces the model's
    recipient ID. The model still writes the action and draft; this binding is a
    syntax policy, not a semantic selector. Set ``bind_addressed=False`` for a
    model-only comparison.
    """
    payload = loads_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    if "mentions" in payload:
        return _resolve_legacy(packet, payload)
    model = ModelDecision.model_validate(payload)
    recipients = (
        [
            _resolve_alias(packet, value, "contact")
            for value in model.recipient_contact_ids
        ]
        if model.action == "draft"
        else []
    )
    if model.action == "draft" and bind_addressed:
        contacts = [
            {
                "contact_id": row.contact_id,
                "name": row.name,
                "aliases": list(row.aliases),
            }
            for row in packet.search.evidence.contacts
        ]
        addressed = single_addressed_contact(packet.request.text, contacts)
        if addressed is not None and addressed in packet.search.candidates.contact_ids:
            recipients = [addressed]
    event_id = (
        _resolve_alias(packet, model.selected_event_id, "event")
        if model.action == "draft" and model.selected_event_id is not None
        else None
    )
    return OutputEnvelope(
        decision=_derive_decision(
            model, recipients=recipients, event_id=event_id, packet=packet
        ),
        mentions=[],
    )


def execute_condition(
    dataset: PublicDataset,
    message_id: str,
    condition: Condition,
    prepared: PreparedInputs,
    provider: GenerationProvider,
    *,
    root: Path,
    run_id: str,
    binding: TimezoneBinding,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> ApplicationResult:
    """Execute just one condition, not a study coordinator or durable runner.

    ``ok`` means bounded technical validity, never semantic task success. Planned
    ``attempt=1`` is retained independently from the observed provider call count.
    """
    if condition not in ("baseline", "B", "A+B"):
        raise ValueError("unknown condition")
    if type(max_output_tokens) is not int or max_output_tokens not in (
        256,
        MAX_OUTPUT_TOKENS,
    ):
        raise ValueError("unsupported generation ceiling")
    dataset = PublicDataset.model_validate(dataset.model_dump())
    bundle = dataset.bundle(message_id)
    request_hash = digest(canonical_json(bundle.request.model_dump()))
    context_hash = digest(canonical_json(bundle.context.model_dump()))
    sealed = prepared.baseline if condition == "baseline" else prepared.assisted
    packet = None
    prompt = None
    envelope = None
    raw = None
    calls = 0
    failure = None
    status = "unexecuted"
    try:
        if sealed is None:
            failure = prepared.assisted_failure or TechnicalFailure(
                stage="preprocessing",
                code="missing",
                message="shared assisted preparation is missing",
            )
        else:
            verified = verify_input(
                sealed,
                dataset=dataset,
                message_id=message_id,
                root=root,
                binding=binding,
            )
            if (condition == "baseline") != (verified.extraction_result is None):
                raise ValueError("condition does not match saved preparation")
            packet = verified
            prompt = build_prompt(packet, condition)
    except Exception as error:
        failure = TechnicalFailure(
            stage="preprocessing",
            code="invalid_output",
            message=f"preparation rejected: {type(error).__name__}: {error}",
        )
    if failure is None:
        try:
            generate = provider.generate
            if not callable(generate):
                raise TypeError("provider generate must be callable")
            calls = 1
            status = "failed"
            generated = generate(
                prompt,
                max_input_tokens=MAX_INPUT_TOKENS,
                max_output_tokens=max_output_tokens,
            )
            if not isinstance(generated, GenerationOutput):
                raise TypeError("provider must return GenerationOutput")
            if not isinstance(generated.raw_output, str):
                raise TypeError("provider raw output must be text")
            raw = generated.raw_output
            if generated.failure is not None:
                if generated.failure.stage != "generation":
                    raise ValueError("provider failure must have generation stage")
                failure = generated.failure
        except Exception as error:
            partial = getattr(error, "raw_output", None)
            if calls and isinstance(partial, str):
                raw = partial
            failure = TechnicalFailure(
                stage="generation",
                code="runtime_error",
                message=f"provider failed: {type(error).__name__}: {error}",
            )
        if failure is None:
            try:
                envelope = resolve_envelope(raw, packet)
                validate_envelope(
                    envelope,
                    bundle=bundle,
                    candidates=packet.search.candidates,
                    anchors=packet.anchors,
                    binding=binding,
                )
                status = "ok"
            except Exception as error:
                failure = TechnicalFailure(
                    stage="validation",
                    code="invalid_output",
                    message=f"output rejected: {type(error).__name__}: {error}",
                )
    row = RunRow(
        schema_version=1,
        run_id=run_id,
        condition=condition,
        request=bundle.request,
        request_sha256=request_hash,
        context_sha256=context_hash,
        # A rejected typed payload has no authentic immutable-byte digest. Do not
        # re-raise while saving the failure, coerce it, or substitute its claimed
        # hash. Actual bytes keep their observed digest even when stale/invalid.
        preprocessing_sha256=(
            digest(sealed.data)
            if sealed is not None and type(sealed.data) is bytes
            else None
        ),
        extraction_result=packet.extraction_result if packet is not None else None,
        candidates=packet.search.candidates if packet is not None else None,
        status=status,
        decision=envelope.decision if status == "ok" else None,
        failure=failure,
        raw_output=raw,
        attempt=1,
    )
    return ApplicationResult(row, calls, prompt, envelope)
