"""Strict single-turn contracts. Valid references do not prove draft semantics."""

from datetime import datetime, timezone
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Digest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Condition = Literal["baseline", "B", "A+B"]


def clock(value: str) -> datetime:
    """Parse an absolute instant, preserving the original supplied text."""
    at = datetime.fromisoformat(value)
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("timestamp requires an explicit UTC offset")
    return at.astimezone(timezone.utc)


def unique(values: list, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Request(StrictModel):
    message_id: NonBlank
    text: NonBlank
    owner_id: NonBlank
    request_time: str
    time_zone: NonBlank
    context_snapshot_id: NonBlank

    @model_validator(mode="after")
    def valid_clock(self):
        clock(self.request_time)
        try:
            ZoneInfo(self.time_zone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("unknown IANA time zone") from error
        # Source offsets sometimes differ from local civil offsets: the instant
        # is authoritative, ZoneInfo supplies the local calendar interpretation.
        return self


class Entity(StrictModel):
    type: Literal["PERSON", "DATE", "EVENT", "TOPIC"]
    text: NonBlank
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered_bounds(self):
        if self.end <= self.start:
            raise ValueError("end must exceed start")
        return self


class Extraction(StrictModel):
    entities: list[Entity]

    def validate_text(self, text: str):
        previous_end = -1
        for entity in self.entities:
            if entity.start < previous_end:
                raise ValueError("entities overlap, repeat, or are out of order")
            if entity.end > len(text) or text[entity.start : entity.end] != entity.text:
                raise ValueError("entity slice differs from exact input text")
            previous_end = entity.end
        return self


class TechnicalFailure(StrictModel):
    stage: Literal["extraction", "preprocessing", "generation", "validation", "runner"]
    code: Literal[
        "timeout",
        "invalid_output",
        "missing",
        "runtime_error",
        "truncated",
        "launch_error",
        "sensor_error",
        "interrupted",
        "resource_stop",
    ]
    message: NonBlank


class ExtractionResult(StrictModel):
    """One extraction outcome; identity binds the extractor's complete policy.

    ``input_sha256`` hashes the original request text's UTF-8 bytes. The later
    encoder adapter must define the checkpoint/tokenizer/offset/settings receipt
    represented by ``extractor_sha256``; this schema is not a cache validator.
    """

    message_id: NonBlank
    status: Literal["ok", "failed"]
    extraction: Extraction | None
    failure: TechnicalFailure | None
    raw_output: str | None
    input_sha256: Digest
    extractor_sha256: Digest

    @model_validator(mode="after")
    def coherent_result(self):
        if self.status == "ok":
            if self.extraction is None or self.failure is not None:
                raise ValueError("success requires extraction and no failure")
        elif self.extraction is not None or self.failure is None:
            raise ValueError("failure requires failure detail and no extraction")
        if self.failure is not None and self.failure.stage != "extraction":
            raise ValueError("extraction failure has wrong stage")
        return self


class CandidateEvidence(StrictModel):
    """Candidate IDs and their public supporting record IDs; no gold/state."""

    contact_ids: list[NonBlank]
    event_ids: list[NonBlank]
    source_record_ids: list[NonBlank]

    @model_validator(mode="after")
    def unique_ids(self):
        for field in ("contact_ids", "event_ids", "source_record_ids"):
            unique(getattr(self, field), field)
        return self


class SelectedFact(StrictModel):
    field: Literal["name", "title", "start", "end", "location", "message_text"]
    value: NonBlank
    source_record_ids: list[NonBlank] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_fact(self):
        unique(self.source_record_ids, "fact sources")
        if self.field in ("start", "end"):
            clock(self.value)
        return self


class Selection(StrictModel):
    """Source-bound literals, not semantic roles or draft verification.

    Names and message text may have multiple source-distinct entries (for
    example, a recipient plus a third party or a two-message evidence chain).
    Facts are duplicates only when field, literal value and unordered source
    IDs are identical. Each field of the one selected event is single-valued.
    """

    recipient_contact_ids: list[NonBlank]
    selected_event_id: NonBlank | None
    facts: list[SelectedFact]
    source_record_ids: list[NonBlank]

    @model_validator(mode="after")
    def unique_ids(self):
        unique(self.recipient_contact_ids, "recipients")
        unique(self.source_record_ids, "selection sources")
        unique(
            [
                (fact.field, fact.value, tuple(sorted(fact.source_record_ids)))
                for fact in self.facts
            ],
            "selected facts",
        )
        unique(
            [
                fact.field
                for fact in self.facts
                if fact.field in ("title", "start", "end", "location")
            ],
            "selected event fact fields",
        )
        if any(
            not set(fact.source_record_ids) <= set(self.source_record_ids)
            for fact in self.facts
        ):
            raise ValueError("fact sources must be selection sources")
        return self


class ModelDecision(StrictModel):
    """The flat model-facing decision. Provenance is bound by Python later.

    This is deliberately smaller than the resolved ``Decision``: the model names
    an action, supported recipients/event and draft text, while the application
    derives record citations and literal facts from eligible candidate evidence.
    It must not be used as the saved final shape; ``Decision`` remains that.
    """

    action: Literal["draft", "clarify", "unsupported"]
    recipient_contact_ids: list[NonBlank] | None
    selected_event_id: NonBlank | None
    draft: NonBlank | None
    clarification: NonBlank | None
    reason: NonBlank | None

    @model_validator(mode="after")
    def coherent_action(self):
        if self.recipient_contact_ids is not None:
            unique(self.recipient_contact_ids, "recipients")
        if self.action == "draft":
            if not self.recipient_contact_ids or self.draft is None:
                raise ValueError("draft requires recipients and text")
            if self.clarification is not None or self.reason is not None:
                raise ValueError("draft cannot also clarify or fail")
        else:
            if self.draft is not None:
                raise ValueError("non-draft cannot retain draft text")
            if self.action == "clarify":
                if self.clarification is None:
                    raise ValueError("clarify requires a question")
                if self.reason is not None and not self.reason.strip():
                    raise ValueError("clarify reason must be null or nonblank")
            elif self.clarification is not None or self.reason is None:
                raise ValueError("unsupported requires only a reason")
        return self


class Decision(StrictModel):
    action: Literal["draft", "clarify", "unsupported"]
    selection: Selection | None
    draft: NonBlank | None
    clarification: NonBlank | None
    reason: NonBlank | None

    @model_validator(mode="after")
    def coherent_action(self):
        if self.action == "draft":
            if (
                self.selection is None
                or not self.selection.recipient_contact_ids
                or not self.selection.source_record_ids
                or self.draft is None
            ):
                raise ValueError("draft requires supported selection and text")
            if self.clarification is not None or self.reason is not None:
                raise ValueError("draft cannot also clarify or fail")
        else:
            if self.selection is not None or self.draft is not None:
                raise ValueError("non-draft cannot retain selection or draft")
            if self.action == "clarify":
                if self.clarification is None:
                    raise ValueError("clarification requires a question")
            elif self.clarification is not None or self.reason is None:
                raise ValueError("unsupported requires only reason")
        return self


class RunRow(StrictModel):
    """Saved outcome, including failures/unexecuted work; no retry semantics.

    ``request_sha256`` hashes canonical JSON of the six public request fields.
    ``attempt`` identifies the sole planned attempt, not an observed call count:
    an unexecuted row records no model output and is not a measured prediction.
    This schema does not implement durable execution or semantic review.
    """

    schema_version: Literal[1]
    run_id: NonBlank
    condition: Condition
    request: Request
    request_sha256: Digest
    context_sha256: Digest
    preprocessing_sha256: Digest | None
    extraction_result: ExtractionResult | None
    candidates: CandidateEvidence | None
    status: Literal["ok", "failed", "unexecuted"]
    decision: Decision | None
    failure: TechnicalFailure | None
    raw_output: str | None
    attempt: Literal[1]

    @field_validator("schema_version", "attempt", mode="before")
    @classmethod
    def strict_single_attempt(cls, value):
        # Pydantic's Literal[1] alone also accepts True and 1.0, even in
        # strict mode. Persisted schema/attempt counters must be integers.
        if type(value) is not int:
            raise ValueError("schema version and attempt require strict integers")
        return value

    @model_validator(mode="after")
    def coherent_row(self):
        import hashlib

        from messaging_nlp.io import canonical_json

        if (
            hashlib.sha256(canonical_json(self.request.model_dump())).hexdigest()
            != self.request_sha256
        ):
            raise ValueError("request hash mismatch")
        if self.status == "ok":
            if (
                self.decision is None
                or self.failure is not None
                or self.raw_output is None
            ):
                raise ValueError("success requires decision/raw output, no failure")
            if self.candidates is None or self.preprocessing_sha256 is None:
                raise ValueError("success requires preprocessing and candidates")
        elif self.decision is not None or self.failure is None:
            raise ValueError("failed/unexecuted rows require failure, no decision")
        if self.status == "unexecuted" and self.raw_output is not None:
            raise ValueError("unexecuted row cannot have model output")
        if self.condition == "baseline" and self.extraction_result is not None:
            raise ValueError("baseline cannot have extraction annotations")
        if self.extraction_result is not None:
            result = self.extraction_result
            if result.message_id != self.request.message_id:
                raise ValueError("extraction request mismatch")
            text_hash = hashlib.sha256(self.request.text.encode("utf-8")).hexdigest()
            if result.input_sha256 != text_hash:
                raise ValueError("extraction exact-text hash mismatch")
            if result.extraction is not None:
                result.extraction.validate_text(self.request.text)
        if self.condition != "baseline" and self.status == "ok":
            if self.extraction_result is None or self.extraction_result.status != "ok":
                raise ValueError("B/A+B success requires successful shared extraction")
        return self
