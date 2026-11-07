"""Small shared Stage 3 contracts; no model execution or semantic-success field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, field_validator, model_validator

from messaging_nlp.context import PublicContext
from messaging_nlp.contracts import (
    CandidateEvidence,
    Decision,
    Digest,
    ExtractionResult,
    NonBlank,
    Request,
    RunRow,
    StrictModel,
    TechnicalFailure,
)


class SearchQuery(StrictModel):
    channel: Literal["original", "person", "associated", "event", "topic", "date"]
    text: str
    contact_ids: list[str] = []


class Coverage(StrictModel):
    eligible_records: int
    matched_records: int
    packed_records: int
    partial: bool
    notes: list[str]


class SearchResult(StrictModel):
    candidates: CandidateEvidence
    evidence: PublicContext
    queries: list[SearchQuery]
    coverage: Coverage


class RequestAnchor(StrictModel):
    anchor_id: NonBlank
    kind: Literal["proposal_time", "quotation"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: NonBlank
    value: NonBlank
    scope_start: int = Field(ge=0)
    scope_end: int = Field(gt=0)


class AliasTarget(StrictModel):
    kind: Literal["owner", "contact", "event", "record", "anchor"]
    internal_id: NonBlank


class PreparationPacket(StrictModel):
    schema_version: Literal[1] = 1
    request: Request
    context: PublicContext
    request_sha256: Digest
    context_sha256: Digest
    policy_sha256: Digest
    timezone_sha256: Digest
    search: SearchResult
    anchors: list[RequestAnchor]
    aliases: dict[str, AliasTarget]
    extraction_result: ExtractionResult | None
    extraction_artifact_path: str | None
    extraction_artifact_bytes: str | None
    extraction_artifact_sha256: Digest | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_version(cls, value):
        if type(value) is not int:
            raise ValueError("preparation version requires a strict integer")
        return value


class Mention(StrictModel):
    surface: NonBlank
    rendering_kind: Literal[
        "literal",
        "excerpt",
        "clock",
        "date",
        "datetime",
        "faithful_quote",
        "proposal_clock",
    ]
    fact_index: int | None = Field(default=None, ge=0)
    anchor_id: NonBlank | None = None

    @model_validator(mode="after")
    def exactly_one_reference(self):
        if (self.fact_index is None) == (self.anchor_id is None):
            raise ValueError("mention requires exactly one fact or request anchor")
        return self


class OutputEnvelope(StrictModel):
    decision: Decision
    mentions: list[Mention]

    @model_validator(mode="after")
    def nondraft_mentions(self):
        if self.decision.action != "draft" and self.mentions:
            raise ValueError("non-draft mentions must be empty")
        return self


@dataclass(frozen=True)
class SealedInput:
    """The immutable bytes of one actual saved preparation, not mutable models."""

    path: str
    data: bytes
    sha256: str


@dataclass(frozen=True)
class PreparedInputs:
    baseline: SealedInput
    assisted: SealedInput | None
    assisted_failure: TechnicalFailure | None


@dataclass(frozen=True)
class Prompt:
    system: str
    common: bytes
    entities: bytes | None

    @property
    def user(self) -> bytes:
        return self.common + (
            b"\nENTITY ANNOTATIONS\n" + self.entities
            if self.entities is not None
            else b""
        )


@dataclass(frozen=True)
class GenerationOutput:
    raw_output: str
    failure: TechnicalFailure | None = None


@dataclass(frozen=True)
class ApplicationResult:
    row: RunRow
    provider_calls: int
    prompt: Prompt | None
    envelope: OutputEnvelope | None
    validation_scope: str = "bounded deterministic checks; prose semantics unmeasured"
