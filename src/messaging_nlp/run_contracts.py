"""Canonical Stage 4A/4B execution wrappers (not semantic study outcomes).

Only this module defines the persisted plan/protocol/receipt types. Stage 3 types
are embedded by reference to their exact retained bytes, never changed or shadowed.
"""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Literal, get_args, get_origin

from pydantic import (
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from messaging_nlp.contracts import (
    Condition,
    Digest,
    NonBlank,
    Request,
    StrictModel,
    TechnicalFailure,
)
from messaging_nlp.encoder_cache import portable_path
from messaging_nlp.limits import MAX_OUTPUT_TOKENS

PROTOCOL_VERSION = "nlp3-qwen-serial-v1"
OPENING_SHA256 = "3f7a172dd44af30695f9171555f13070d7408bf1e2ba88d4c2e8e6d17af4f67d"
QWEN_RECEIPT_SHA256 = "27140c9e319f33cc86c8b84f4599f0bb9cc6ad02e4aa6e08b08ebba3ca9459bf"
QWEN_REPO = "mlx-community/Qwen3.5-2B-OptiQ-4bit"
QWEN_REVISION = "0aeef18d47261ffbb35f423160cc7a6abdefc3b4"
RESET_SHA256 = "4620b44ee99596ab087e690be848bde07da883ba60ee3acd90aa64b0b9673a9f"
LEDGER_PATH = "runs/study-state/budget.json"
MUTEX_PATH = "runs/study-state/model.lock"
RunIdentifier = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
]


def valid_boot_identity(value: str) -> str:
    match = re.fullmatch(
        r"darwin-kern-boottime:([1-9][0-9]*):(0|[1-9][0-9]{0,5})", value
    )
    if match is None:
        raise ValueError("invalid canonical OS boot identity")
    return value


class RunContract(StrictModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )

    @model_validator(mode="before")
    @classmethod
    def strict_literal_types(cls, value):
        # Pydantic Literal[1]/[False] otherwise accepts True/0 despite strict=True.
        if isinstance(value, dict):
            for name, field in cls.model_fields.items():
                if name in value and get_origin(field.annotation) is Literal:
                    if not any(
                        type(value[name]) is type(option) and value[name] == option
                        for option in get_args(field.annotation)
                    ):
                        raise ValueError(
                            f"{name} requires its exact literal type/value"
                        )
        return value


class ArtifactRef(RunContract):
    path: NonBlank
    sha256: Digest
    bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def relative_path(cls, value):
        portable_path(value)
        return value


class FrozenFile(ArtifactRef):
    role: Literal[
        "source",
        "environment",
        "lock",
        "public",
        "policy",
        "roster",
        "preparation",
        "extraction",
        "prompt",
        "tokenizer",
        "other",
    ]


class DecodePolicy(RunContract):
    temperature: Literal[0] = 0
    seed: Literal[42] = 42
    enable_thinking: Literal[False] = False
    max_input_tokens: Literal[2048] = 2048
    max_output_tokens: Literal[256, 1024] = MAX_OUTPUT_TOKENS
    call_timeout_seconds: Literal[120] = 120
    cache_policy: Literal["fresh_normal_cache_per_row_no_kv_quantization"] = (
        "fresh_normal_cache_per_row_no_kv_quantization"
    )


class ResourceLimits(RunContract):
    max_tree_rss_bytes: Literal[3758096384] = 3758096384
    max_tree_phys_footprint_bytes: Literal[3758096384] = 3758096384
    max_swap_growth_bytes: Literal[268435456] = 268435456
    min_free_disk_bytes: Literal[10737418240, 16106127360] = 16106127360
    sample_interval_seconds: float = Field(default=0.25, gt=0, le=2)
    cleanup_seconds: int = Field(default=10, ge=5, le=30)
    memory_warning_seconds: Literal[10] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class SpendingPolicy(RunContract):
    account: Literal["stage3"] = "stage3"
    mode: Literal["bounded_attempt_to_complete"] = "bounded_attempt_to_complete"
    startup_seconds: int = Field(gt=0)
    overhead_seconds: int = Field(gt=0)
    max_run_seconds: int = Field(gt=0, le=21600)
    storage_bytes: int = Field(gt=0, le=5368709120)
    assistant_version: NonBlank
    assistant_version_refs: dict[NonBlank, NonBlank]
    extraction_version: NonBlank | None = None
    extraction_version_refs: dict[NonBlank, NonBlank] = {}


class ModelSpec(RunContract):
    repo: Literal["mlx-community/Qwen3.5-2B-OptiQ-4bit"] = QWEN_REPO
    revision: Literal["0aeef18d47261ffbb35f423160cc7a6abdefc3b4"] = QWEN_REVISION
    path: NonBlank
    receipt: ArtifactRef
    template_sha256: Digest
    runtime_versions: dict[NonBlank, NonBlank] = {
        "mlx": "0.32.4",
        "mlx-lm": "0.32.2",
        "transformers": "5.20.2",
        "pydantic": "2.14.1",
    }

    @field_validator("path")
    @classmethod
    def relative_path(cls, value):
        portable_path(value)
        return value

    @model_validator(mode="after")
    def pinned_runtime(self):
        if self.runtime_versions != {
            "mlx": "0.32.4",
            "mlx-lm": "0.32.2",
            "transformers": "5.20.2",
            "pydantic": "2.14.1",
        }:
            raise ValueError("Qwen runtime versions must retain pinned identities")
        return self


class TokenReceipt(RunContract):
    schema_version: Literal["nlp3-chat-tokens-v1"] = "nlp3-chat-tokens-v1"
    messages_sha256: Digest
    template_sha256: Digest
    model_receipt_sha256: Digest
    token_ids_sha256: Digest
    token_count: int = Field(gt=0, le=2048)
    token_ids: list[int] = Field(min_length=1, max_length=2048)
    tokenize: Literal[True] = True
    add_generation_prompt: Literal[True] = True
    enable_thinking: Literal[False] = False
    return_dict: Literal[False] = False
    truncation: Literal[False] = False
    padding: Literal[False] = False

    @model_validator(mode="after")
    def exact_retained_tokens(self):
        from messaging_nlp.io import canonical_json

        if any(token < 0 for token in self.token_ids):
            raise ValueError("token IDs must be nonnegative")
        if (
            self.token_count != len(self.token_ids)
            or self.token_ids_sha256
            != hashlib.sha256(canonical_json(self.token_ids)).hexdigest()
        ):
            raise ValueError("token receipt must retain exact count-bound token bytes")
        return self


class PlannedRow(RunContract):
    row_id: RunIdentifier
    ordinal: int = Field(ge=0)
    condition: Condition
    request: Request
    preparation: ArtifactRef | None
    preparation_failure: TechnicalFailure | None
    prompt: ArtifactRef | None
    token_receipt: ArtifactRef | None
    attempt: Literal[1] = 1

    @model_validator(mode="after")
    def prepared_or_blocked(self):
        if self.preparation_failure is None:
            if any(
                x is None for x in (self.preparation, self.prompt, self.token_receipt)
            ):
                raise ValueError("ready row requires preparation/prompt/token receipt")
        elif any(x is not None for x in (self.prompt, self.token_receipt)):
            raise ValueError("blocked row cannot have generation inputs")
        return self


class RunPlan(RunContract):
    schema_version: Literal["nlp3-run-plan-v1"] = "nlp3-run-plan-v1"
    project_root: NonBlank = "."
    run_id: RunIdentifier
    output_path: NonBlank
    ledger_path: Literal["runs/study-state/budget.json"] = LEDGER_PATH
    mutex_path: Literal["runs/study-state/model.lock"] = MUTEX_PATH
    readiness_path: NonBlank
    approval_path: NonBlank
    opening: ArtifactRef
    usage_reset: ArtifactRef
    public_root: NonBlank
    owner_id: NonBlank
    timezone_name: NonBlank
    timezone: ArtifactRef
    roster: ArtifactRef
    frozen_files: list[FrozenFile] = Field(min_length=1)
    model: ModelSpec
    main_environment: dict[NonBlank, NonBlank] = {
        "python": "3.11.15",
        "pydantic": "2.14.1",
        "messaging-nlp": "0.1.0",
    }
    decoding: DecodePolicy = DecodePolicy()
    resources: ResourceLimits = ResourceLimits()
    spending: SpendingPolicy
    rows: list[PlannedRow] = Field(min_length=1)
    revision_authority: ArtifactRef | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator(
        "output_path",
        "ledger_path",
        "mutex_path",
        "readiness_path",
        "approval_path",
        "public_root",
    )
    @classmethod
    def relative_path(cls, value):
        portable_path(value)
        return value

    @model_validator(mode="after")
    def frozen_order(self):
        if self.spending.overhead_seconds < self.resources.cleanup_seconds:
            raise ValueError("overhead allocation must include bounded cleanup")
        if (
            self.spending.startup_seconds + self.spending.overhead_seconds
            > self.spending.max_run_seconds
        ):
            raise ValueError("run deadline cannot exclude startup and cleanup")
        if [r.ordinal for r in self.rows] != list(range(len(self.rows))):
            raise ValueError("rows must retain contiguous frozen ordinal order")
        if len({r.row_id for r in self.rows}) != len(self.rows):
            raise ValueError("duplicate row id")
        if len({(r.request.message_id, r.condition) for r in self.rows}) != len(
            self.rows
        ):
            raise ValueError("duplicate request condition")
        if len({f.path for f in self.frozen_files}) != len(self.frozen_files):
            raise ValueError("duplicate frozen file")
        return self


class ReadinessReceipt(RunContract):
    schema_version: Literal["nlp3-readiness-v1"] = "nlp3-readiness-v1"
    plan_sha256: Digest
    boot_identity: NonBlank
    ledger_sha256: Digest
    usage_reset_sha256: Literal[
        "4620b44ee99596ab087e690be848bde07da883ba60ee3acd90aa64b0b9673a9f"
    ] = RESET_SHA256
    prepared: Literal[True] = True

    @field_validator("boot_identity")
    @classmethod
    def canonical_boot(cls, value):
        return valid_boot_identity(value)


class ExecutionApproval(RunContract):
    schema_version: Literal["nlp3-execution-approval-v1"] = "nlp3-execution-approval-v1"
    plan_sha256: Digest
    readiness_sha256: Digest
    approved: Literal[True] = True
    approved_by: NonBlank


class WorkerRequest(RunContract):
    protocol: Literal["nlp3-qwen-serial-v1"] = PROTOCOL_VERSION
    kind: Literal["load", "generate", "shutdown"]
    row_id: str | None = None
    payload: dict = {}


class WorkerEvent(RunContract):
    protocol: Literal["nlp3-qwen-serial-v1"] = PROTOCOL_VERSION
    kind: Literal[
        "ready", "generation_started", "fragment", "progress", "terminal", "error"
    ]
    row_id: str | None = None
    payload: dict = {}


class ResourceSample(RunContract):
    monotonic_seconds: float = Field(ge=0)
    pressure_level: int = Field(ge=0)
    swap_used_bytes: int = Field(ge=0)
    free_disk_bytes: int = Field(ge=0)
    tree_rss_bytes: int = Field(ge=0)
    tree_phys_footprint_bytes: int = Field(ge=0)
    coordinator_rss_bytes: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )
    coordinator_phys_footprint_bytes: int | None = Field(
        default=None, ge=0, exclude_if=lambda value: value is None
    )


class CleanupReceipt(RunContract):
    verified: bool
    process_group_id: int | None = None
    returncode: int | None = None
    sigterm_sent: bool = False
    sigkill_sent: bool = False
    detail: str = ""


class RowReceipt(RunContract):
    schema_version: Literal["nlp3-row-receipt-v1"] = "nlp3-row-receipt-v1"
    run_id: RunIdentifier
    plan_sha256: Digest
    row_id: RunIdentifier
    ordinal: int = Field(ge=0)
    condition: Condition
    message_id: NonBlank
    attempt: Literal[1] = 1
    disposition: Literal[
        "completed", "attempted_failure", "preparation_blocked", "unattempted_stop"
    ]
    provider_calls: int = Field(ge=0, le=1)
    dispatch_intent: bool
    dispatched: bool
    generation_started: bool
    generation_completed: bool
    finish_reason: str | None
    stop_reason: str | None
    reservation_id: str | None
    startup_reservation_id: str | None
    charged_seconds: int = Field(ge=0)
    charged_generations: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    generation_tokens: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    run_row: ArtifactRef
    resolved_envelope: ArtifactRef | None
    raw_output: ArtifactRef | None
    bindings: dict[str, ArtifactRef]
    logs: dict[str, ArtifactRef]
    cleanup: CleanupReceipt | None

    @model_validator(mode="after")
    def observed_states(self):
        if self.generation_completed and not self.generation_started:
            raise ValueError("completion requires model-start acknowledgement")
        if self.generation_started and not self.dispatched:
            raise ValueError("model-start requires dispatch")
        if self.dispatched and not self.dispatch_intent:
            raise ValueError("dispatch requires durable intent")
        if self.dispatched and self.provider_calls != 1:
            raise ValueError("dispatch requires one observed provider invocation")
        if self.generation_completed and self.finish_reason not in ("stop", "length"):
            raise ValueError("completion requires an observed runtime terminal reason")
        if self.disposition == "completed" and (
            not self.generation_completed or self.finish_reason != "stop"
        ):
            raise ValueError("technical completion requires a stopped generation")
        if self.disposition in ("preparation_blocked", "unattempted_stop") and (
            self.provider_calls or self.dispatched or self.generation_started
        ):
            raise ValueError("unexecuted work cannot claim calls")
        return self


class PendingRowRecord(RunContract):
    """Observed work whose immutable RowReceipt could not be committed.

    This wrapper preserves incomplete evidence without treating it as an executed
    success, an unexecuted row, or permission to regenerate anything.
    """

    row_id: RunIdentifier
    ordinal: int = Field(ge=0)
    condition: Condition
    message_id: NonBlank
    attempt: Literal[1] = 1
    provider_calls: int = Field(ge=0, le=1)
    dispatch_intent: bool
    dispatched: bool
    generation_started: bool
    generation_completed: bool
    finish_reason: str | None
    reservation_id: str | None
    raw_output: str | None
    run_row_bytes_utf8: str | None
    stop_reason: NonBlank
    persistence_errors: list[NonBlank]
    artifacts: dict[str, ArtifactRef]
    unavailable_artifacts: list[NonBlank]


def load_row_receipt(root, path):
    """Authenticate the actual shared wrapper and retained Stage 3 bytes for 4B.

    This is structural evidence validation, not task scoring or semantic review.
    All bound bytes must remain present; no reconstruction or model invocation.
    """
    from pathlib import Path

    from messaging_nlp.application_types import OutputEnvelope
    from messaging_nlp.contracts import RunRow
    from messaging_nlp.io import canonical_json, loads_json
    from messaging_nlp.run_storage import read_ref

    root = Path(root).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("receipt leaves retained run root")
    raw = path.read_bytes()
    receipt = RowReceipt.model_validate(loads_json(raw.decode("utf-8")))
    if canonical_json(receipt.model_dump()) != raw:
        raise ValueError("receipt bytes must be canonical")
    plan_ref = receipt.bindings.get("plan")
    if plan_ref is None or plan_ref.sha256 != receipt.plan_sha256:
        raise ValueError("receipt requires exact retained plan binding")
    plan_raw = read_ref(root, plan_ref)
    plan = RunPlan.model_validate(loads_json(plan_raw.decode("utf-8")))
    if canonical_json(plan.model_dump()) != plan_raw or receipt.ordinal >= len(
        plan.rows
    ):
        raise ValueError("invalid retained plan or receipt ordinal")
    planned = plan.rows[receipt.ordinal]
    if (plan.run_id, planned.row_id, planned.condition, planned.request.message_id) != (
        receipt.run_id,
        receipt.row_id,
        receipt.condition,
        receipt.message_id,
    ):
        raise ValueError("receipt does not match its frozen planned row")
    row_raw = read_ref(root, receipt.run_row)
    row = RunRow.model_validate(loads_json(row_raw.decode("utf-8")))
    if canonical_json(row.model_dump()) != row_raw:
        raise ValueError("RunRow bytes must be canonical")
    if (receipt.run_id, receipt.condition, receipt.message_id, receipt.attempt) != (
        row.run_id,
        row.condition,
        row.request.message_id,
        row.attempt,
    ):
        raise ValueError("execution wrapper and RunRow identity disagree")
    if row.request != planned.request:
        raise ValueError("retained original request differs from frozen planned row")
    expected = {
        "completed": "ok",
        "attempted_failure": "failed",
        "preparation_blocked": "unexecuted",
        "unattempted_stop": "unexecuted",
    }
    if row.status != expected[receipt.disposition]:
        raise ValueError("execution wrapper and RunRow status disagree")
    if (row.raw_output is None) != (receipt.raw_output is None):
        raise ValueError("raw output presence mismatch")
    if receipt.raw_output is not None:
        if read_ref(root, receipt.raw_output) != row.raw_output.encode("utf-8"):
            raise ValueError("raw output differs from retained RunRow")
    envelope = None
    if receipt.resolved_envelope is not None:
        encoded = read_ref(root, receipt.resolved_envelope)
        envelope = OutputEnvelope.model_validate(loads_json(encoded.decode("utf-8")))
        if canonical_json(envelope.model_dump()) != encoded:
            raise ValueError("resolved envelope bytes must be canonical")
    if row.status == "ok" and (envelope is None or envelope.decision != row.decision):
        raise ValueError(
            "successful RunRow requires matching complete resolved envelope"
        )
    for reference in (*receipt.bindings.values(), *receipt.logs.values()):
        read_ref(root, reference)
    return receipt, row, envelope
