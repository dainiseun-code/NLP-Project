"""Portable, fail-closed Stage 4A budget ledger.

The ledger imports the frozen Stage 1 opening (hash ``3f7a...f67d``) as an
immutable anchor, preserves its 416 historical reservation records verbatim and
applies the single user-authorized usage-reset epoch
(``development-usage-reset-20261027T222217Z``) exactly once. Generations and
seconds usage start at zero in that epoch; prior storage usage and the
prompt/extraction version registry remain charged. A ledger created without the
reset authority keeps the historical ``opening_import`` epoch and can be
advanced to the reset epoch once, retaining earlier new charges as history.

Charging is conservative and never refunds work silently:

* ``finished`` settles to the observed actual cost, even when the output was
  invalid (the ledger only accounts execution, not task success);
* ``failed`` keeps ``max(planned, actual)`` independently per unit;
* ``reserved`` (a crash or a live entry) keeps the planned maximum and blocks
  unrelated new work until the owning coordinator settles it;
* ``cancelled`` is zero only with explicit ``never_launched`` proof.

Every constructor step and mutation takes a portable sibling ``fcntl`` lock,
re-verifies the immutable opening/reset bytes and the anchored caps/records,
then atomically replaces an fsynced JSON document. A changed opening, reset
document, caps, imported record or prompt/extraction registry fails closed. Only
the single Stage 3 development account is charged: no Stage 4 allowance is
borrowed and no training/profile kind exists here.

This module deliberately depends on the standard library only so a future
runtime coordinator can use it without optional model packages.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from messaging_nlp.run_revision import (
    EPOCH_REVISION,
    load_authority,
    validate_row_scope,
)

SCHEMA_VERSION = "nlp3-portable-budget-ledger-v1"
OPENING_SCHEMA_VERSION = "nlp3-budget-opening-v1"
RESET_SCHEMA_VERSION = "nlp3-authorized-usage-reset-v1"

OPENING_SHA256 = "3f7a172dd44af30695f9171555f13070d7408bf1e2ba88d4c2e8e6d17af4f67d"
RESET_AUTHORITY_SHA256 = (
    "4620b44ee99596ab087e690be848bde07da883ba60ee3acd90aa64b0b9673a9f"
)
RESET_ID = "development-usage-reset-20261027T222217Z"

IMPORTED_RESERVATION_COUNT = 416
IMPORTED_GENERATIONS = 2054
IMPORTED_SECONDS = 11075
IMPORTED_STORAGE_BYTES = 78719238

STAGE3_MAX_SECONDS = 21600
STAGE3_MAX_GENERATIONS = 3000
GLOBAL_MAX_SECONDS = 108000
MAX_STORAGE_BYTES = 5368709120
MAX_ASSISTANT_PROMPT_VERSIONS = 4
MAX_EXTRACTION_VERSIONS = 2

EPOCH_OPENING = "opening_import"
EPOCH_RESET = "authorized_reset"

VERSION_KINDS = ("assistant_prompt", "extraction")
VERSION_LIMITS = {
    "assistant_prompt": "max_assistant_prompt_versions",
    "extraction": "max_extraction_versions",
}
ALLOWED_KINDS = (
    "startup",
    "overhead",
    "row",
    "preparation",
    "generation",
    "storage",
)
SECONDS_ONLY_KINDS = ("startup", "overhead", "preparation")
GENERATION_KINDS = ("row", "generation")
REFUSED_KINDS = {
    "training": "training jobs are not part of the Stage 4A inference ledger",
    "training_job": "training jobs are not part of the Stage 4A inference ledger",
    "profile": "profiling reservations are not authorized for Stage 4A",
}
RESERVATION_STATUSES = ("reserved", "finished", "failed", "cancelled")
SETTLE_STATUSES = ("finished", "failed", "cancelled")
_UNITS = ("seconds", "generations", "storage_bytes")

_LEDGER_KEYS = {
    "schema_version",
    "revision",
    "opening",
    "imported_records",
    "epochs",
    "current_epoch",
    "prompt_versions",
    "reservations",
    "integrity_sha256",
}
_ANCHOR_KEYS = {
    "path",
    "sha256",
    "bytes",
    "caps",
    "reservation_count",
    "derived_usage",
    "registered_versions",
    "unfinished_reservation_ids",
}
_EPOCH_KEYS = {
    "index",
    "mode",
    "reset_id",
    "reset_sha256",
    "reset_path",
    "source_opening_sha256",
    "base_usage",
    "started_at",
}
_IMPORTED_KEYS = {
    "reservation_id",
    "stage",
    "kind",
    "status",
    "planned_seconds",
    "actual_seconds",
    "planned_generations",
    "actual_generations",
    "planned_bytes",
    "actual_bytes",
    "created_at",
    "finished_at",
    "cancel_reason",
    "over_planned",
}
_OWN_KEYS = {
    "reservation_id",
    "kind",
    "status",
    "planned_seconds",
    "planned_generations",
    "planned_bytes",
    "actual_seconds",
    "actual_generations",
    "actual_bytes",
    "created_at",
    "finished_at",
    "cancel_reason",
    "never_launched",
    "over_planned",
    "refs",
    "epoch",
}
_VERSION_KEYS = {"version_id", "registered_at", "refs"}


class BudgetLedgerError(ValueError):
    """A budget request or ledger state is refused; writes stay fail-closed."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonblank(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _zero_usage() -> dict:
    return {"seconds": 0, "generations": 0, "storage_bytes": 0}


def _checked_usage(value, what: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(_UNITS):
        raise BudgetLedgerError(f"{what} usage must record seconds/generations/storage")
    for unit in _UNITS:
        if not _is_int(value[unit]) or value[unit] < 0:
            raise BudgetLedgerError(f"{what} {unit} must be a nonnegative integer")
    return {unit: value[unit] for unit in _UNITS}


def _checked_charge(value, name: str) -> int:
    if not _is_int(value) or value < 0:
        raise BudgetLedgerError(f"{name} must be a nonnegative integer")
    return value


def _checked_refs(refs) -> dict:
    if refs is None:
        return {}
    if not isinstance(refs, Mapping):
        raise BudgetLedgerError("refs must be a mapping of nonblank strings")
    checked = {}
    for key, value in refs.items():
        if not _nonblank(key) or not _nonblank(value):
            raise BudgetLedgerError("refs keys and values must be nonblank strings")
        if not isinstance(value, str):
            raise BudgetLedgerError("refs values must be strings")
        checked[key] = value
    if len(checked) > 16:
        raise BudgetLedgerError("refs may hold at most 16 entries")
    return checked


def _charged(record: dict) -> tuple[int, int, int]:
    """Conservative charge: cancelled=0, finished=actual, else max per unit."""
    if record["status"] == "cancelled":
        return (0, 0, 0)
    if record["status"] == "finished":
        return (
            record["actual_seconds"],
            record["actual_generations"],
            record["actual_bytes"],
        )
    return (
        max(record["planned_seconds"], record["actual_seconds"]),
        max(record["planned_generations"], record["actual_generations"]),
        max(record["planned_bytes"], record["actual_bytes"]),
    )


def _sum_usage(records) -> dict:
    usage = _zero_usage()
    for record in records:
        seconds, generations, storage = _charged(record)
        usage["seconds"] += seconds
        usage["generations"] += generations
        usage["storage_bytes"] += storage
    return usage


def _validate_imported_record(record) -> dict:
    if not isinstance(record, dict) or set(record) != _IMPORTED_KEYS:
        raise BudgetLedgerError("imported opening record fields changed")
    if not _nonblank(record["reservation_id"]):
        raise BudgetLedgerError("imported opening record has no reservation id")
    if record["stage"] != "stage3":
        raise BudgetLedgerError("the portable ledger charges only the stage3 account")
    if record["kind"] not in ("generation", "storage"):
        raise BudgetLedgerError("the opening may not import training/profile records")
    if record["status"] not in RESERVATION_STATUSES + ("crashed",):
        raise BudgetLedgerError("imported opening record has an unknown status")
    for field in (
        "planned_seconds",
        "actual_seconds",
        "planned_generations",
        "actual_generations",
        "planned_bytes",
        "actual_bytes",
    ):
        if not _is_int(record[field]) or record[field] < 0:
            raise BudgetLedgerError(f"imported {field} must be a nonnegative integer")
    if not isinstance(record["over_planned"], bool):
        raise BudgetLedgerError("imported over_planned must be a boolean")
    for field in ("created_at", "finished_at", "cancel_reason"):
        if not isinstance(record[field], str):
            raise BudgetLedgerError(f"imported {field} must be a string")
    return dict(record)


def _validate_version_record(record) -> dict:
    if not isinstance(record, dict) or set(record) != _VERSION_KEYS:
        raise BudgetLedgerError("version record fields changed")
    if not _nonblank(record["version_id"]):
        raise BudgetLedgerError("version record needs a version id")
    if not _nonblank(record["registered_at"]):
        raise BudgetLedgerError("version record needs a registration time")
    refs = _checked_refs(record["refs"])
    return {
        "version_id": record["version_id"],
        "registered_at": record["registered_at"],
        "refs": refs,
    }


def _validate_caps(caps) -> dict:
    if not isinstance(caps, dict):
        raise BudgetLedgerError("opening caps must be a mapping")
    fixed = {
        "stage3_max_seconds": STAGE3_MAX_SECONDS,
        "stage3_max_generations": STAGE3_MAX_GENERATIONS,
        "global_max_seconds": GLOBAL_MAX_SECONDS,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "max_assistant_prompt_versions": MAX_ASSISTANT_PROMPT_VERSIONS,
        "max_extraction_versions": MAX_EXTRACTION_VERSIONS,
    }
    for name, expected in fixed.items():
        if caps.get(name) != expected:
            raise BudgetLedgerError(
                f"opening cap {name} disagrees with the authorized ceiling"
            )
    for name, value in caps.items():
        if not _is_int(value) or value < 0:
            raise BudgetLedgerError(f"opening cap {name} must be a nonnegative int")
    for name in (
        "stage4_max_seconds",
        "stage4_max_generations",
        "max_training_jobs",
        "training_job_max_seconds",
        "max_profiles",
        "profile_max_seconds",
        "min_free_disk_bytes",
    ):
        if name not in caps:
            raise BudgetLedgerError(f"opening misses the {name} cap")
    return dict(caps)


def _import_opening(path: Path, expected_sha256: str) -> dict:
    """Read and fully verify the immutable opening; nothing is cached."""
    if expected_sha256 != OPENING_SHA256:
        raise BudgetLedgerError(
            "the budget ledger is bound to the frozen stage1 opening hash"
        )
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise BudgetLedgerError(
            f"the immutable opening is unreadable: {error}"
        ) from error
    digest = _sha256(raw)
    if digest != expected_sha256:
        raise BudgetLedgerError(
            "the opening bytes changed: refusing to charge a substituted budget"
        )
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise BudgetLedgerError(f"the opening is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise BudgetLedgerError("the opening must be a JSON object")
    if document.get("schema_version") != OPENING_SCHEMA_VERSION:
        raise BudgetLedgerError("unexpected opening schema version")
    caps = _validate_caps(document.get("caps"))
    reservations = document.get("reservations")
    if not isinstance(reservations, list):
        raise BudgetLedgerError("the opening reservations must be a list")
    if len(reservations) != IMPORTED_RESERVATION_COUNT:
        raise BudgetLedgerError(
            "the opening no longer holds the 416 reconciled reservations"
        )
    records = [_validate_imported_record(record) for record in reservations]
    identifiers = [record["reservation_id"] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise BudgetLedgerError("the opening repeats a reservation id")
    usage = _sum_usage(records)
    if usage != {
        "seconds": IMPORTED_SECONDS,
        "generations": IMPORTED_GENERATIONS,
        "storage_bytes": IMPORTED_STORAGE_BYTES,
    }:
        raise BudgetLedgerError(
            "the opening records do not add up to the reconciled prior usage"
        )
    accounts = document.get("prior_usage_by_historical_account")
    if not isinstance(accounts, dict):
        raise BudgetLedgerError("the opening misses the historical accounts")
    if accounts.get("stage3") != {
        "seconds": usage["seconds"],
        "generations": usage["generations"],
        "bytes": usage["storage_bytes"],
    }:
        raise BudgetLedgerError("the opening stage3 totals disagree with its records")
    if accounts.get("stage4") != {"seconds": 0, "generations": 0, "bytes": 0}:
        raise BudgetLedgerError("stage4 historical usage must stay unused")
    if document.get("prior_global_seconds") != usage["seconds"]:
        raise BudgetLedgerError("the opening global seconds disagree with its records")
    if document.get("prior_storage_bytes") != usage["storage_bytes"]:
        raise BudgetLedgerError("the opening storage total disagrees with its records")
    if document.get("prior_training_jobs") != 0:
        raise BudgetLedgerError("the opening must not carry training jobs")
    if document.get("prior_profiles") != 0:
        raise BudgetLedgerError("the opening must not carry profile attempts")
    versions = document.get("registered_prompt_versions")
    if not isinstance(versions, dict) or set(versions) != set(VERSION_KINDS):
        raise BudgetLedgerError(
            "the opening must register prompt and extraction versions"
        )
    checked_versions = {}
    for kind in VERSION_KINDS:
        entries = versions[kind]
        if not isinstance(entries, list):
            raise BudgetLedgerError("opening version registrations must be lists")
        checked = [_validate_version_record(entry) for entry in entries]
        names = [entry["version_id"] for entry in checked]
        if len(set(names)) != len(names):
            raise BudgetLedgerError("the opening repeats a version id")
        if len(checked) > caps[VERSION_LIMITS[kind]]:
            raise BudgetLedgerError("the opening exceeds a version ceiling")
        checked_versions[kind] = checked
    unfinished = document.get("unfinished_reservation_ids")
    if not isinstance(unfinished, list) or not all(
        isinstance(identifier, str) for identifier in unfinished
    ):
        raise BudgetLedgerError("the opening unfinished ids must be a list of ids")
    derived_unfinished = [
        record["reservation_id"] for record in records if record["status"] == "reserved"
    ]
    if list(unfinished) != derived_unfinished:
        raise BudgetLedgerError(
            "the opening unfinished ids disagree with its reservation records"
        )
    return {
        "sha256": digest,
        "bytes": len(raw),
        "caps": caps,
        "records": records,
        "usage": usage,
        "versions": checked_versions,
        "unfinished": list(unfinished),
    }


def _reset_scope() -> dict:
    return {
        "reset_generation_usage": True,
        "reset_development_and_global_seconds_usage": True,
        "reset_storage_usage": False,
        "reset_version_registrations": False,
        "rewrite_historical_reservations_or_receipts": False,
        "authorize_Qwen_execution": False,
    }


def _import_reset(path: Path, expected_sha256: str, opening_info: dict) -> dict:
    """Verify the authorized reset authority against the opening and caps."""
    if expected_sha256 != RESET_AUTHORITY_SHA256:
        raise BudgetLedgerError(
            "the usage reset authority is hash-bound; refusing another document"
        )
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise BudgetLedgerError(
            f"the usage reset authority is unreadable: {error}"
        ) from error
    digest = _sha256(raw)
    if digest != expected_sha256:
        raise BudgetLedgerError(
            "the usage reset authority bytes changed; refusing a stale or"
            " duplicate reset id"
        )
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise BudgetLedgerError(
            f"the usage reset authority is not JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise BudgetLedgerError("the usage reset authority must be a JSON object")
    if document.get("schema_version") != RESET_SCHEMA_VERSION:
        raise BudgetLedgerError("unexpected usage reset schema version")
    if document.get("reset_id") != RESET_ID:
        raise BudgetLedgerError("the usage reset id disagrees with the authority")
    source = document.get("source_opening")
    if not isinstance(source, dict) or source.get("sha256") != OPENING_SHA256:
        raise BudgetLedgerError("the usage reset is bound to another opening")
    if not _nonblank(source.get("path")):
        raise BudgetLedgerError("the usage reset must name its source opening")
    caps = opening_info["caps"]
    expected_caps = {
        "development_generations": caps["stage3_max_generations"],
        "development_seconds": caps["stage3_max_seconds"],
        "global_seconds": caps["global_max_seconds"],
        "storage_bytes": caps["max_storage_bytes"],
    }
    if document.get("unchanged_caps") != expected_caps:
        raise BudgetLedgerError("the usage reset disagrees with the opening caps")
    if document.get("scope") != _reset_scope():
        raise BudgetLedgerError(
            "the usage reset scope may not reset storage or version registrations"
        )
    preserved = document.get("historical_usage_preserved")
    expected_preserved = {
        "generations": opening_info["usage"]["generations"],
        "seconds": opening_info["usage"]["seconds"],
        "storage_bytes": opening_info["usage"]["storage_bytes"],
        "reservation_records": len(opening_info["records"]),
    }
    if preserved != expected_preserved:
        raise BudgetLedgerError(
            "the usage reset historical usage disagrees with the opening records"
        )
    epoch_usage = document.get("new_epoch_opening_usage")
    expected_epoch = {
        "generations": 0,
        "seconds": 0,
        "storage_bytes": expected_preserved["storage_bytes"],
    }
    if epoch_usage != expected_epoch:
        raise BudgetLedgerError(
            "the usage reset must zero generations/seconds and preserve storage"
        )
    return {
        "sha256": digest,
        "reset_id": RESET_ID,
        "path": str(path),
    }


def _field(spending, name, default=None):
    if isinstance(spending, Mapping):
        return spending.get(name, default)
    return getattr(spending, name, default)


def _integrity(document: dict) -> str:
    body = {key: value for key, value in document.items() if key != "integrity_sha256"}
    return _sha256(_canonical(body))


class BudgetLedger:
    """Exclusive, atomic Stage 4A budget accounting for one portable ledger."""

    def __init__(
        self,
        path,
        opening_path,
        opening_sha256,
        *,
        reset_path=None,
        reset_sha256=None,
        revision_path=None,
        revision_sha256=None,
        apply_revision_reset=False,
    ):
        if (reset_path is None) != (reset_sha256 is None):
            raise BudgetLedgerError(
                "the authorized reset needs both reset_path and reset_sha256"
            )
        self._path = Path(path)
        self._opening_path = Path(opening_path)
        self._opening_sha256 = str(opening_sha256)
        self._reset_path = None if reset_path is None else Path(reset_path)
        self._reset_sha256 = None if reset_sha256 is None else str(reset_sha256)
        if (revision_path is None) != (revision_sha256 is None):
            raise BudgetLedgerError("revision requires its path and trusted hash")
        if apply_revision_reset and revision_path is None:
            raise BudgetLedgerError("a revision reset requires explicit authority")
        self._revision_path = None if revision_path is None else Path(revision_path)
        self._revision_sha256 = revision_sha256
        self._revision_info = None
        self._check_paths()
        opening_info, reset_info = self._authorities()
        with self._lock(exclusive=True):
            if self._path.exists():
                document = self._read_document()
                self._revalidate(
                    document, opening_info, reset_info, mode_required=False
                )
                current = document["epochs"][document["current_epoch"]]
                if reset_info is not None and current["mode"] == EPOCH_OPENING:
                    self._advance_epoch(document, opening_info, reset_info)
                if (
                    apply_revision_reset
                    and self._revision_info is not None
                    and document["epochs"][document["current_epoch"]]["mode"]
                    != EPOCH_REVISION
                ):
                    self._advance_revision(document, opening_info, reset_info)
                self._revalidate(document, opening_info, reset_info)
            else:
                if revision_path is not None:
                    raise BudgetLedgerError("revision requires the existing ledger")
                self._write(self._new_document(opening_info, reset_info))

    # -- public surface -------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def reserve(
        self,
        kind,
        seconds,
        generations=0,
        storage_bytes=0,
        refs=None,
    ) -> str:
        """Reserve the planned maximum before dispatch; returns the id."""
        seconds = _checked_charge(seconds, "planned seconds")
        generations = _checked_charge(generations, "planned generations")
        storage_bytes = _checked_charge(storage_bytes, "planned storage bytes")
        checked_refs = _checked_refs(refs)
        if kind in REFUSED_KINDS:
            raise BudgetLedgerError(REFUSED_KINDS[kind])
        if kind not in ALLOWED_KINDS:
            raise BudgetLedgerError(f"unknown reservation kind: {kind!r}")

        def change(document):
            if "repair_series_id" in checked_refs:
                if kind != "generation":
                    raise BudgetLedgerError(
                        "repair series calls use generation accounting only"
                    )
                self._check_repair_slot(checked_refs, seconds, generations)
                if any(
                    old["refs"].get("repair_series_id")
                    == checked_refs["repair_series_id"]
                    and old["refs"].get("round") == checked_refs.get("round")
                    and old["refs"].get("diagnostic_slot")
                    == checked_refs.get("diagnostic_slot")
                    for old in document["reservations"]
                ):
                    raise BudgetLedgerError(
                        "repair series slot already reserved; no retry"
                    )
            if "diagnostic_id" in checked_refs:
                if kind != "generation":
                    raise BudgetLedgerError(
                        "diagnostic calls use generation accounting only"
                    )
                self._check_diagnostic_slot(checked_refs, seconds, generations)
                if any(
                    old["refs"].get("diagnostic_id") == checked_refs["diagnostic_id"]
                    and old["refs"].get("diagnostic_slot")
                    == checked_refs.get("diagnostic_slot")
                    for old in document["reservations"]
                ):
                    raise BudgetLedgerError(
                        "diagnostic slot already reserved; no retry"
                    )
            attempt_key = checked_refs.get("attempt_key")
            revised = self._revision_info is not None
            if kind == "row" and revised:
                validate_row_scope(self._revision_info, checked_refs)
            if (
                kind == "row"
                and attempt_key
                and any(
                    old["kind"] == "row"
                    and old["status"] != "cancelled"
                    and old["refs"].get("attempt_key") == attempt_key
                    and (
                        not revised
                        or old["refs"].get("experiment_id")
                        == checked_refs.get("experiment_id")
                    )
                    for old in document["reservations"]
                )
            ):
                raise BudgetLedgerError(
                    "planned request/condition already attempted or reserved; retries refused"
                )
            record = {
                "reservation_id": str(uuid.uuid4()),
                "kind": kind,
                "status": "reserved",
                "planned_seconds": seconds,
                "planned_generations": generations,
                "planned_bytes": storage_bytes,
                "actual_seconds": 0,
                "actual_generations": 0,
                "actual_bytes": 0,
                "created_at": _now(),
                "finished_at": "",
                "cancel_reason": "",
                "never_launched": False,
                "over_planned": False,
                "refs": checked_refs,
                "epoch": document["current_epoch"],
            }
            self._validate_own_record(record, len(document["epochs"]))
            self._check_blockers(document, checked_refs)
            self._check_caps(document, seconds, generations, storage_bytes)
            document["reservations"].append(record)
            return document

        document = self._mutate(change)
        return document["reservations"][-1]["reservation_id"]

    def settle(
        self,
        identifier,
        status,
        seconds,
        generations=0,
        storage_bytes=0,
        never_launched=False,
    ):
        """Settle one reservation; finished/failed keep observed or max charges."""
        if status not in SETTLE_STATUSES:
            raise BudgetLedgerError(
                "settle status must be finished/failed/cancelled; a crash"
                " leaves the reservation reserved and charged"
            )
        seconds = _checked_charge(seconds, "actual seconds")
        generations = _checked_charge(generations, "actual generations")
        storage_bytes = _checked_charge(storage_bytes, "actual storage bytes")
        if not isinstance(never_launched, bool):
            raise BudgetLedgerError("never_launched must be a boolean proof")
        if status == "cancelled":
            if never_launched is not True:
                raise BudgetLedgerError(
                    "cancelling is zero only with proof the work never launched"
                )
            if seconds or generations or storage_bytes:
                raise BudgetLedgerError(
                    "a never-launched reservation cannot report observed usage"
                )

        def change(document):
            index = next(
                (
                    position
                    for position, record in enumerate(document["reservations"])
                    if record["reservation_id"] == identifier
                ),
                None,
            )
            if index is None:
                imported = {
                    record["reservation_id"] for record in document["imported_records"]
                }
                if identifier in imported:
                    raise BudgetLedgerError(
                        "imported opening reservations are immutable history"
                    )
                raise BudgetLedgerError(f"unknown reservation: {identifier}")
            record = document["reservations"][index]
            if record["status"] != "reserved":
                raise BudgetLedgerError(
                    "reservation already settled; replays are refused"
                )
            updated = dict(record)
            updated["status"] = status
            updated["finished_at"] = _now()
            updated["never_launched"] = never_launched
            if status == "cancelled":
                updated["cancel_reason"] = "proved never launched"
                updated["actual_seconds"] = 0
                updated["actual_generations"] = 0
                updated["actual_bytes"] = 0
            else:
                updated["actual_seconds"] = seconds
                updated["actual_generations"] = generations
                updated["actual_bytes"] = storage_bytes
            updated["over_planned"] = any(
                updated[actual] > updated[planned]
                for actual, planned in (
                    ("actual_seconds", "planned_seconds"),
                    ("actual_generations", "planned_generations"),
                    ("actual_bytes", "planned_bytes"),
                )
            )
            self._validate_own_record(updated, len(document["epochs"]))
            document["reservations"][index] = updated
            return document

        document = self._mutate(change)
        return next(
            record
            for record in document["reservations"]
            if record["reservation_id"] == identifier
        )

    def settle_storage(self, identifier, status, *, output_bytes, ledger_start_bytes):
        """Settle final retained bytes, including this ledger's own growth.

        The coordinator closes all output writes first and holds the model mutex.
        Serialize the prospective settlement under the mutation lock to count its
        own metadata exactly, not a fabricated full-capacity 'actual'. A crash
        before replace still leaves the full reservation charged. No model work or
        outcome is retried; an impossible size fixed point simply fails closed.
        """
        output_bytes = _checked_charge(output_bytes, "observed output bytes")
        ledger_start_bytes = _checked_charge(ledger_start_bytes, "opening ledger bytes")
        if status not in ("finished", "failed"):
            raise BudgetLedgerError("storage settlement requires finished or failed")

        def change(document):
            record = next(
                (
                    r
                    for r in document["reservations"]
                    if r["reservation_id"] == identifier
                ),
                None,
            )
            if record is None or record["kind"] != "storage":
                raise BudgetLedgerError("unknown storage reservation")
            if record["status"] != "reserved":
                raise BudgetLedgerError("storage reservation already settled")
            record["status"] = status
            record["finished_at"] = _now()
            document["revision"] += 1  # Anticipate _mutate's single increment.
            for _ in range(32):
                serialized = copy.deepcopy(document)
                serialized["integrity_sha256"] = _integrity(serialized)
                size = len(
                    (
                        json.dumps(
                            serialized, ensure_ascii=False, sort_keys=True, indent=2
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                actual = output_bytes + max(0, size - ledger_start_bytes)
                over = actual > record["planned_bytes"]
                if record["actual_bytes"] == actual and record["over_planned"] == over:
                    self._validate_own_record(record, len(document["epochs"]))
                    return document
                record["actual_bytes"] = actual
                record["over_planned"] = over
            raise BudgetLedgerError(
                "storage settlement size did not stabilize; reservation retained"
            )

        document = self._mutate(change)
        return next(
            r for r in document["reservations"] if r["reservation_id"] == identifier
        )

    def register_versions(self, spending) -> dict:
        """Register the shared assistant/extraction versions before execution."""
        assistant = _field(spending, "assistant_version", None)
        if assistant is None:
            raise BudgetLedgerError("spending must name the assistant prompt version")
        assistant_refs = _checked_refs(
            _field(spending, "assistant_version_refs", {}) or {}
        )
        extraction = _field(spending, "extraction_version", None)
        extraction_refs = _checked_refs(
            _field(spending, "extraction_version_refs", {}) or {}
        )
        if extraction is None and extraction_refs:
            raise BudgetLedgerError(
                "extraction refs were supplied without an extraction version"
            )
        requested = {"assistant_prompt": (assistant, assistant_refs)}
        if extraction is not None:
            requested["extraction"] = (extraction, extraction_refs)

        def change(document):
            for kind, (version_id, refs) in requested.items():
                if not _nonblank(version_id):
                    raise BudgetLedgerError("version ids must be nonblank")
                registry = document["prompt_versions"][kind]
                existing = next(
                    (
                        record
                        for record in registry
                        if record["version_id"] == version_id
                    ),
                    None,
                )
                if existing is not None:
                    if existing["refs"] != refs:
                        raise BudgetLedgerError(
                            f"version {version_id!r} collides with another identity"
                        )
                    continue
                limit = document["opening"]["caps"][VERSION_LIMITS[kind]]
                if kind == "assistant_prompt" and len(registry) == limit == 4:
                    self._check_diagnostic_version(registry, version_id, refs)
                    limit = 5
                elif (
                    kind == "assistant_prompt"
                    and len(registry) >= 5
                    and "repair_series_id" in refs
                ):
                    self._check_repair_version(registry, version_id, refs)
                    limit = 10
                if len(registry) >= limit:
                    raise BudgetLedgerError(
                        f"{kind} version registrations are exhausted at {limit}"
                    )
                registry.append(
                    {
                        "version_id": version_id,
                        "registered_at": _now(),
                        "refs": dict(refs),
                    }
                )
            return document

        document = self._mutate(change)
        return {
            "assistant_prompt": next(
                (
                    record
                    for record in document["prompt_versions"]["assistant_prompt"]
                    if record["version_id"] == assistant
                ),
                None,
            ),
            "extraction": (
                next(
                    (
                        record
                        for record in document["prompt_versions"]["extraction"]
                        if record["version_id"] == extraction
                    ),
                    None,
                )
                if extraction is not None
                else None
            ),
            "version_remaining": {
                kind: self._effective_version_limit(document, kind)
                - len(document["prompt_versions"][kind])
                for kind in VERSION_KINDS
            },
            "epoch": document["current_epoch"],
        }

    def charge(self, identifier) -> tuple[int, int, int]:
        """The conservative charged triple for one own or imported reservation."""
        with self._lock(exclusive=False):
            opening_info, reset_info = self._authorities()
            document = self._read_document()
            self._revalidate(document, opening_info, reset_info)
        for record in document["reservations"]:
            if record["reservation_id"] == identifier:
                return _charged(record)
        for record in document["imported_records"]:
            if record["reservation_id"] == identifier:
                return _charged(record)
        raise BudgetLedgerError(f"unknown reservation: {identifier}")

    def snapshot(self) -> dict:
        """Committed usage, remaining allowances and every own reservation."""
        with self._lock(exclusive=False):
            opening_info, reset_info = self._authorities()
            document = self._read_document()
            self._revalidate(document, opening_info, reset_info)
            return self._snapshot_document(document, opening_info)

    # -- validation -----------------------------------------------------

    def _effective_version_limit(self, document, kind):
        entries = document["prompt_versions"][kind]
        if kind == "assistant_prompt" and len(entries) > 5:
            return 10
        return max(document["opening"]["caps"][VERSION_LIMITS[kind]], len(entries))

    def _repair_authority(self):
        from messaging_nlp.repair_authority import AUTHORITY_NAME, load_authority

        if self._reset_path is None:
            raise BudgetLedgerError(
                "repair series requires the retained reset authority"
            )
        try:
            return load_authority(self._reset_path.with_name(AUTHORITY_NAME))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise BudgetLedgerError(
                f"invalid repair series authority: {error}"
            ) from error

    def _check_repair_version(self, entries, version_id, refs):
        from messaging_nlp.repair_authority import check_version

        try:
            check_version(self._repair_authority(), entries, version_id, refs)
        except ValueError as error:
            raise BudgetLedgerError(str(error)) from error

    def _check_registered_repairs(self, entries):
        from messaging_nlp.repair_authority import check_registered_versions

        try:
            check_registered_versions(self._repair_authority(), entries)
        except ValueError as error:
            raise BudgetLedgerError(str(error)) from error

    def _check_repair_slot(self, refs, seconds, generations):
        from messaging_nlp.repair_authority import check_slot

        try:
            check_slot(self._repair_authority(), refs, seconds, generations)
        except ValueError as error:
            raise BudgetLedgerError(str(error)) from error

    def _diagnostic_authority(self):
        from messaging_nlp.diagnostic_authority import AUTHORITY_NAME, load_authority

        if self._reset_path is None:
            raise BudgetLedgerError(
                "diagnostic extension requires the retained reset authority"
            )
        try:
            return load_authority(self._reset_path.with_name(AUTHORITY_NAME))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise BudgetLedgerError(f"invalid diagnostic authority: {error}") from error

    def _check_diagnostic_version(self, entries, version_id, refs):
        from messaging_nlp.diagnostic_authority import check_version

        try:
            check_version(self._diagnostic_authority(), entries, version_id, refs)
        except ValueError as error:
            raise BudgetLedgerError(str(error)) from error

    def _check_diagnostic_slot(self, refs, seconds, generations):
        from messaging_nlp.diagnostic_authority import check_slot

        try:
            check_slot(self._diagnostic_authority(), refs, seconds, generations)
        except ValueError as error:
            raise BudgetLedgerError(str(error)) from error

    def _check_paths(self) -> None:
        candidate = os.path.abspath(self._path)
        if ".git" in Path(candidate).parts:
            raise BudgetLedgerError(
                "the portable ledger may not live inside a .git directory"
            )
        if self._path.name in ("", ".", ".."):
            raise BudgetLedgerError("the ledger needs a real file name")
        opening = os.path.abspath(self._opening_path)
        if candidate == opening:
            raise BudgetLedgerError("the ledger may not overwrite the opening")
        if self._reset_path is not None:
            if candidate == os.path.abspath(self._reset_path):
                raise BudgetLedgerError(
                    "the ledger may not overwrite the reset authority"
                )

    def _authorities(self) -> tuple[dict, dict | None]:
        opening_info = _import_opening(self._opening_path, self._opening_sha256)
        reset_info = None
        if self._reset_path is not None:
            reset_info = _import_reset(
                self._reset_path, self._reset_sha256, opening_info
            )
        if self._revision_path is not None:
            try:
                self._revision_info = load_authority(
                    self._revision_path, self._revision_sha256
                )
                caps = opening_info["caps"]
                if self._revision_info["body"]["unchanged_caps"] != {
                    "development_generations": caps["stage3_max_generations"],
                    "development_seconds": caps["stage3_max_seconds"],
                    "global_seconds": caps["global_max_seconds"],
                    "storage_bytes": caps["max_storage_bytes"],
                    "assistant_versions": caps["max_assistant_prompt_versions"],
                    "extraction_versions": caps["max_extraction_versions"],
                }:
                    raise ValueError("revision cannot change budget caps")
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise BudgetLedgerError(
                    f"invalid revision authority: {error}"
                ) from error
        return opening_info, reset_info

    def _validate_own_record(self, record, epochs: int) -> None:
        if not isinstance(record, dict) or set(record) != _OWN_KEYS:
            raise BudgetLedgerError("ledger reservation fields changed")
        identifier = record["reservation_id"]
        kind = record["kind"]
        status = record["status"]
        if not _nonblank(identifier):
            raise BudgetLedgerError("a reservation needs an id")
        if kind not in ALLOWED_KINDS:
            raise BudgetLedgerError(f"unknown reservation kind: {kind!r}")
        if status not in RESERVATION_STATUSES:
            raise BudgetLedgerError(f"unknown reservation status: {status!r}")
        for field in (
            "planned_seconds",
            "planned_generations",
            "planned_bytes",
            "actual_seconds",
            "actual_generations",
            "actual_bytes",
        ):
            if not _is_int(record[field]) or record[field] < 0:
                raise BudgetLedgerError(f"{field} must be a nonnegative integer")
        if not _is_int(record["epoch"]) or not 0 <= record["epoch"] < epochs:
            raise BudgetLedgerError("a reservation must belong to a known epoch")
        _checked_refs(record["refs"])
        if not _nonblank(record["created_at"]):
            raise BudgetLedgerError("a reservation needs its creation time")
        if not isinstance(record["finished_at"], str):
            raise BudgetLedgerError("finished_at must be a string")
        if not isinstance(record["cancel_reason"], str):
            raise BudgetLedgerError("cancel_reason must be a string")
        if not isinstance(record["never_launched"], bool):
            raise BudgetLedgerError("never_launched must be a boolean")
        if not isinstance(record["over_planned"], bool):
            raise BudgetLedgerError("over_planned must be a boolean")
        if kind == "storage":
            if record["planned_seconds"] or record["planned_generations"]:
                raise BudgetLedgerError("storage reservations charge bytes only")
            if record["planned_bytes"] <= 0:
                raise BudgetLedgerError("a storage reservation needs its bytes")
        elif kind in SECONDS_ONLY_KINDS:
            if record["planned_seconds"] <= 0:
                raise BudgetLedgerError(f"{kind} reservations need their seconds")
            if record["planned_generations"] or record["planned_bytes"]:
                raise BudgetLedgerError(
                    f"{kind} reservations are charged in seconds only"
                )
        else:
            if record["planned_generations"] <= 0:
                raise BudgetLedgerError(
                    "generation reservations need a generation charge"
                )
            if record["planned_bytes"]:
                raise BudgetLedgerError(
                    "storage must be reserved as a storage reservation"
                )
        if status == "reserved":
            if any(
                record[field]
                for field in ("actual_seconds", "actual_generations", "actual_bytes")
            ):
                raise BudgetLedgerError("an unfinished reservation has no actuals")
            if record["finished_at"] or record["never_launched"]:
                raise BudgetLedgerError("an unfinished reservation is not settled")
            if record["over_planned"]:
                raise BudgetLedgerError("an unfinished reservation cannot overrun")
        elif status == "cancelled":
            if not record["never_launched"]:
                raise BudgetLedgerError(
                    "only a proved never-launched reservation may be cancelled"
                )
            if any(
                record[field]
                for field in ("actual_seconds", "actual_generations", "actual_bytes")
            ):
                raise BudgetLedgerError("a cancelled reservation has no charges")
            if not record["finished_at"] or not _nonblank(record["cancel_reason"]):
                raise BudgetLedgerError("a cancellation must be recorded")
            if record["over_planned"]:
                raise BudgetLedgerError("a cancelled reservation cannot overrun")
        else:
            if record["never_launched"]:
                raise BudgetLedgerError(
                    "never_launched is only valid for a cancellation"
                )
            if not record["finished_at"]:
                raise BudgetLedgerError("a settled reservation needs a finish time")
            if kind == "storage":
                if record["actual_seconds"] or record["actual_generations"]:
                    raise BudgetLedgerError("storage settles bytes only")
                if status == "finished" and record["actual_bytes"] <= 0:
                    raise BudgetLedgerError(
                        "a finished storage reservation must report its bytes;"
                        " cancel an unused reservation instead"
                    )
            else:
                if record["actual_bytes"]:
                    raise BudgetLedgerError(
                        "a non-storage reservation cannot settle storage bytes"
                    )
                if kind in GENERATION_KINDS and status == "finished":
                    if record["actual_generations"] < 1:
                        raise BudgetLedgerError(
                            "a finished generation must report its generation"
                        )
        expected_over = any(
            record[actual] > record[planned]
            for actual, planned in (
                ("actual_seconds", "planned_seconds"),
                ("actual_generations", "planned_generations"),
                ("actual_bytes", "planned_bytes"),
            )
        )
        if record["over_planned"] != expected_over:
            raise BudgetLedgerError("over_planned must describe the actual charges")

    def _validate_document(self, document, opening_info, *, verify_integrity=True):
        if not isinstance(document, dict) or set(document) != _LEDGER_KEYS:
            raise BudgetLedgerError("ledger document fields changed")
        if document["schema_version"] != SCHEMA_VERSION:
            raise BudgetLedgerError("unexpected ledger schema version")
        if not _is_int(document["revision"]) or document["revision"] < 1:
            raise BudgetLedgerError("ledger revision must be a positive integer")
        if verify_integrity:
            if document["integrity_sha256"] != _integrity(document):
                raise BudgetLedgerError(
                    "ledger integrity does not match its content; external edits"
                    " or reset attempts are refused"
                )
        anchor = document["opening"]
        if not isinstance(anchor, dict) or set(anchor) != _ANCHOR_KEYS:
            raise BudgetLedgerError("the ledger opening anchor changed")
        if anchor["sha256"] != opening_info["sha256"]:
            raise BudgetLedgerError("the ledger is anchored to another opening")
        if not isinstance(anchor["path"], str) or not anchor["path"]:
            raise BudgetLedgerError("the ledger opening anchor needs its path")
        if not _is_int(anchor["bytes"]) or anchor["bytes"] != opening_info["bytes"]:
            raise BudgetLedgerError("the anchored opening size changed")
        if anchor["caps"] != opening_info["caps"]:
            raise BudgetLedgerError("the anchored caps changed")
        if anchor["reservation_count"] != len(opening_info["records"]):
            raise BudgetLedgerError("the anchored record count changed")
        if anchor["derived_usage"] != opening_info["usage"]:
            raise BudgetLedgerError("the anchored imported usage changed")
        if anchor["registered_versions"] != opening_info["versions"]:
            raise BudgetLedgerError("the anchored version registry changed")
        if anchor["unfinished_reservation_ids"] != opening_info["unfinished"]:
            raise BudgetLedgerError("the anchored unfinished ids changed")
        if document["imported_records"] != opening_info["records"]:
            raise BudgetLedgerError(
                "imported opening records were rewritten; failing closed"
            )
        epochs = document["epochs"]
        if not isinstance(epochs, list) or not epochs:
            raise BudgetLedgerError("the ledger needs its epoch history")
        if not isinstance(epochs[0], dict) or epochs[0].get("mode") != EPOCH_OPENING:
            raise BudgetLedgerError(
                "the imported opening epoch must stay the first epoch"
            )
        reset_seen = False
        for position, entry in enumerate(epochs):
            if not isinstance(entry, dict) or set(entry) != _EPOCH_KEYS:
                raise BudgetLedgerError("ledger epoch fields changed")
            if entry["index"] != position:
                raise BudgetLedgerError("ledger epochs must stay contiguous")
            if not _nonblank(entry["started_at"]):
                raise BudgetLedgerError("an epoch needs its start time")
            base = _checked_usage(entry["base_usage"], "epoch")
            if entry["source_opening_sha256"] != opening_info["sha256"]:
                raise BudgetLedgerError("an epoch is bound to another opening")
            if entry["mode"] == EPOCH_OPENING:
                if reset_seen:
                    raise BudgetLedgerError(
                        "the opening epoch may not follow the reset epoch"
                    )
                if base != opening_info["usage"]:
                    raise BudgetLedgerError("the opening epoch base usage changed")
                if entry["reset_id"] or entry["reset_sha256"] or entry["reset_path"]:
                    raise BudgetLedgerError("the opening epoch cannot cite a reset")
            elif entry["mode"] == EPOCH_RESET:
                if reset_seen:
                    raise BudgetLedgerError(
                        "the authorized reset may be applied exactly once"
                    )
                reset_seen = True
                if entry["reset_id"] != RESET_ID:
                    raise BudgetLedgerError("the reset epoch id changed")
                if entry["reset_sha256"] != RESET_AUTHORITY_SHA256:
                    raise BudgetLedgerError("the reset epoch authority changed")
                if base["seconds"] or base["generations"]:
                    raise BudgetLedgerError(
                        "the reset epoch must zero generation/seconds usage"
                    )
                if base["storage_bytes"] < opening_info["usage"]["storage_bytes"]:
                    raise BudgetLedgerError(
                        "the reset epoch cannot refund or reduce storage usage"
                    )
            elif entry["mode"] == EPOCH_REVISION:
                info = self._revision_info
                if info is None or position != info["body"]["predecessor_epoch"] + 1:
                    raise BudgetLedgerError("revision epoch requires its own authority")
                previous = info["predecessor"]
                if (
                    entry["reset_id"] != info["body"]["reset_id"]
                    or entry["reset_sha256"] != info["sha256"]
                    or not entry["reset_path"]
                    or base["seconds"]
                    or base["generations"]
                    or base["storage_bytes"]
                    != self._epoch_usage(previous, previous["current_epoch"])[0][
                        "storage_bytes"
                    ]
                    or epochs[:position] != previous["epochs"]
                    or document["reservations"][: len(previous["reservations"])]
                    != previous["reservations"]
                ):
                    raise BudgetLedgerError(
                        "revision history or reset identity changed"
                    )
                for kind in VERSION_KINDS:
                    prefix = previous["prompt_versions"][kind]
                    if document["prompt_versions"][kind][: len(prefix)] != prefix:
                        raise BudgetLedgerError("revision rewrote prior versions")
                if any(
                    record["epoch"] != position
                    for record in document["reservations"][
                        len(previous["reservations"]) :
                    ]
                ):
                    raise BudgetLedgerError("revision moved new charges into history")
            else:
                raise BudgetLedgerError(f"unknown epoch mode: {entry['mode']!r}")
        current = document["current_epoch"]
        if not _is_int(current) or not 0 <= current < len(epochs):
            raise BudgetLedgerError("the current epoch index is invalid")
        if current != len(epochs) - 1:
            raise BudgetLedgerError("only the newest epoch may be current")
        if document["prompt_versions"] is None:
            raise BudgetLedgerError("the ledger misses its version registry")
        versions = document["prompt_versions"]
        if not isinstance(versions, dict) or set(versions) != set(VERSION_KINDS):
            raise BudgetLedgerError("the ledger version registry changed")
        for kind in VERSION_KINDS:
            entries = versions[kind]
            if not isinstance(entries, list):
                raise BudgetLedgerError("version registrations must be lists")
            checked = [_validate_version_record(entry) for entry in entries]
            if len(checked) > opening_info["caps"][VERSION_LIMITS[kind]]:
                if kind != "assistant_prompt" or not 5 <= len(checked) <= 10:
                    raise BudgetLedgerError("the version registry exceeds its ceiling")
                self._check_diagnostic_version(
                    checked[:5], checked[4]["version_id"], checked[4]["refs"]
                )
                if len(checked) > 5:
                    self._check_registered_repairs(checked)
            imported = opening_info["versions"][kind]
            if checked[: len(imported)] != imported:
                raise BudgetLedgerError(
                    "imported version registrations changed; failing closed"
                )
        reservations = document["reservations"]
        if not isinstance(reservations, list):
            raise BudgetLedgerError("ledger reservations must be a list")
        imported_ids = {record["reservation_id"] for record in opening_info["records"]}
        seen = set()
        diagnostic_slots = set()
        repair_slots = set()
        for record in reservations:
            self._validate_own_record(record, len(epochs))
            if "repair_series_id" in record["refs"]:
                if record["kind"] != "generation":
                    raise BudgetLedgerError("invalid repair series accounting kind")
                self._check_repair_slot(
                    record["refs"],
                    record["planned_seconds"],
                    record["planned_generations"],
                )
                slot = (record["refs"]["round"], record["refs"]["diagnostic_slot"])
                if slot in repair_slots:
                    raise BudgetLedgerError("duplicate repair series slot in ledger")
                repair_slots.add(slot)
            if "diagnostic_id" in record["refs"]:
                if record["kind"] != "generation":
                    raise BudgetLedgerError("invalid diagnostic accounting kind")
                self._check_diagnostic_slot(
                    record["refs"],
                    record["planned_seconds"],
                    record["planned_generations"],
                )
                slot = record["refs"]["diagnostic_slot"]
                if slot in diagnostic_slots:
                    raise BudgetLedgerError("duplicate diagnostic slot in ledger")
                diagnostic_slots.add(slot)
            identifier = record["reservation_id"]
            if identifier in imported_ids or identifier in seen:
                raise BudgetLedgerError("ledger reservation ids must stay unique")
            if record["epoch"] > current:
                raise BudgetLedgerError("a reservation cannot belong to a future epoch")
            if (
                self._revision_info is not None
                and record["epoch"] > self._revision_info["body"]["predecessor_epoch"]
                and record["kind"] == "row"
            ):
                validate_row_scope(self._revision_info, record["refs"])
            seen.add(identifier)

    def _revalidate(
        self,
        document,
        opening_info,
        reset_info,
        *,
        mode_required=True,
        verify_integrity=True,
    ) -> None:
        if document.get("schema_version") != SCHEMA_VERSION:
            raise BudgetLedgerError("unexpected ledger schema version")
        self._validate_document(
            document,
            opening_info,
            verify_integrity=verify_integrity,
        )
        if mode_required:
            current = document["epochs"][document["current_epoch"]]
            if self._revision_info is not None:
                if current["mode"] != EPOCH_REVISION:
                    raise BudgetLedgerError(
                        "authorized revision reset has not been applied"
                    )
                return
            if reset_info is None:
                if current["mode"] != EPOCH_OPENING:
                    raise BudgetLedgerError(
                        "the ledger already applied the authorized reset epoch;"
                        " reopening without the reset authority is refused"
                    )
            else:
                if current["mode"] != EPOCH_RESET:
                    raise BudgetLedgerError(
                        "the ledger has not applied the authorized reset epoch yet"
                    )
                if (
                    current["reset_id"] != reset_info["reset_id"]
                    or current["reset_sha256"] != reset_info["sha256"]
                ):
                    raise BudgetLedgerError(
                        "the ledger reset identity disagrees with the authority"
                    )

    # -- charging -------------------------------------------------------

    def _check_blockers(self, document, refs) -> None:
        owner = refs.get("run_id")
        unfinished = [
            (record["reservation_id"], None)
            for record in document["imported_records"]
            if record["status"] == "reserved"
        ]
        unfinished += [
            (record["reservation_id"], record["refs"].get("run_id"))
            for record in document["reservations"]
            if record["status"] == "reserved"
        ]
        if not unfinished:
            return
        if owner is None:
            blocking = [identifier for identifier, _ in unfinished]
        else:
            blocking = [
                identifier
                for identifier, record_owner in unfinished
                if record_owner != owner
            ]
        if blocking:
            raise BudgetLedgerError(
                "unfinished reservations block unrelated new work until they are"
                " settled under their owning run_id: " + ", ".join(blocking)
            )

    def _check_caps(self, document, seconds, generations, storage_bytes) -> None:
        caps = document["opening"]["caps"]
        usage, _ = self._epoch_usage(document, document["current_epoch"])
        if usage["seconds"] + seconds > caps["stage3_max_seconds"]:
            raise BudgetLedgerError(
                "stage3 seconds exhausted: planned"
                f" {seconds} exceeds the remaining"
                f" {caps['stage3_max_seconds'] - usage['seconds']}"
            )
        if usage["seconds"] + seconds > caps["global_max_seconds"]:
            raise BudgetLedgerError(
                "global seconds exhausted: planned"
                f" {seconds} exceeds the remaining"
                f" {caps['global_max_seconds'] - usage['seconds']}"
            )
        if usage["generations"] + generations > caps["stage3_max_generations"]:
            raise BudgetLedgerError(
                "stage3 generations exhausted: planned"
                f" {generations} exceeds the remaining"
                f" {caps['stage3_max_generations'] - usage['generations']}"
            )
        if usage["storage_bytes"] + storage_bytes > caps["max_storage_bytes"]:
            raise BudgetLedgerError(
                "storage exhausted: planned"
                f" {storage_bytes} bytes exceeds the remaining"
                f" {caps['max_storage_bytes'] - usage['storage_bytes']}"
            )

    def _epoch_usage(self, document, index: int) -> tuple[dict, list[str]]:
        base = document["epochs"][index]["base_usage"]
        usage = {unit: base[unit] for unit in _UNITS}
        identifiers = []
        for record in document["reservations"]:
            if record["epoch"] != index:
                continue
            seconds, generations, storage = _charged(record)
            usage["seconds"] += seconds
            usage["generations"] += generations
            usage["storage_bytes"] += storage
            identifiers.append(record["reservation_id"])
        return usage, identifiers

    # -- persistence ----------------------------------------------------

    @contextmanager
    def _lock(self, *, exclusive: bool):
        lock_path = Path(str(self._path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _read_document(self) -> dict:
        try:
            raw = self._path.read_bytes()
        except FileNotFoundError as error:
            raise BudgetLedgerError("the budget ledger is missing; refusing") from error
        except OSError as error:
            raise BudgetLedgerError(
                f"the budget ledger is unreadable: {error}"
            ) from error
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise BudgetLedgerError(
                f"the budget ledger is corrupt; refusing: {error}"
            ) from error
        if not isinstance(document, dict):
            raise BudgetLedgerError("the budget ledger must be a JSON object")
        return document

    def _write(self, document: dict) -> None:
        document["integrity_sha256"] = _integrity(document)
        payload = (
            json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        directory = os.open(self._path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _mutate(self, change: Callable[[dict], dict]) -> dict:
        with self._lock(exclusive=True):
            opening_info, reset_info = self._authorities()
            document = self._read_document()
            self._revalidate(document, opening_info, reset_info)
            updated = change(copy.deepcopy(document))
            if updated == document:
                return document
            updated["revision"] = document["revision"] + 1
            self._revalidate(
                updated,
                opening_info,
                reset_info,
                verify_integrity=False,
            )
            self._write(updated)
            return updated

    # -- documents ------------------------------------------------------

    def _epoch(self, index, mode, base_usage, opening_info, *, reset_info=None):
        entry = {
            "index": index,
            "mode": mode,
            "reset_id": "",
            "reset_sha256": "",
            "reset_path": "",
            "source_opening_sha256": opening_info["sha256"],
            "base_usage": dict(base_usage),
            "started_at": _now(),
        }
        if reset_info is not None:
            entry["reset_id"] = reset_info["reset_id"]
            entry["reset_sha256"] = reset_info["sha256"]
            entry["reset_path"] = str(self._reset_path)
        return entry

    def _new_document(self, opening_info, reset_info) -> dict:
        epochs = [self._epoch(0, EPOCH_OPENING, opening_info["usage"], opening_info)]
        current = 0
        if reset_info is not None:
            epochs.append(
                self._epoch(
                    len(epochs),
                    EPOCH_RESET,
                    {
                        "seconds": 0,
                        "generations": 0,
                        "storage_bytes": opening_info["usage"]["storage_bytes"],
                    },
                    opening_info,
                    reset_info=reset_info,
                )
            )
            current = len(epochs) - 1
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": 1,
            "opening": {
                "path": str(self._opening_path),
                "sha256": opening_info["sha256"],
                "bytes": opening_info["bytes"],
                "caps": dict(opening_info["caps"]),
                "reservation_count": len(opening_info["records"]),
                "derived_usage": dict(opening_info["usage"]),
                "registered_versions": copy.deepcopy(opening_info["versions"]),
                "unfinished_reservation_ids": list(opening_info["unfinished"]),
            },
            "imported_records": copy.deepcopy(opening_info["records"]),
            "epochs": epochs,
            "current_epoch": current,
            "prompt_versions": copy.deepcopy(opening_info["versions"]),
            "reservations": [],
        }

    def _advance_epoch(self, document, opening_info, reset_info) -> None:
        if any(entry["mode"] == EPOCH_RESET for entry in document["epochs"]):
            raise BudgetLedgerError("the authorized reset may be applied exactly once")
        storage_base = sum(
            entry["base_usage"]["storage_bytes"] for entry in document["epochs"]
        )
        for record in document["reservations"]:
            storage_base += _charged(record)[2]
        document["epochs"].append(
            self._epoch(
                len(document["epochs"]),
                EPOCH_RESET,
                {"seconds": 0, "generations": 0, "storage_bytes": storage_base},
                opening_info,
                reset_info=reset_info,
            )
        )
        document["current_epoch"] = len(document["epochs"]) - 1
        document["revision"] += 1
        self._revalidate(document, opening_info, reset_info, verify_integrity=False)
        self._write(document)
        self._revalidate(document, opening_info, reset_info)

    def _advance_revision(self, document, opening_info, reset_info):
        """Exactly once, at the recorded predecessor; never refund retained storage."""
        info = self._revision_info
        if self._path.read_bytes() != info["predecessor_raw"]:
            raise BudgetLedgerError("revision predecessor ledger changed")
        self._check_blockers(document, {})
        if document != info["predecessor"]:
            raise BudgetLedgerError("revision predecessor differs from its snapshot")
        old_usage, _ = self._epoch_usage(document, document["current_epoch"])
        document["epochs"].append(
            dict(
                index=len(document["epochs"]),
                mode=EPOCH_REVISION,
                reset_id=info["body"]["reset_id"],
                reset_sha256=info["sha256"],
                reset_path=str(self._revision_path),
                source_opening_sha256=opening_info["sha256"],
                base_usage=dict(
                    seconds=0, generations=0, storage_bytes=old_usage["storage_bytes"]
                ),
                started_at=_now(),
            )
        )
        document["current_epoch"] = len(document["epochs"]) - 1
        document["revision"] += 1
        self._revalidate(document, opening_info, reset_info, verify_integrity=False)
        self._write(document)

    def _snapshot_document(self, document, opening_info) -> dict:
        views = []
        for entry in document["epochs"]:
            usage, identifiers = self._epoch_usage(document, entry["index"])
            base = entry["base_usage"]
            views.append(
                {
                    "index": entry["index"],
                    "mode": entry["mode"],
                    "reset_id": entry["reset_id"],
                    "reset_sha256": entry["reset_sha256"],
                    "base_usage": dict(base),
                    "new_charges": {unit: usage[unit] - base[unit] for unit in _UNITS},
                    "usage": usage,
                    "reservation_ids": identifiers,
                    "started_at": entry["started_at"],
                }
            )
        current = document["current_epoch"]
        active = views[current]["usage"]
        caps = opening_info["caps"]
        versions = document["prompt_versions"]
        unfinished = [
            record["reservation_id"]
            for record in document["reservations"]
            if record["status"] == "reserved"
        ] + list(opening_info["unfinished"])
        remaining = {
            "seconds": caps["stage3_max_seconds"] - active["seconds"],
            "generations": caps["stage3_max_generations"] - active["generations"],
            "storage_bytes": caps["max_storage_bytes"] - active["storage_bytes"],
            "global_seconds": caps["global_max_seconds"] - active["seconds"],
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "path": str(self._path),
            "epoch": {
                "index": views[current]["index"],
                "mode": views[current]["mode"],
                "reset_id": views[current]["reset_id"],
                "reset_sha256": views[current]["reset_sha256"],
            },
            "epochs": copy.deepcopy(views),
            "history": copy.deepcopy(views[:current]),
            "caps": {
                **caps,
                "max_assistant_prompt_versions": self._effective_version_limit(
                    document, "assistant_prompt"
                ),
            },
            "imported": {
                "sha256": opening_info["sha256"],
                "reservation_count": len(opening_info["records"]),
                "usage": dict(opening_info["usage"]),
                "totals_verified": True,
                "unfinished_reservation_ids": list(opening_info["unfinished"]),
            },
            "usage": dict(active),
            "remaining": dict(remaining),
            "remaining_seconds": remaining["seconds"],
            "remaining_generations": remaining["generations"],
            "remaining_storage_bytes": remaining["storage_bytes"],
            "remaining_global_seconds": remaining["global_seconds"],
            "storage_bytes_used": active["storage_bytes"],
            "caps_exceeded": any(value < 0 for value in remaining.values()),
            "prompt_versions": copy.deepcopy(versions),
            "version_remaining": {
                kind: self._effective_version_limit(document, kind)
                - len(versions[kind])
                for kind in VERSION_KINDS
            },
            "reservations": copy.deepcopy(document["reservations"]),
            "unfinished_reservation_ids": unfinished,
        }
