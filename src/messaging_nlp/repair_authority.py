"""Code-pinned authority for the five-round controlled repair series.

Implements only the confirmed narrow authority contract in
``docs/stage6-repair-series-contract.md``: 100 additional Qwen calls, at most
five 20-call batches, prompt-version cap 5→10 for up to five candidate
identities, a fixed ten-case roster (five archived successes, five failures) and
the diagnostic-only restart waiver. This module is standard library only so the
runtime coordinator can import it without optional model packages. It never
reads or writes the canonical budget ledger; the primary integrates these pure
helpers into the ledger's registration and reservation paths exactly as
``diagnostic_authority`` is integrated.

The authority document (hash-pinned via ``AUTHORITY_SHA256``, which the primary
records after handoff) fixes the first five assistant-prompt registrations, the
roster and the fixed scope. Each round's ``plan.json`` is immutable and
hash-bound; a plan must not embed its own hash (that would be a cycle — the
register refs add ``round_plan_sha256`` instead).

Conventions the primary must match when authoring the authority and round plans:

* ``project_root`` is a relative path from the authority file's parent (``docs/``)
  to the project root (``".."``);
* a candidate ``version_id`` is ``repair-candidate-r{round:02d}-`` followed by
  the first 16 hex characters of ``sha256(canonical(refs))``;
* a slot's ``messages.sha256`` is the SHA-256 of its frozen input messages;
* register refs are ``candidate_version.refs`` plus ``repair_series_id``,
  ``round`` (decimal string) and ``round_plan_sha256``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SERIES_ID = "contract-repair-series-20261028"
AUTHORITY_NAME = "stage6-repair-series-authorization-20261028.json"
# Exact user-authorized series document; retain it and its bound manifest.
AUTHORITY_SHA256 = "fc8cc4e0b17e1e689fb849aeb2627f5b283d995b1a4775b2634a098817f3738f"
DIRECTORY = "runs/stage6-repair-series-20261028"

_PRIOR = {"previous": 5, "approved": 10}
_ROUNDS = ("1", "2", "3", "4", "5")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex64(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _run_id(round_number: int) -> str:
    return f"{SERIES_ID}-r{round_number:02d}"


def load_authority(path: Path) -> dict:
    path = Path(path)
    if path.is_symlink():
        raise ValueError("repair series authority cannot be a symlink")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"repair series authority is unreadable: {error}") from error
    if digest(raw) != AUTHORITY_SHA256:
        raise ValueError(
            "repair series authority differs from its code-pinned approval"
        )
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(
            f"repair series authority is not valid JSON: {error}"
        ) from error
    if not isinstance(body, dict):
        raise ValueError("repair series authority must be a JSON object")

    project_root = body.get("project_root")
    if (
        not isinstance(project_root, str)
        or not project_root.strip()
        or Path(project_root).is_absolute()
    ):
        raise ValueError("repair series authority must name a relative project root")
    try:
        root = (path.parent / project_root).resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"repair series authority project root is invalid: {error}"
        ) from error
    if not path.resolve().is_relative_to(root):
        raise ValueError("repair series authority escapes its explicit project root")

    if (
        body.get("series_id") != SERIES_ID
        or body.get("directory") != DIRECTORY
        or body.get("max_calls") != 100
        or body.get("max_rounds") != 5
        or body.get("calls_per_round") != 20
        or body.get("assistant_version_cap") != _PRIOR
        or body.get("restart_waived") is not True
        or not _is_hex64(body.get("prior_assistant_versions_sha256"))
    ):
        raise ValueError("repair series authority scope changed")

    roster = body.get("roster")
    if not isinstance(roster, list) or len(roster) != 10:
        raise ValueError("repair series authority roster must hold ten cases")
    message_ids = []
    strata = {"success": 0, "failure": 0}
    for entry in roster:
        if not isinstance(entry, dict):
            raise ValueError("repair series roster entries must be objects")
        message_id = entry.get("message_id")
        stratum = entry.get("stratum")
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("repair series roster entries need a message id")
        if stratum not in strata:
            raise ValueError("repair series roster stratum must be success or failure")
        message_ids.append(message_id)
        strata[stratum] += 1
    if len(set(message_ids)) != len(message_ids):
        raise ValueError("repair series roster message ids must be unique")
    if strata != {"success": 5, "failure": 5}:
        raise ValueError(
            "repair series roster must hold five successes and five failures"
        )

    if "inputs_manifest" in body:
        reference = body["inputs_manifest"]
        manifest_path = root / reference["path"]
        if manifest_path.is_symlink() or not manifest_path.resolve().is_relative_to(
            root
        ):
            raise ValueError("repair inputs manifest escapes its explicit root")
        manifest_raw = manifest_path.read_bytes()
        if (
            len(manifest_raw) != reference["bytes"]
            or digest(manifest_raw) != reference["sha256"]
        ):
            raise ValueError("repair inputs manifest differs from its authority")
        manifest = json.loads(manifest_raw)
        if [(r["message_id"], r["stratum"]) for r in manifest["roster"]] != [
            (r["message_id"], r["stratum"]) for r in roster
        ]:
            raise ValueError("repair inputs manifest changes the frozen strata")
        if (set(manifest) - {"roster"}) & set(body):
            raise ValueError("repair inputs manifest cannot override authority fields")
        body.update(manifest)

    body["_root"] = root  # private helper only; not part of the canonical authority
    return body


def round_plan(body: dict, round_number, expected_sha256) -> dict:
    if not _is_int(round_number) or not 1 <= round_number <= 5:
        raise ValueError("round number must be an integer from 1 to 5")
    if not _is_hex64(expected_sha256):
        raise ValueError("a round plan requires its 64-hex SHA-256")
    root = body.get("_root")
    if not isinstance(root, Path):
        raise ValueError("repair series authority was not loaded with its project root")
    plan_path = root / DIRECTORY / "rounds" / f"{round_number:02d}" / "plan.json"
    if plan_path.is_symlink() or not plan_path.resolve().is_relative_to(root):
        raise ValueError("round plan escapes its explicit project root")
    try:
        raw = plan_path.read_bytes()
    except OSError as error:
        raise ValueError(f"round plan is unreadable: {error}") from error
    if digest(raw) != expected_sha256:
        raise ValueError("round plan differs from its hash-bound approval")
    try:
        plan = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"round plan is not valid JSON: {error}") from error
    if not isinstance(plan, dict):
        raise ValueError("round plan must be a JSON object")

    if plan.get("series_id") != SERIES_ID:
        raise ValueError("round plan series id changed")
    if plan.get("round") != round_number:
        raise ValueError("round plan number changed")
    if plan.get("run_id") != _run_id(round_number):
        raise ValueError("round plan run id changed")
    if "round_plan_sha256" in plan:
        raise ValueError("round plan must not embed its own hash")

    slots = plan.get("slots")
    if not isinstance(slots, list) or len(slots) != 20:
        raise ValueError("round plan must hold twenty slots")
    if not all(isinstance(slot, dict) for slot in slots):
        raise ValueError("round plan slots must be objects")
    if [slot.get("slot") for slot in slots] != [f"{i:02d}" for i in range(20)]:
        raise ValueError("round plan slots must be the contiguous 00..19")
    roster = body["roster"]
    for index, slot in enumerate(slots):
        pair = index // 2
        position = index % 2
        start = (pair + round_number - 1) % 2
        expected_variant = "control" if position == start else "candidate"
        if slot.get("pair") != pair:
            raise ValueError("round plan pair order changed")
        if slot.get("variant") != expected_variant:
            raise ValueError("round plan control/candidate alternation changed")
        if slot.get("request_id") != roster[pair]["message_id"]:
            raise ValueError("round plan request differs from the frozen roster")
        if slot.get("condition") != "A+B":
            raise ValueError("round plan study condition changed")
        messages = slot.get("messages")
        if not isinstance(messages, dict) or not _is_hex64(messages.get("sha256")):
            raise ValueError("round plan slot messages need their frozen SHA-256")

    candidate = plan.get("candidate_version")
    if not isinstance(candidate, dict):
        raise ValueError("round plan candidate version missing")
    refs = candidate.get("refs")
    if not isinstance(refs, dict):
        raise ValueError("round plan candidate refs must be an object")
    if "round_plan_sha256" in refs:
        raise ValueError("round plan candidate refs must not embed the plan hash")
    prefix = digest(canonical(refs))[:16]
    expected_id = f"repair-candidate-r{round_number:02d}-{prefix}"
    if candidate.get("version_id") != expected_id:
        raise ValueError("round plan candidate version id disagrees with its refs")
    return plan


def check_version(body: dict, entries: list, version_id: str, refs: dict) -> None:
    if not isinstance(entries, list) or len(entries) < 5:
        raise ValueError("repair series must preserve the five earlier prompt versions")
    if digest(canonical(entries[:5])) != body["prior_assistant_versions_sha256"]:
        raise ValueError("repair series must preserve all five earlier prompt versions")
    if len(entries) >= 10:
        raise ValueError("repair series assistant versions are exhausted at ten")
    if not isinstance(refs, dict) or not _is_hex64(refs.get("round_plan_sha256")):
        raise ValueError("candidate registration must bind its round plan hash")
    round_number = len(entries) - 4
    plan = round_plan(body, round_number, refs["round_plan_sha256"])
    expected_refs = {
        **plan["candidate_version"]["refs"],
        "repair_series_id": SERIES_ID,
        "round": str(round_number),
        "round_plan_sha256": refs["round_plan_sha256"],
    }
    if version_id != plan["candidate_version"]["version_id"] or refs != expected_refs:
        raise ValueError("only the explicitly authorized round candidate is allowed")


def check_registered_versions(body: dict, entries: list) -> None:
    if not isinstance(entries, list) or len(entries) < 5:
        raise ValueError("repair series must preserve the five earlier prompt versions")
    if len(entries) > 10:
        raise ValueError("repair series assistant versions are exhausted at ten")
    if digest(canonical(entries[:5])) != body["prior_assistant_versions_sha256"]:
        raise ValueError("repair series must preserve all five earlier prompt versions")
    identifiers = [entry.get("version_id") for entry in entries]
    if any(not isinstance(name, str) or not name for name in identifiers):
        raise ValueError("repair series version records need their identities")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("repair series may not re-register a version identity")
    for position in range(5, len(entries)):
        round_number = position - 4
        entry = entries[position]
        if not isinstance(entry, dict):
            raise ValueError("repair series version records must be objects")
        entry_refs = entry.get("refs")
        if not isinstance(entry_refs, dict) or not _is_hex64(
            entry_refs.get("round_plan_sha256")
        ):
            raise ValueError("registered round must bind its round plan hash")
        plan = round_plan(body, round_number, entry_refs["round_plan_sha256"])
        expected_refs = {
            **plan["candidate_version"]["refs"],
            "repair_series_id": SERIES_ID,
            "round": str(round_number),
            "round_plan_sha256": entry_refs["round_plan_sha256"],
        }
        if (
            entry.get("version_id") != plan["candidate_version"]["version_id"]
            or entry_refs != expected_refs
        ):
            raise ValueError("registered round disagrees with its immutable plan")


def check_slot(body: dict, refs: dict, seconds: int, generations: int) -> None:
    if not isinstance(refs, dict):
        raise ValueError("slot refs must bind the repair series")
    if refs.get("repair_series_id") != SERIES_ID:
        raise ValueError("generation is outside the frozen repair series slots")
    round_str = refs.get("round")
    if round_str not in _ROUNDS:
        raise ValueError("generation is outside the frozen repair series slots")
    round_number = int(round_str)
    if refs.get("run_id") != _run_id(round_number):
        raise ValueError("generation is outside the frozen repair series slots")
    plan = round_plan(body, round_number, refs.get("round_plan_sha256"))
    slot = next(
        (s for s in plan["slots"] if s["slot"] == refs.get("diagnostic_slot")), None
    )
    if (
        slot is None
        or refs.get("messages_sha256") != slot["messages"]["sha256"]
        or not _is_int(seconds)
        or not 0 < seconds <= 120
        or not _is_int(generations)
        or generations != 1
    ):
        raise ValueError("generation is outside the frozen repair series slots")
