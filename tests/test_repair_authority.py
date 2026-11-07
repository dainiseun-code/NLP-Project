"""Behavioral tests for the code-pinned Stage 6 repair-series authority.

Derived from ``docs/stage6-repair-series-contract.md`` *before* the
implementation was written. The real authority and round plans are never read;
every authority is a fictional document under ``tmp_path`` whose digest is bound
to ``repair_authority.AUTHORITY_SHA256`` via ``monkeypatch`` (the production
constant is a placeholder the primary fills after handoff). No model, tokenizer,
Torch/MLX import, subprocess, network use or canonical budget is touched.

The pinned invariants under test:

* the authority digest is code-pinned, so any byte change fails closed, and the
  authority must name a relative project root it cannot escape;
* fixed scope (100 calls, five rounds, twenty calls per round, cap 5→10,
  restart waiver) and a ten-case roster of five successes/five failures;
* every round plan is hash-bound, holds twenty contiguous slots ``00``..``19``
  with pair-ordered control/candidate alternation that flips by round, binds the
  frozen roster/condition and derives its candidate version id from its refs;
* a candidate registration preserves the first five versions by canonical hash,
  numbers rounds by ``len(entries) - 4``, admits at most ten versions and adds
  exactly three ledger refs to the plan's base refs;
* slot refs require the series/round/run ids, the actual round-plan hash, a known
  slot and the exact messages SHA, bounded to ``0 < seconds <= 120`` and one
  generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from messaging_nlp import repair_authority

# Pinned contract values, repeated here independently of the module.
SERIES_ID = "contract-repair-series-20261028"
AUTHORITY_NAME = "stage6-repair-series-authorization-20261028.json"
DIRECTORY = "runs/stage6-repair-series-20261028"

ROSTER_CASES = tuple(f"fictional-case-{index:02d}" for index in range(10))


def fictional_roster():
    strata = ["success"] * 5 + ["failure"] * 5
    return [
        {"message_id": case, "stratum": stratum}
        for case, stratum in zip(ROSTER_CASES, strata, strict=True)
    ]


def fictional_authority(**overrides):
    """A valid, deterministic fictional repair-series authority document."""
    document = {
        "series_id": SERIES_ID,
        "directory": DIRECTORY,
        "project_root": "..",
        "max_calls": 100,
        "max_rounds": 5,
        "calls_per_round": 20,
        "assistant_version_cap": {"previous": 5, "approved": 10},
        "restart_waived": True,
        "prior_assistant_versions_sha256": "0" * 64,
        "roster": fictional_roster(),
    }
    document.update(overrides)
    return document


def write_authority(project, document) -> tuple[Path, str]:
    """Write ``document`` to ``project/docs`` and return (path, sha256)."""
    docs = Path(project) / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / AUTHORITY_NAME
    raw = (json.dumps(document, sort_keys=True, indent=1) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def sealed_authority(project, monkeypatch, document):
    path, sha = write_authority(project, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    return repair_authority.load_authority(path)


def write_plan(project, round_number, plan) -> tuple[Path, str]:
    """Write ``plan`` to ``project/DIRECTORY/rounds/NN`` and return (path, sha256)."""
    plan_dir = Path(project) / DIRECTORY / "rounds" / f"{round_number:02d}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    path = plan_dir / "plan.json"
    raw = (json.dumps(plan, sort_keys=True, indent=1) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def fictional_plan(body, round_number):
    """A valid round plan consistent with ``body``'s roster and round number."""
    refs = {"base_identity_sha256": str(round_number) * 64}
    prefix = hashlib.sha256(repair_authority.canonical(refs)).hexdigest()[:16]
    slots = []
    for index in range(20):
        pair = index // 2
        position = index % 2
        start = (pair + round_number - 1) % 2
        variant = "control" if position == start else "candidate"
        slots.append(
            {
                "slot": f"{index:02d}",
                "pair": pair,
                "variant": variant,
                "request_id": body["roster"][pair]["message_id"],
                "condition": "A+B",
                "messages": {"sha256": slot_messages_sha256(round_number, index)},
            }
        )
    return {
        "series_id": SERIES_ID,
        "round": round_number,
        "run_id": f"{SERIES_ID}-r{round_number:02d}",
        "slots": slots,
        "candidate_version": {
            "version_id": f"repair-candidate-r{round_number:02d}-{prefix}",
            "refs": refs,
        },
    }


def slot_messages_sha256(round_number: int, index: int) -> str:
    return hashlib.sha256(
        f"fictional-message-r{round_number}-{index:02d}".encode()
    ).hexdigest()


def five_entries():
    """Five version records standing in for the five earlier assistant prompts."""
    return [
        {
            "version_id": f"assistant_prompt-{index}",
            "registered_at": f"2026-10-2{index}T00:00:00+00:00",
            "refs": {"template_identity_sha256": str(index) * 64},
        }
        for index in range(5)
    ]


def prior_sha():
    return repair_authority.digest(repair_authority.canonical(five_entries()))


def register_refs(plan, round_number, plan_sha):
    return {
        **plan["candidate_version"]["refs"],
        "repair_series_id": SERIES_ID,
        "round": str(round_number),
        "round_plan_sha256": plan_sha,
    }


def slot_refs(plan, round_number, plan_sha, slot="00", messages_sha256=None):
    if messages_sha256 is None:
        found = [s for s in plan["slots"] if s["slot"] == slot]
        messages_sha256 = found[0]["messages"]["sha256"] if found else "0" * 64
    return {
        "repair_series_id": SERIES_ID,
        "round": str(round_number),
        "run_id": f"{SERIES_ID}-r{round_number:02d}",
        "round_plan_sha256": plan_sha,
        "diagnostic_slot": slot,
        "messages_sha256": messages_sha256,
    }


def sealed_body(tmp_path, monkeypatch, **overrides):
    return sealed_authority(
        tmp_path,
        monkeypatch,
        fictional_authority(prior_assistant_versions_sha256=prior_sha(), **overrides),
    )


def sealed_round(tmp_path, monkeypatch, round_number, **overrides):
    """Return (body, plan, plan_sha) for a sealed authority and a written plan."""
    body = sealed_body(tmp_path, monkeypatch, **overrides)
    plan = fictional_plan(body, round_number)
    _, plan_sha = write_plan(tmp_path, round_number, plan)
    return body, plan, plan_sha


def build_chain(body, project, rounds):
    """Write plans for rounds 1..N and return the assembled registry entries."""
    entries = five_entries()
    for round_number in range(1, rounds + 1):
        plan = fictional_plan(body, round_number)
        _, plan_sha = write_plan(project, round_number, plan)
        entries.append(
            {
                "version_id": plan["candidate_version"]["version_id"],
                "registered_at": f"2026-10-2{round_number}T00:00:00+00:00",
                "refs": register_refs(plan, round_number, plan_sha),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Digests and the code-pinned authority digest
# ---------------------------------------------------------------------------


def test_digest_and_canonical_are_stable_and_sorted():
    assert repair_authority.digest(b"abc") == hashlib.sha256(b"abc").hexdigest()
    a = repair_authority.canonical({"b": 1, "a": 2})
    b = repair_authority.canonical({"a": 2, "b": 1})
    assert a == b == b'{"a":2,"b":1}'


def test_load_authority_accepts_the_exact_frozen_body(tmp_path, monkeypatch):
    body = sealed_authority(tmp_path, monkeypatch, fictional_authority())
    assert body["series_id"] == SERIES_ID
    assert body["directory"] == DIRECTORY
    assert body["max_calls"] == 100
    assert body["max_rounds"] == 5
    assert body["calls_per_round"] == 20
    assert body["assistant_version_cap"] == {"previous": 5, "approved": 10}
    assert body["restart_waived"] is True
    assert len(body["roster"]) == 10
    assert body["_root"] == (tmp_path / "docs" / "..").resolve()


def test_load_authority_rejects_any_byte_change(tmp_path, monkeypatch):
    _, sha = write_authority(tmp_path, fictional_authority())
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    changed = fictional_authority()
    changed["max_calls"] = 101
    path = tmp_path / "docs" / AUTHORITY_NAME
    path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="code-pinned approval"):
        repair_authority.load_authority(path)


def test_load_authority_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="unreadable"):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


def test_load_authority_rejects_a_symlink(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = docs / AUTHORITY_NAME
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        repair_authority.load_authority(link)


def test_load_authority_rejects_malformed_json(tmp_path, monkeypatch):
    raw = b"{not json"
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / AUTHORITY_NAME
    path.write_bytes(raw)
    monkeypatch.setattr(
        repair_authority, "AUTHORITY_SHA256", hashlib.sha256(raw).hexdigest()
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        repair_authority.load_authority(path)


def test_load_authority_rejects_an_absolute_project_root(tmp_path, monkeypatch):
    document = fictional_authority(project_root="/tmp")
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="relative project root"):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


def test_load_authority_rejects_an_escaping_project_root(tmp_path, monkeypatch):
    (tmp_path / "docs" / "elsewhere").mkdir(parents=True)
    document = fictional_authority(project_root="elsewhere")
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="escapes"):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


@pytest.mark.parametrize(
    "field", ["series_id", "directory", "prior_assistant_versions_sha256"]
)
def test_load_authority_rejects_missing_fields(tmp_path, monkeypatch, field):
    document = fictional_authority()
    del document[field]
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises((KeyError, ValueError)):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(series_id="other-series"),
        lambda d: d.update(directory="runs/elsewhere"),
        lambda d: d.update(max_calls=99),
        lambda d: d.update(max_calls=101),
        lambda d: d.update(max_rounds=4),
        lambda d: d.update(max_rounds=6),
        lambda d: d.update(calls_per_round=19),
        lambda d: d.update(calls_per_round=21),
        lambda d: d.update(assistant_version_cap={"previous": 5, "approved": 11}),
        lambda d: d.update(assistant_version_cap={"previous": 4, "approved": 10}),
        lambda d: d.update(restart_waived=False),
        lambda d: d.update(restart_waived=1),
    ],
)
def test_load_authority_rejects_scope_changes(tmp_path, monkeypatch, mutate):
    document = fictional_authority()
    mutate(document)
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="scope changed"):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


def test_load_authority_rejects_a_shortened_roster(tmp_path, monkeypatch):
    document = fictional_authority()
    document["roster"] = document["roster"][:9]
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="ten cases"):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


def test_load_authority_rejects_duplicate_roster_message_ids(tmp_path, monkeypatch):
    document = fictional_authority()
    document["roster"][9]["message_id"] = document["roster"][0]["message_id"]
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="unique"):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


def test_load_authority_rejects_wrong_strata(tmp_path, monkeypatch):
    document = fictional_authority()
    document["roster"][5]["stratum"] = "success"
    document["roster"][6]["stratum"] = "failure"
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="five successes and five failures"):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


def test_load_authority_rejects_unknown_stratum(tmp_path, monkeypatch):
    document = fictional_authority()
    document["roster"][0]["stratum"] = "technical-failure"
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(repair_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="success or failure"):
        repair_authority.load_authority(tmp_path / "docs" / AUTHORITY_NAME)


# ---------------------------------------------------------------------------
# Round plans
# ---------------------------------------------------------------------------


def test_round_plan_accepts_every_round(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    for round_number in range(1, 6):
        plan = fictional_plan(body, round_number)
        _, plan_sha = write_plan(tmp_path, round_number, plan)
        assert repair_authority.round_plan(body, round_number, plan_sha) == plan


@pytest.mark.parametrize("round_number", [0, 6, 100, True, 1.0, "1"])
def test_round_plan_rejects_wrong_round_number(tmp_path, monkeypatch, round_number):
    body = sealed_body(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="integer from 1 to 5"):
        repair_authority.round_plan(body, round_number, "0" * 64)


def test_round_plan_rejects_a_missing_plan(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unreadable"):
        repair_authority.round_plan(body, 1, "0" * 64)


def test_round_plan_rejects_a_hash_mismatch(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    fictional_plan(body, 1)
    write_plan(tmp_path, 1, fictional_plan(body, 1))
    with pytest.raises(ValueError, match="hash-bound approval"):
        repair_authority.round_plan(body, 1, "f" * 64)


def test_round_plan_rejects_malformed_json(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan_dir = tmp_path / DIRECTORY / "rounds" / "01"
    plan_dir.mkdir(parents=True)
    raw = b"{not json"
    (plan_dir / "plan.json").write_bytes(raw)
    with pytest.raises(ValueError, match="not valid JSON"):
        repair_authority.round_plan(body, 1, hashlib.sha256(raw).hexdigest())


def test_round_plan_rejects_a_symlink_escape(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan_dir = tmp_path / DIRECTORY / "rounds" / "01"
    plan_dir.mkdir(parents=True)
    external = tmp_path / "external-plan.json"
    external.write_text("{}")
    (plan_dir / "plan.json").symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        repair_authority.round_plan(body, 1, "0" * 64)


def test_round_plan_rejects_an_embedded_own_hash(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    plan["round_plan_sha256"] = "0" * 64
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="own hash"):
        repair_authority.round_plan(body, 1, plan_sha)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.update(series_id="other"), "series id changed"),
        (lambda p: p.update(round=2), "number changed"),
        (lambda p: p.update(run_id="other-run"), "run id changed"),
    ],
)
def test_round_plan_rejects_foreign_identity(tmp_path, monkeypatch, mutate, match):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    mutate(plan)
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match=match):
        repair_authority.round_plan(body, 1, plan_sha)


def test_round_plan_rejects_a_21st_slot(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    plan["slots"].append(
        {
            "slot": "20",
            "pair": 10,
            "variant": "control",
            "request_id": body["roster"][0]["message_id"],
            "condition": "A+B",
            "messages": {"sha256": "0" * 64},
        }
    )
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="twenty slots"):
        repair_authority.round_plan(body, 1, plan_sha)


def test_round_plan_rejects_wrong_pair_order(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    plan["slots"][2]["pair"] = 0
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="pair order changed"):
        repair_authority.round_plan(body, 1, plan_sha)


def test_round_plan_rejects_wrong_alternation(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    plan["slots"][1]["variant"] = "control"  # pair 0 must be control, candidate
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="alternation changed"):
        repair_authority.round_plan(body, 1, plan_sha)


def test_round_plan_rejects_wrong_request_id(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    plan["slots"][0]["request_id"] = "someone-else"
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="frozen roster"):
        repair_authority.round_plan(body, 1, plan_sha)


def test_round_plan_rejects_wrong_condition(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    plan["slots"][0]["condition"] = "B"
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="condition changed"):
        repair_authority.round_plan(body, 1, plan_sha)


def test_round_plan_rejects_a_bad_candidate_prefix(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    plan["candidate_version"]["version_id"] = "repair-candidate-r01-not-the-digest"
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="disagrees with its refs"):
        repair_authority.round_plan(body, 1, plan_sha)


def test_round_plan_rejects_refs_embedding_the_plan_hash(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    plan = fictional_plan(body, 1)
    plan["candidate_version"]["refs"]["round_plan_sha256"] = "0" * 64
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="must not embed the plan hash"):
        repair_authority.round_plan(body, 1, plan_sha)


# ---------------------------------------------------------------------------
# Candidate version registration
# ---------------------------------------------------------------------------


def test_check_version_accepts_the_first_round_candidate(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    repair_authority.check_version(
        body,
        five_entries(),
        plan["candidate_version"]["version_id"],
        register_refs(plan, 1, plan_sha),
    )


def test_check_version_accepts_all_five_rounds(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    entries = five_entries()
    for round_number in range(1, 6):
        plan = fictional_plan(body, round_number)
        _, plan_sha = write_plan(tmp_path, round_number, plan)
        repair_authority.check_version(
            body,
            entries,
            plan["candidate_version"]["version_id"],
            register_refs(plan, round_number, plan_sha),
        )
        entries.append(
            {
                "version_id": plan["candidate_version"]["version_id"],
                "registered_at": f"2026-10-2{round_number}T00:00:00+00:00",
                "refs": register_refs(plan, round_number, plan_sha),
            }
        )


def test_check_version_rejects_altered_first_five(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    altered = five_entries()
    altered[0] = dict(altered[0], version_id="assistant_prompt-rewritten")
    with pytest.raises(ValueError, match="preserve all five"):
        repair_authority.check_version(
            body,
            altered,
            plan["candidate_version"]["version_id"],
            register_refs(plan, 1, plan_sha),
        )


def test_check_version_rejects_a_wrong_version_id(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    with pytest.raises(ValueError, match="explicitly authorized"):
        repair_authority.check_version(
            body,
            five_entries(),
            "repair-candidate-r01-some-other",
            register_refs(plan, 1, plan_sha),
        )


def test_check_version_rejects_wrong_refs(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    refs = register_refs(plan, 1, plan_sha)
    refs["base_identity_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="explicitly authorized"):
        repair_authority.check_version(
            body, five_entries(), plan["candidate_version"]["version_id"], refs
        )


def test_check_version_rejects_a_missing_added_ref(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    refs = register_refs(plan, 1, plan_sha)
    del refs["round"]
    with pytest.raises(ValueError, match="explicitly authorized"):
        repair_authority.check_version(
            body, five_entries(), plan["candidate_version"]["version_id"], refs
        )


def test_check_version_rejects_a_wrong_round_plan_hash(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    refs = register_refs(plan, 1, plan_sha)
    refs["round_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash-bound approval"):
        repair_authority.check_version(
            body, five_entries(), plan["candidate_version"]["version_id"], refs
        )


def test_check_version_rejects_a_sixth_candidate(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    entries = build_chain(body, tmp_path, 5)  # five prior + five rounds = ten
    plan = fictional_plan(body, 1)
    _, plan_sha = write_plan(tmp_path, 1, plan)
    with pytest.raises(ValueError, match="exhausted at ten"):
        repair_authority.check_version(
            body,
            entries,
            plan["candidate_version"]["version_id"],
            register_refs(plan, 1, plan_sha),
        )


# ---------------------------------------------------------------------------
# Already-registered round validation
# ---------------------------------------------------------------------------


def test_check_registered_versions_accepts_the_full_chain(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    entries = build_chain(body, tmp_path, 5)
    repair_authority.check_registered_versions(body, entries)


def test_check_registered_versions_accepts_no_new_rounds(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    repair_authority.check_registered_versions(body, five_entries())


def test_check_registered_versions_rejects_a_sixth_beyond_ten(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    entries = build_chain(body, tmp_path, 5) + [
        {
            "version_id": "repair-candidate-r06-extra",
            "refs": {"round_plan_sha256": "0" * 64},
        }
    ]
    with pytest.raises(ValueError, match="exhausted at ten"):
        repair_authority.check_registered_versions(body, entries)


def test_check_registered_versions_rejects_a_collision(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    entries = build_chain(body, tmp_path, 1)
    entries.append(dict(entries[5]))
    with pytest.raises(ValueError, match="re-register"):
        repair_authority.check_registered_versions(body, entries)


def test_check_registered_versions_rejects_a_tampered_round(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    entries = build_chain(body, tmp_path, 2)
    entries[5]["version_id"] = "repair-candidate-r01-tampered"
    with pytest.raises(ValueError, match="immutable plan"):
        repair_authority.check_registered_versions(body, entries)


# ---------------------------------------------------------------------------
# Frozen slot refs and the per-call bounds
# ---------------------------------------------------------------------------


def test_check_slot_accepts_every_frozen_slot(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    for slot in plan["slots"]:
        repair_authority.check_slot(
            body, slot_refs(plan, 1, plan_sha, slot=slot["slot"]), 120, 1
        )


def test_check_slot_accepts_slots_across_rounds(tmp_path, monkeypatch):
    body = sealed_body(tmp_path, monkeypatch)
    for round_number in range(1, 6):
        plan = fictional_plan(body, round_number)
        _, plan_sha = write_plan(tmp_path, round_number, plan)
        for slot in plan["slots"]:
            repair_authority.check_slot(
                body, slot_refs(plan, round_number, plan_sha, slot=slot["slot"]), 120, 1
            )


def test_check_slot_rejects_a_sixth_round(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    refs = slot_refs(plan, 1, plan_sha)
    refs["round"] = "6"
    refs["run_id"] = f"{SERIES_ID}-r06"
    with pytest.raises(ValueError, match="frozen repair series slots"):
        repair_authority.check_slot(body, refs, 120, 1)


def test_check_slot_rejects_a_101st_slot(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    with pytest.raises(ValueError, match="frozen repair series slots"):
        repair_authority.check_slot(
            body, slot_refs(plan, 1, plan_sha, slot="20"), 120, 1
        )


def test_check_slot_rejects_a_foreign_messages_sha256(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    refs = slot_refs(plan, 1, plan_sha, slot="00", messages_sha256="0" * 64)
    with pytest.raises(ValueError, match="frozen repair series slots"):
        repair_authority.check_slot(body, refs, 120, 1)


def test_check_slot_rejects_a_wrong_run_id(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    refs = slot_refs(plan, 1, plan_sha)
    refs["run_id"] = f"{SERIES_ID}-r02"
    with pytest.raises(ValueError, match="frozen repair series slots"):
        repair_authority.check_slot(body, refs, 120, 1)


def test_check_slot_rejects_a_wrong_series_id(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    refs = slot_refs(plan, 1, plan_sha)
    refs["repair_series_id"] = "other-series"
    with pytest.raises(ValueError, match="frozen repair series slots"):
        repair_authority.check_slot(body, refs, 120, 1)


def test_check_slot_rejects_a_wrong_plan_hash(tmp_path, monkeypatch):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    refs = slot_refs(plan, 1, plan_sha)
    refs["round_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash-bound approval"):
        repair_authority.check_slot(body, refs, 120, 1)


@pytest.mark.parametrize("seconds", [0, 121, 200, -1])
def test_check_slot_rejects_seconds_outside_bounds(tmp_path, monkeypatch, seconds):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    with pytest.raises(ValueError, match="frozen repair series slots"):
        repair_authority.check_slot(body, slot_refs(plan, 1, plan_sha), seconds, 1)


@pytest.mark.parametrize("generations", [0, 2])
def test_check_slot_rejects_wrong_generation_count(tmp_path, monkeypatch, generations):
    body, plan, plan_sha = sealed_round(tmp_path, monkeypatch, 1)
    with pytest.raises(ValueError, match="frozen repair series slots"):
        repair_authority.check_slot(
            body, slot_refs(plan, 1, plan_sha), 120, generations
        )
