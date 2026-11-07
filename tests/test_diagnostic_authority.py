"""Behavioral tests for the code-pinned Stage 6 diagnostic authority.

Derived from ``docs/stage6-controlled-diagnostic-contract.md`` *before* the
implementation was written. The real authority is never read; every authority
is a fictional document under ``tmp_path`` whose digest is bound to
``diagnostic_authority.AUTHORITY_SHA256`` via ``monkeypatch`` (the production
constant is a placeholder the primary fills before the real run). No model,
tokenizer, Torch/MLX import, subprocess, network use or canonical budget is
touched.

The pinned invariants under test:

* the authority digest is code-pinned, so any byte change fails closed;
* exactly twenty unique contiguous slots ``00``..``19``;
* one exact fifth prompt version that preserves the SHA of the original four
  registry entries; a wrong fifth, an altered first four or a sixth fail;
* slot refs require the diagnostic id/run id, a known slot and the exact
  messages SHA, and are bounded to ``0 < seconds <= 120`` and one generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from messaging_nlp import diagnostic_authority

# Pinned contract values, repeated here independently of the module.
DIAGNOSTIC_ID = "contract-diagnostic-20261028"
DESTINATION = "runs/stage6-contract-diagnostic-20261028"
AUTHORITY_NAME = "stage6-diagnostic-authorization-20261028.json"
FIFTH_ID = "assistant_prompt-fictional-fifth"


def slot_messages_sha256(index: int) -> str:
    """The fictional messages SHA bound to slot ``index`` in the builder."""
    return hashlib.sha256(f"fictional-message-{index:02d}".encode()).hexdigest()


def fictional_authority(**overrides):
    """A valid, deterministic fictional diagnostic authority document."""
    document = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "destination": DESTINATION,
        "max_calls": 20,
        "restart_waived": True,
        "assistant_version_cap": {"previous": 4, "approved": 5},
        "prior_assistant_versions_sha256": "0" * 64,
        "fifth_version": {
            "version_id": FIFTH_ID,
            "refs": {"template_identity_sha256": "f" * 64},
        },
        "slots": [
            {
                "slot": f"{index:02d}",
                "messages": {"sha256": slot_messages_sha256(index)},
            }
            for index in range(20)
        ],
    }
    document.update(overrides)
    return document


def write_authority(directory, document) -> tuple[Path, str]:
    """Write ``document`` beside the reset authority and return (path, sha256)."""
    path = Path(directory) / AUTHORITY_NAME
    raw = (json.dumps(document, sort_keys=True, indent=1) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def sealed_authority(tmp_path, monkeypatch, document):
    path, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    return diagnostic_authority.load_authority(path)


def four_entries():
    """Four version records standing in for the four earlier assistant prompts."""
    return [
        {
            "version_id": f"assistant_prompt-{index}",
            "registered_at": f"2026-10-2{index}T00:00:00+00:00",
            "refs": {"template_identity_sha256": str(index) * 64},
        }
        for index in range(4)
    ]


def slot_refs(slot="00", messages_sha256=None):
    """Refs that satisfy ``check_slot`` for the fictional authority by default."""
    if messages_sha256 is None:
        messages_sha256 = slot_messages_sha256(int(slot))
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "run_id": DIAGNOSTIC_ID,
        "diagnostic_slot": slot,
        "messages_sha256": messages_sha256,
    }


# ---------------------------------------------------------------------------
# Digests and the code-pinned authority digest
# ---------------------------------------------------------------------------


def test_digest_and_canonical_are_stable_and_sorted():
    assert diagnostic_authority.digest(b"abc") == hashlib.sha256(b"abc").hexdigest()
    a = diagnostic_authority.canonical({"b": 1, "a": 2})
    b = diagnostic_authority.canonical({"a": 2, "b": 1})
    assert a == b == b'{"a":2,"b":1}'


def test_load_authority_accepts_the_exact_frozen_body(tmp_path, monkeypatch):
    body = sealed_authority(tmp_path, monkeypatch, fictional_authority())
    assert body["diagnostic_id"] == DIAGNOSTIC_ID
    assert body["destination"] == DESTINATION
    assert body["max_calls"] == 20
    assert body["restart_waived"] is True
    assert body["assistant_version_cap"] == {"previous": 4, "approved": 5}
    assert [slot["slot"] for slot in body["slots"]] == [f"{i:02d}" for i in range(20)]


def test_load_authority_rejects_any_byte_change(tmp_path, monkeypatch):
    _, sha = write_authority(tmp_path, fictional_authority())
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    changed = fictional_authority()
    changed["max_calls"] = 21
    path = tmp_path / AUTHORITY_NAME
    path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="code-pinned approval"):
        diagnostic_authority.load_authority(path)


def test_load_authority_rejects_a_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / AUTHORITY_NAME
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        diagnostic_authority.load_authority(link)


def test_load_authority_rejects_malformed_json(tmp_path, monkeypatch):
    raw = b"{not json"
    path = tmp_path / AUTHORITY_NAME
    path.write_bytes(raw)
    monkeypatch.setattr(
        diagnostic_authority, "AUTHORITY_SHA256", hashlib.sha256(raw).hexdigest()
    )
    with pytest.raises(ValueError):
        diagnostic_authority.load_authority(path)


def test_load_authority_rejects_missing_fields(tmp_path, monkeypatch):
    document = fictional_authority()
    del document["destination"]
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises((KeyError, ValueError)):
        diagnostic_authority.load_authority(tmp_path / AUTHORITY_NAME)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(diagnostic_id="other-diagnostic"),
        lambda d: d.update(destination="elsewhere"),
        lambda d: d.update(max_calls=19),
        lambda d: d.update(max_calls=21),
        lambda d: d.update(restart_waived=False),
        lambda d: d.update(restart_waived=1),
        lambda d: d.update(assistant_version_cap={"previous": 4, "approved": 6}),
        lambda d: d.update(assistant_version_cap={"previous": 3, "approved": 5}),
    ],
)
def test_load_authority_rejects_scope_changes(tmp_path, monkeypatch, mutate):
    document = fictional_authority()
    mutate(document)
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="scope changed"):
        diagnostic_authority.load_authority(tmp_path / AUTHORITY_NAME)


def test_load_authority_rejects_a_21st_slot(tmp_path, monkeypatch):
    document = fictional_authority()
    document["slots"].append({"slot": "20", "messages": {"sha256": "0" * 64}})
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="scope changed"):
        diagnostic_authority.load_authority(tmp_path / AUTHORITY_NAME)


def test_load_authority_rejects_shortened_slots(tmp_path, monkeypatch):
    document = fictional_authority()
    document["slots"] = document["slots"][:19]
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="scope changed"):
        diagnostic_authority.load_authority(tmp_path / AUTHORITY_NAME)


def test_load_authority_rejects_duplicate_slots(tmp_path, monkeypatch):
    document = fictional_authority()
    document["slots"][7]["slot"] = "04"
    _, sha = write_authority(tmp_path, document)
    monkeypatch.setattr(diagnostic_authority, "AUTHORITY_SHA256", sha)
    with pytest.raises(ValueError, match="scope changed"):
        diagnostic_authority.load_authority(tmp_path / AUTHORITY_NAME)


# ---------------------------------------------------------------------------
# The single authorized fifth prompt version
# ---------------------------------------------------------------------------


def test_check_version_accepts_the_exact_fifth(tmp_path, monkeypatch):
    entries = four_entries()
    prior_sha = diagnostic_authority.digest(diagnostic_authority.canonical(entries))
    body = sealed_authority(
        tmp_path,
        monkeypatch,
        fictional_authority(prior_assistant_versions_sha256=prior_sha),
    )
    diagnostic_authority.check_version(
        body, entries, FIFTH_ID, {"template_identity_sha256": "f" * 64}
    )


def test_check_version_rejects_altered_first_four(tmp_path, monkeypatch):
    entries = four_entries()
    prior_sha = diagnostic_authority.digest(diagnostic_authority.canonical(entries))
    body = sealed_authority(
        tmp_path,
        monkeypatch,
        fictional_authority(prior_assistant_versions_sha256=prior_sha),
    )
    altered = list(entries)
    altered[0] = dict(altered[0], version_id="assistant_prompt-rewritten")
    with pytest.raises(ValueError, match="preserve all four"):
        diagnostic_authority.check_version(
            body, altered, FIFTH_ID, {"template_identity_sha256": "f" * 64}
        )


def test_check_version_rejects_a_wrong_fifth_identity(tmp_path, monkeypatch):
    entries = four_entries()
    prior_sha = diagnostic_authority.digest(diagnostic_authority.canonical(entries))
    body = sealed_authority(
        tmp_path,
        monkeypatch,
        fictional_authority(prior_assistant_versions_sha256=prior_sha),
    )
    with pytest.raises(ValueError, match="only the explicitly authorized fifth"):
        diagnostic_authority.check_version(
            body,
            entries,
            "assistant_prompt-some-other",
            {"template_identity_sha256": "f" * 64},
        )
    with pytest.raises(ValueError, match="only the explicitly authorized fifth"):
        diagnostic_authority.check_version(
            body, entries, FIFTH_ID, {"template_identity_sha256": "e" * 64}
        )


# ---------------------------------------------------------------------------
# Frozen slot refs and the per-call bounds
# ---------------------------------------------------------------------------


def test_check_slot_accepts_every_frozen_slot(tmp_path, monkeypatch):
    body = sealed_authority(tmp_path, monkeypatch, fictional_authority())
    for index in range(20):
        slot = f"{index:02d}"
        diagnostic_authority.check_slot(body, slot_refs(slot), 120, 1)


def test_check_slot_rejects_a_21st_slot(tmp_path, monkeypatch):
    body = sealed_authority(tmp_path, monkeypatch, fictional_authority())
    with pytest.raises(ValueError, match="outside the twenty frozen"):
        diagnostic_authority.check_slot(body, slot_refs("20"), 120, 1)


@pytest.mark.parametrize("seconds", [0, 121, 200])
def test_check_slot_rejects_seconds_outside_bounds(tmp_path, monkeypatch, seconds):
    body = sealed_authority(tmp_path, monkeypatch, fictional_authority())
    with pytest.raises(ValueError, match="outside the twenty frozen"):
        diagnostic_authority.check_slot(body, slot_refs("00"), seconds, 1)


@pytest.mark.parametrize("generations", [0, 2])
def test_check_slot_rejects_wrong_generation_count(tmp_path, monkeypatch, generations):
    body = sealed_authority(tmp_path, monkeypatch, fictional_authority())
    with pytest.raises(ValueError, match="outside the twenty frozen"):
        diagnostic_authority.check_slot(body, slot_refs("00"), 120, generations)


def test_check_slot_rejects_a_foreign_messages_sha256(tmp_path, monkeypatch):
    body = sealed_authority(tmp_path, monkeypatch, fictional_authority())
    refs = slot_refs("00")
    refs["messages_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="outside the twenty frozen"):
        diagnostic_authority.check_slot(body, refs, 120, 1)


@pytest.mark.parametrize("field", ["diagnostic_id", "run_id"])
def test_check_slot_rejects_mismatched_ids(tmp_path, monkeypatch, field):
    body = sealed_authority(tmp_path, monkeypatch, fictional_authority())
    refs = slot_refs("00")
    refs[field] = "some-other-id"
    with pytest.raises(ValueError, match="outside the twenty frozen"):
        diagnostic_authority.check_slot(body, refs, 120, 1)
