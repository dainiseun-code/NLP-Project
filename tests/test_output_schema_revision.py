"""Requirement-first audit of the common output instructions against strict schema.

``docs/stage6-revision-contract.md`` asks for fictional independent cases, no
relaxed validation and no repair of saved outputs. The expected keys, actions and
rendering kinds are read from the published ``PROMPT_POLICY``/system instruction and
compared with the unchanged strict contract models. Only existing fictional helpers
from ``tests/test_application.py`` are reused; no prompt byte, validator, model,
process or study-data file is touched.
"""

import builtins
import copy
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from typing import get_args
from zoneinfo import TZPATH

import pytest
import test_application as fiction
from pydantic import ValidationError

from messaging_nlp.application import execute_condition
from messaging_nlp.application_types import (
    AliasTarget,
    GenerationOutput,
    Mention,
    OutputEnvelope,
)
from messaging_nlp.contracts import Decision, SelectedFact, Selection
from messaging_nlp.prompts import PROMPT_POLICY
from messaging_nlp.temporal import TimezoneBinding

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = (ROOT / "models", ROOT / "data", ROOT / "Old-version")
POLICY = PROMPT_POLICY["output_envelope"]


@pytest.fixture(autouse=True)
def fictional_only(monkeypatch):
    """Fail even when error handling swallows a forbidden attempt."""
    attempts = []
    original_import = builtins.__import__
    original_open = builtins.open
    original_io_open = io.open
    original_path_open = Path.open
    original_os_open = os.open

    def import_guard(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "mlx",
            "mlx_lm",
            "transformers",
            "tokenizers",
            "safetensors",
        }:
            attempts.append(name)
            raise AssertionError(f"optional model import forbidden: {name}")
        return original_import(name, *args, **kwargs)

    def check(path):
        try:
            resolved = Path(os.fspath(path)).resolve()
        except (TypeError, ValueError, OSError):
            return
        for root in FORBIDDEN_ROOTS:
            if resolved == root or root in resolved.parents:
                attempts.append(str(resolved))
                raise AssertionError(f"forbidden asset open: {resolved}")

    def open_guard(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            check(file)
        return original_open(file, *args, **kwargs)

    def io_open_guard(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            check(file)
        return original_io_open(file, *args, **kwargs)

    def path_open(self, *args, **kwargs):
        check(self)
        return original_path_open(self, *args, **kwargs)

    def os_open_guard(path, flags, mode=0o777, *, dir_fd=None):
        if isinstance(path, (str, bytes, os.PathLike)):
            check(path)
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    def no_process(*args, **kwargs):
        attempts.append("process launch")
        raise AssertionError("schema-audit tests cannot launch processes")

    monkeypatch.setattr(builtins, "__import__", import_guard)
    monkeypatch.setattr(builtins, "open", open_guard)
    monkeypatch.setattr(io, "open", io_open_guard)
    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(os, "open", os_open_guard)
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, no_process)
    yield
    assert not attempts, f"forbidden attempts: {attempts}"


@pytest.fixture
def binding(tmp_path):
    # Same fictional UTC binding as tests/test_application.py: a copied system TZif.
    source = next(
        (Path(p) / "UTC" for p in TZPATH if (Path(p) / "UTC").is_file()), None
    )
    assert source is not None, "fixture requires a system UTC TZif"
    content = source.read_bytes()
    assert content.startswith(b"TZif")
    path = tmp_path / "fictional-UTC.tzif"
    path.write_bytes(content)
    return TimezoneBinding("UTC", path, hashlib.sha256(content).hexdigest())


def draft_payload():
    """A minimal schema-valid draft using every prompt-declared key explicitly."""
    return {
        "decision": {
            "action": "draft",
            "selection": {
                "recipient_contact_ids": ["contact-1"],
                "selected_event_id": None,
                "facts": [
                    {
                        "field": "name",
                        "value": "Ren",
                        "source_record_ids": ["record-1"],
                    }
                ],
                "source_record_ids": ["record-1"],
            },
            "draft": "Hi Ren",
            "clarification": None,
            "reason": None,
        },
        "mentions": [{"surface": "Ren", "rendering_kind": "literal", "fact_index": 0}],
    }


def delete(value, path):
    target = value
    for step in path[:-1]:
        target = target[step]
    del target[path[-1]]
    return value


def replace(value, path, new):
    target = value
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = new
    return value


class RawProvider:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def generate(self, prompt, *, max_input_tokens, max_output_tokens):
        assert (max_input_tokens, max_output_tokens) == (2048, 1024)
        self.calls += 1
        return GenerationOutput(raw_output=self.raw)


def run_raw(dataset, inputs, binding, root, raw, condition="B"):
    provider = RawProvider(raw)
    result = execute_condition(
        dataset,
        dataset.requests[0].message_id,
        condition,
        inputs,
        provider,
        root=root,
        run_id="fictional-run",
        binding=binding,
    )
    return result, provider


def test_prompt_policy_declares_exactly_the_strict_schema():
    assert tuple(OutputEnvelope.model_fields) == POLICY["keys"]
    assert tuple(Decision.model_fields) == POLICY["decision_keys"]
    assert tuple(Selection.model_fields) == POLICY["selection_keys"]
    assert tuple(SelectedFact.model_fields) == POLICY["fact_keys"]
    assert tuple(Mention.model_fields) == POLICY["mention_keys"]
    assert get_args(Decision.model_fields["action"].annotation) == POLICY["actions"]
    assert (
        get_args(SelectedFact.model_fields["field"].annotation) == POLICY["fact_fields"]
    )
    assert (
        get_args(Mention.model_fields["rendering_kind"].annotation)
        == POLICY["rendering_kinds"]
    )
    assert (
        get_args(AliasTarget.model_fields["kind"].annotation)
        == (PROMPT_POLICY["alias_kinds"])
    )
    # "decision always contains all of" is enforceable: no field is optional-by-default.
    assert all(
        OutputEnvelope.model_fields[name].is_required() for name in POLICY["keys"]
    )
    assert all(
        Decision.model_fields[name].is_required() for name in POLICY["decision_keys"]
    )
    assert all(
        Selection.model_fields[name].is_required() for name in POLICY["selection_keys"]
    )
    assert all(
        SelectedFact.model_fields[name].is_required() for name in POLICY["fact_keys"]
    )
    assert Mention.model_fields["surface"].is_required()
    assert Mention.model_fields["rendering_kind"].is_required()
    # "exactly one of fact_index or anchor_id" is a validator over two optional keys.
    assert not Mention.model_fields["fact_index"].is_required()
    assert not Mention.model_fields["anchor_id"].is_required()
    assert POLICY["mentions_reference"] == "exactly one of fact_index or anchor_id"
    assert POLICY["non_draft_mentions"] == ()


@pytest.mark.parametrize("action", ["clarify", "unsupported"])
def test_non_draft_prompt_declared_null_fields_are_accepted(action):
    envelope = OutputEnvelope.model_validate(fiction.nondraft(action))
    decision = envelope.decision
    assert decision.action == action
    assert decision.selection is None and decision.draft is None
    assert envelope.mentions == []
    if action == "clarify":
        assert decision.clarification and decision.reason
    else:
        assert decision.clarification is None and decision.reason


def test_draft_prompt_declared_null_fields_are_accepted():
    envelope = OutputEnvelope.model_validate(draft_payload())
    decision = envelope.decision
    assert decision.action == "draft"
    assert decision.clarification is None and decision.reason is None
    assert decision.selection is not None and decision.draft == "Hi Ren"
    assert [mention.fact_index for mention in envelope.mentions] == [0]
    assert envelope.mentions[0].anchor_id is None


@pytest.mark.parametrize(
    "path",
    [
        ("decision",),
        ("mentions",),
        ("decision", "action"),
        ("decision", "selection"),
        ("decision", "draft"),
        ("decision", "clarification"),
        ("decision", "reason"),
        ("decision", "selection", "recipient_contact_ids"),
        ("decision", "selection", "selected_event_id"),
        ("decision", "selection", "facts"),
        ("decision", "selection", "source_record_ids"),
        ("decision", "selection", "facts", 0, "field"),
        ("decision", "selection", "facts", 0, "value"),
        ("decision", "selection", "facts", 0, "source_record_ids"),
        ("mentions", 0, "surface"),
        ("mentions", 0, "rendering_kind"),
    ],
)
def test_omitted_fields_are_rejected_without_defaults(path):
    with pytest.raises(ValidationError):
        OutputEnvelope.model_validate(delete(copy.deepcopy(draft_payload()), path))


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("decision",),
        ("decision", "selection"),
        ("decision", "selection", "facts", 0),
        ("mentions", 0),
    ],
)
def test_unknown_keys_are_rejected_not_stripped(path):
    value = copy.deepcopy(draft_payload())
    target = value
    for step in path:
        target = target[step]
    target["extra_key"] = "must be rejected"
    with pytest.raises(ValidationError):
        OutputEnvelope.model_validate(value)


@pytest.mark.parametrize(
    "path,value",
    [
        (("decision", "action"), "Draft"),
        (("decision", "action"), "send"),
        (("decision", "selection", "facts", 0, "field"), "Name"),
        (("decision", "selection", "facts", 0, "field"), "description"),
        (("mentions", 0, "rendering_kind"), "Literal"),
        (("mentions", 0, "rendering_kind"), "quote"),
    ],
)
def test_prompt_enumerated_values_are_the_only_accepted_values(path, value):
    with pytest.raises(ValidationError):
        OutputEnvelope.model_validate(
            replace(copy.deepcopy(draft_payload()), path, value)
        )


def test_non_draft_mentions_must_be_empty_and_references_exclusive():
    mention = draft_payload()["mentions"][0]
    non_draft = fiction.nondraft("unsupported")
    non_draft["mentions"] = [mention]
    with pytest.raises(ValidationError):
        OutputEnvelope.model_validate(non_draft)

    neither = copy.deepcopy(draft_payload())
    neither["mentions"] = [{"surface": "Ren", "rendering_kind": "literal"}]
    with pytest.raises(ValidationError):
        OutputEnvelope.model_validate(neither)

    both_null = copy.deepcopy(draft_payload())
    both_null["mentions"] = [
        {
            "surface": "Ren",
            "rendering_kind": "literal",
            "fact_index": None,
            "anchor_id": None,
        }
    ]
    with pytest.raises(ValidationError):
        OutputEnvelope.model_validate(both_null)

    both = copy.deepcopy(draft_payload())
    both["mentions"] = [
        {
            "surface": "Ren",
            "rendering_kind": "literal",
            "fact_index": 0,
            "anchor_id": "request-anchor-0000",
        }
    ]
    with pytest.raises(ValidationError):
        OutputEnvelope.model_validate(both)

    explicit_null = copy.deepcopy(draft_payload())
    explicit_null["mentions"] = [
        {
            "surface": "Ren",
            "rendering_kind": "literal",
            "fact_index": 0,
            "anchor_id": None,
        }
    ]
    validated = OutputEnvelope.model_validate(explicit_null)
    assert validated.mentions[0].fact_index == 0
    assert validated.mentions[0].anchor_id is None


@pytest.mark.parametrize(
    "action,path,value",
    [
        ("draft", ("decision", "clarification"), "Which one?"),
        ("draft", ("decision", "reason"), "because"),
        ("draft", ("decision", "selection"), None),
        ("draft", ("decision", "draft"), ""),
        ("draft", ("decision", "selection", "recipient_contact_ids"), []),
        ("draft", ("decision", "selection", "source_record_ids"), []),
        ("clarify", ("decision", "draft"), "prose"),
        ("clarify", ("decision", "selection"), {"recipient_contact_ids": []}),
        ("clarify", ("decision", "clarification"), None),
        ("unsupported", ("decision", "clarification"), "why?"),
        ("unsupported", ("decision", "selection"), {"recipient_contact_ids": []}),
        ("unsupported", ("decision", "reason"), None),
    ],
)
def test_incoherent_actions_are_rejected_without_repair(action, path, value):
    base = draft_payload() if action == "draft" else fiction.nondraft(action)
    with pytest.raises(ValidationError):
        OutputEnvelope.model_validate(replace(copy.deepcopy(base), path, value))


@pytest.mark.parametrize("fault", ["omit_reason", "extra_decision_key"])
def test_schema_invalid_saved_output_is_retained_verbatim(tmp_path, binding, fault):
    dataset, inputs = fiction.prepare(tmp_path, binding)
    saved = fiction.packet(dataset, inputs, tmp_path, binding)
    envelope = fiction.draft_envelope(saved)
    if fault == "omit_reason":
        del envelope["decision"]["reason"]
    else:
        envelope["decision"]["extra_key"] = "must be rejected"
    raw = json.dumps(envelope)
    result, provider = run_raw(dataset, inputs, binding, tmp_path, raw)
    assert provider.calls == 1
    assert result.row.status == "failed"
    assert result.row.failure.stage == "validation"
    assert result.row.failure.code == "invalid_output"
    assert result.row.raw_output == raw
    assert result.row.decision is None
    retained = json.loads(result.row.raw_output)
    if fault == "omit_reason":
        assert "reason" not in retained["decision"]
    else:
        assert retained["decision"]["extra_key"] == "must be rejected"


def test_invented_alias_is_rejected_not_guessed(tmp_path, binding):
    dataset, inputs = fiction.prepare(tmp_path, binding)
    saved = fiction.packet(dataset, inputs, tmp_path, binding)
    envelope = fiction.draft_envelope(saved)
    envelope["decision"]["selection"]["recipient_contact_ids"] = [
        "contact-alias-not-printed"
    ]
    raw = json.dumps(envelope)
    result, provider = run_raw(dataset, inputs, binding, tmp_path, raw)
    assert provider.calls == 1
    assert result.row.status == "failed"
    assert result.row.failure.stage == "validation"
    assert "unknown or wrong-kind contact alias" in result.row.failure.message
    assert result.row.raw_output == raw and result.row.decision is None
