"""Requirement-first Stage6 revision checks for the current1024 output ceiling.

Expected behavior comes from ``docs/stage6-revision-contract.md``: current
input2048/output1024/call120; historical256 plans stay structurally readable but
remain bounded at their own256 ceiling; the worker streams every ``1..ceiling``
count, requires progress for empty-text updates and emits one terminal event.
All runtimes below are fictional objects: no mlx/mlx-lm, tokenizer, process,
model or study-data access occurs.
"""

import builtins
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import TZPATH

import pytest
import test_application as fiction
from pydantic import ValidationError

from messaging_nlp import limits, qwen_worker
from messaging_nlp.application import execute_condition
from messaging_nlp.application_types import GenerationOutput
from messaging_nlp.io import canonical_json
from messaging_nlp.run_contracts import (
    QWEN_RECEIPT_SHA256,
    DecodePolicy,
    TokenReceipt,
)
from messaging_nlp.temporal import TimezoneBinding

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = (ROOT / "models", ROOT / "data", ROOT / "Old-version")
MESSAGES = [{"role": "user", "content": "fictional"}]
MESSAGE_TOKENS = [7, 8]
TEMPLATE = "fictional qwen chat template"
TEMPLATE_KWARGS = {
    "tokenize": True,
    "add_generation_prompt": True,
    "enable_thinking": False,
    "return_dict": False,
    "truncation": False,
    "padding": False,
}


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
        raise AssertionError("limit tests cannot launch processes")

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


def digest(value):
    return hashlib.sha256(value).hexdigest()


class FakeTokenizer:
    def __init__(self, token_ids):
        self.chat_template = TEMPLATE
        self.token_ids = list(token_ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == TEMPLATE_KWARGS
        return list(self.token_ids)


def response(text, finish, count, prompt_tokens=2):
    return SimpleNamespace(
        text=text,
        finish_reason=finish,
        generation_tokens=count,
        prompt_tokens=prompt_tokens,
    )


def runtime_with(ceiling, responses, calls, token_ids=MESSAGE_TOKENS):
    runtime = object.__new__(qwen_worker.QwenRuntime)
    runtime.tokenizer = FakeTokenizer(token_ids)
    runtime.decoding = DecodePolicy(max_output_tokens=ceiling)
    runtime.model = object()
    runtime.template_sha256 = digest(TEMPLATE.encode())
    runtime.identity = {"receipt_sha256": QWEN_RECEIPT_SHA256}
    runtime.mx = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: None), clear_cache=lambda: None
    )
    runtime.make_cache = lambda model: object()
    runtime.make_sampler = lambda **settings: settings

    def stream(*args, **kwargs):
        calls.append(kwargs)
        return iter(responses)

    runtime.stream = stream
    return runtime


def wire_payload(runtime):
    receipt = TokenReceipt(
        messages_sha256=digest(canonical_json(MESSAGES)),
        template_sha256=runtime.template_sha256,
        model_receipt_sha256=QWEN_RECEIPT_SHA256,
        token_ids_sha256=digest(canonical_json(MESSAGE_TOKENS)),
        token_count=len(MESSAGE_TOKENS),
        token_ids=MESSAGE_TOKENS,
    )
    return {"messages": MESSAGES, "token_receipt": receipt.model_dump(mode="json")}


class RecordingProvider:
    def __init__(self):
        self.ceilings = []

    def generate(self, prompt, *, max_input_tokens, max_output_tokens):
        self.ceilings.append((max_input_tokens, max_output_tokens))
        return GenerationOutput(raw_output=json.dumps(fiction.nondraft()))


def run_condition(tmp_path, binding, **kwargs):
    dataset, inputs = fiction.prepare(tmp_path, binding)
    provider = RecordingProvider()
    result = execute_condition(
        dataset,
        dataset.requests[0].message_id,
        "B",
        inputs,
        provider,
        root=tmp_path,
        run_id="fictional-run",
        binding=binding,
        **kwargs,
    )
    return result, provider


def test_central_limits_are_the_current_policy():
    assert limits.MAX_INPUT_TOKENS == 2048
    assert limits.MAX_OUTPUT_TOKENS == 1024
    assert limits.CALL_TIMEOUT_SECONDS == 120
    assert limits.MEMORY_WARNING_SECONDS == 10
    assert limits.PROGRESS_INTERVAL_SECONDS == 10


def test_protocol_queue_reserves_a_full_token_burst():
    # start + up to (1024 updates + terminal-carrying update) + terminal = 1027.
    assert limits.MAX_PROTOCOL_QUEUE_EVENTS == limits.MAX_OUTPUT_TOKENS + 4
    assert limits.MAX_PROTOCOL_QUEUE_EVENTS >= limits.MAX_OUTPUT_TOKENS + 3


def test_decode_policy_defaults_current_and_reads_legacy_256():
    policy = DecodePolicy()
    assert policy.max_output_tokens == limits.MAX_OUTPUT_TOKENS == 1024
    assert policy.max_input_tokens == limits.MAX_INPUT_TOKENS == 2048
    assert policy.call_timeout_seconds == limits.CALL_TIMEOUT_SECONDS == 120
    assert policy.temperature == 0 and policy.seed == 42
    assert policy.enable_thinking is False
    legacy = DecodePolicy(max_output_tokens=256)
    assert DecodePolicy.model_validate(legacy.model_dump()) == legacy
    assert legacy.model_dump(mode="json")["max_output_tokens"] == 256


@pytest.mark.parametrize(
    "ceiling",
    [0, 1, 128, 255, 257, 512, 1025, 2048, -1024, True, False, 1024.0, "1024", None],
)
def test_decode_policy_rejects_other_ceilings(ceiling):
    with pytest.raises(ValidationError):
        DecodePolicy(max_output_tokens=ceiling)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_input_tokens", 1024),
        ("max_input_tokens", 2048.0),
        ("temperature", 1),
        ("seed", 0),
        ("enable_thinking", True),
        ("call_timeout_seconds", 60),
        ("call_timeout_seconds", 180),
    ],
)
def test_unchanged_decode_fields_stay_exact(field, value):
    with pytest.raises(ValidationError):
        DecodePolicy(**{field: value})


def test_application_default_ceiling_is_current(tmp_path, binding):
    result, provider = run_condition(tmp_path, binding)
    assert provider.ceilings == [(2048, 1024)]
    assert result.row.status == "ok"


def test_application_accepts_explicit_legacy_256(tmp_path, binding):
    result, provider = run_condition(tmp_path, binding, max_output_tokens=256)
    assert provider.ceilings == [(2048, 256)]
    assert result.row.status == "ok"


@pytest.mark.parametrize(
    "ceiling", [0, 255, 512, 2048, -1024, True, False, 1024.0, 1024.5, "1024", None]
)
def test_application_rejects_other_ceilings(tmp_path, binding, ceiling):
    dataset, inputs = fiction.prepare(tmp_path, binding)
    provider = RecordingProvider()
    with pytest.raises(ValueError, match="generation ceiling"):
        execute_condition(
            dataset,
            dataset.requests[0].message_id,
            "B",
            inputs,
            provider,
            root=tmp_path,
            run_id="fictional-run",
            binding=binding,
            max_output_tokens=ceiling,
        )
    assert provider.ceilings == []


def test_worker_streams_full_1024_burst_and_one_terminal():
    ceiling = limits.MAX_OUTPUT_TOKENS
    responses = [
        response("x" if count % 4 else "", None, count)
        for count in range(1, ceiling + 1)
    ]
    responses.append(response("done", "length", ceiling))
    calls, events = [], []
    runtime = runtime_with(ceiling, responses, calls)
    runtime.generate(
        wire_payload(runtime), lambda kind, payload: events.append((kind, payload))
    )
    kinds = [kind for kind, _ in events]
    assert calls[0]["max_tokens"] == ceiling == 1024
    assert kinds[0] == "generation_started"
    assert kinds[-1] == "terminal"
    assert kinds.count("generation_started") == kinds.count("terminal") == 1
    assert len(kinds) <= limits.MAX_PROTOCOL_QUEUE_EVENTS
    streamed = [payload["generation_tokens"] for _, payload in events[1:-1]]
    assert streamed[:ceiling] == list(range(1, ceiling + 1))
    assert all(1 <= count <= ceiling for count in streamed)
    expected_progress = set(range(4, ceiling + 1, 4))
    progress = [p["generation_tokens"] for k, p in events[1:-1] if k == "progress"]
    assert expected_progress <= set(progress)
    for kind, payload in events[1:-1]:
        if kind == "fragment":
            assert payload["text"]
        else:
            assert kind == "progress"
            assert set(payload) == {"generation_tokens"}
    terminal = events[-1][1]
    assert terminal == {
        "finish_reason": "length",
        "generation_tokens": ceiling,
        "input_tokens": 2,
        "elapsed_seconds": terminal["elapsed_seconds"],
    }


@pytest.mark.parametrize("finish", [None, "stop", "length"])
def test_worker_rejects_count_above_current_ceiling(finish):
    calls, events = [], []
    runtime = runtime_with(1024, [response("x", finish, 1025)], calls)
    with pytest.raises(RuntimeError, match="invalid runtime"):
        runtime.generate(
            wire_payload(runtime), lambda kind, payload: events.append((kind, payload))
        )
    assert events == [("generation_started", {})]


def test_worker_legacy_256_plan_stays_bounded_at_256():
    calls, events = [], []
    responses = [
        response("x" if count % 3 else "", None, count) for count in range(1, 257)
    ]
    responses.append(response("done", "stop", 256))
    runtime = runtime_with(256, responses, calls)
    runtime.generate(
        wire_payload(runtime), lambda kind, payload: events.append((kind, payload))
    )
    assert calls[0]["max_tokens"] == 256
    assert 256 != limits.MAX_OUTPUT_TOKENS
    assert events[-1][0] == "terminal"
    assert events[-1][1]["generation_tokens"] == 256
    streamed = [payload["generation_tokens"] for _, payload in events[1:-1]]
    assert streamed[:256] == list(range(1, 257))
    assert all(1 <= count <= 256 for count in streamed)
    progress = [p["generation_tokens"] for k, p in events[1:-1] if k == "progress"]
    assert set(range(3, 257, 3)) <= set(progress)
    # 257 is valid under the current1024 policy but not under this legacy plan.
    calls, events = [], []
    runtime = runtime_with(256, [response("x", None, 257)], calls)
    with pytest.raises(RuntimeError, match="invalid runtime"):
        runtime.generate(
            wire_payload(runtime), lambda kind, payload: events.append((kind, payload))
        )
    assert events == [("generation_started", {})]


def test_worker_rejects_a_runtime_that_yields_after_terminal():
    calls, events = [], []
    runtime = runtime_with(
        1024, [response("done", "stop", 7), response("late", None, 8)], calls
    )
    with pytest.raises(RuntimeError, match="after terminal"):
        runtime.generate(
            wire_payload(runtime), lambda kind, payload: events.append((kind, payload))
        )
    # The observed fragment is retained; no terminal is claimed for a failed call.
    assert [kind for kind, _ in events] == ["generation_started", "fragment"]


def test_worker_prompt_tokens_stay_bounded_at_2048():
    calls, events = [], []
    runtime = runtime_with(1024, [], calls, token_ids=list(range(2049)))
    with pytest.raises(ValueError, match="2048"):
        runtime.generate(
            wire_payload(runtime), lambda kind, payload: events.append((kind, payload))
        )
    assert calls == [] and events == []
