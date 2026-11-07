"""Contract-derived fictional tests for the preparation worker (model-free)."""

import builtins
import hashlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from messaging_nlp import preparation_worker as w
from messaging_nlp.contracts import Extraction, ExtractionResult, TechnicalFailure
from messaging_nlp.encoder_cache import (
    ExtractionArtifact,
    retained_identity,
    text_sha256,
)
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.run_contracts import PROTOCOL_VERSION, ModelSpec


@pytest.fixture(autouse=True)
def deny_real_assets_and_optional_imports(monkeypatch):
    original_import = builtins.__import__
    original_open = io.open
    project = Path(__file__).resolve().parents[1]

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {
            "mlx",
            "mlx_lm",
            "torch",
            "transformers",
            "tokenizers",
            "safetensors",
        }:
            raise AssertionError(f"optional runtime import forbidden: {name}")
        return original_import(name, *args, **kwargs)

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, Path)):
            candidate = Path(file).resolve()
            if any(
                candidate.is_relative_to(project / name)
                for name in ("models", "Old-version", "data")
            ):
                raise AssertionError(f"real asset open forbidden: {candidate}")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)


def digest(value) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class FakeTokenizer:
    chat_template = "fictional template"

    def __init__(self, result=None):
        self.result = [7, 8] if result is None else result
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.result


def line(kind, payload=None, row_id=None) -> bytes:
    return (
        canonical_json(
            {
                "protocol": PROTOCOL_VERSION,
                "kind": kind,
                "row_id": row_id,
                "payload": payload or {},
            }
        )
        + b"\n"
    )


def emitter():
    events = []

    def emit(kind, payload, row_id=None):
        events.append({"kind": kind, "row_id": row_id, "payload": payload})

    return events, emit


class FakeRuntime:
    def __init__(self, payload):
        self.identity = {"mode": payload.get("mode"), "fictional": True}
        self.payload = payload
        self.generated = []

    def generate(self, payload, emit):
        self.generated.append(payload)
        reply = {"mode": self.payload.get("mode"), "echo": payload}
        w._emit_preparation(
            emit,
            f"{self.payload.get('mode')}_op",
            reply,
            input_tokens=0,
            started=time.monotonic(),
        )


# --------------------------------------------------------------------------- #
# Tokenizer helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "invalid", [{"input_ids": [1]}, [[1]], [], [True], [1.0], [-1], "one"]
)
def test_chat_token_ids_reject_malformed_flat_ids(invalid):
    with pytest.raises(ValueError):
        w.chat_token_ids(FakeTokenizer(invalid), [])


def test_chat_token_ids_exact_flags_and_no_ceiling():
    tokenizer = FakeTokenizer()
    assert w.chat_token_ids(tokenizer, []) == [7, 8]
    assert tokenizer.calls[0][1] == dict(
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
        truncation=False,
        padding=False,
    )
    assert len(w.chat_token_ids(FakeTokenizer([1] * 2049), [])) == 2049


# --------------------------------------------------------------------------- #
# Deny boundary: imports and weight payload opens
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["mlx", "mlx_lm", "torch", "safetensors", "mlx.core"])
def test_deny_import_blocks_optional_runtimes(name):
    with pytest.raises(RuntimeError, match="forbidden"):
        w._deny_import(name)


@pytest.mark.parametrize("name", ["transformers", "json", "pathlib", "messaging_nlp"])
def test_deny_import_permits_static_and_stdlib(name):
    w._deny_import(name)


@pytest.mark.parametrize(
    "path", ["model.safetensors", "weights.bin", "x.pt", "x.pth", "x.ckpt", "x.gguf"]
)
def test_deny_open_blocks_weight_payloads(path):
    with pytest.raises(RuntimeError, match="weight payload"):
        w._deny_open(Path(path))
    with pytest.raises(RuntimeError, match="weight payload"):
        w._deny_open(path)


@pytest.mark.parametrize(
    "path",
    ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
)
def test_deny_open_permits_config_tokenizer_template(path):
    w._deny_open(Path(path))


def test_deny_open_ignores_file_descriptors():
    w._deny_open(3)


def test_runtime_boundary_denies_weight_open_permits_static(tmp_path):
    config = tmp_path / "config.json"
    weights = tmp_path / "model.safetensors"
    config.write_bytes(b"{}")
    weights.write_bytes(b"tiny fake weight payload")
    boundary = w._TokenizerRuntimeBoundary()
    boundary.__enter__()
    try:
        with open(config, "rb") as handle:
            assert handle.read() == b"{}"
        with pytest.raises(RuntimeError, match="weight payload"):
            open(weights, "rb")
        with pytest.raises(RuntimeError, match="weight payload"):
            weights.read_bytes()
    finally:
        boundary.__exit__(None, None, None)
    # The boundary restores ordinary opens after the tokenizer load scope.
    assert weights.read_bytes() == b"tiny fake weight payload"


# --------------------------------------------------------------------------- #
# Encoder generate: reply shape, event order, failed vs valid-empty
# --------------------------------------------------------------------------- #


def encoder_runtime(fake_encoder):
    runtime = object.__new__(w._EncoderRuntime)
    runtime.encoder = fake_encoder
    return runtime


def encoder_artifact(status, entities, text="fictional request"):
    identity = retained_identity()
    result = ExtractionResult(
        message_id="m1",
        status=status,
        extraction=Extraction(entities=entities) if status == "ok" else None,
        failure=(
            None
            if status == "ok"
            else TechnicalFailure(
                stage="extraction", code="runtime_error", message="fictional failure"
            )
        ),
        raw_output=None,
        input_sha256=text_sha256(text),
        extractor_sha256=identity.digest(),
    )
    return ExtractionArtifact(
        text=text,
        identity=identity,
        result=result,
        diagnostics={"text_modified": False, "forward_performed": False},
    )


class FakeEncoder:
    def __init__(self, artifact):
        self._artifact = artifact
        self.calls = []

    def extract(self, message_id, text):
        self.calls.append((message_id, text))
        return self._artifact


@pytest.mark.parametrize(
    "status,entities",
    [
        ("failed", None),
        ("ok", []),
        ("ok", [{"type": "TOPIC", "text": "fictional", "start": 0, "end": 9}]),
    ],
)
def test_encoder_generate_preserves_failed_and_empty_distinctly(status, entities):
    runtime = encoder_runtime(FakeEncoder(encoder_artifact(status, entities)))
    events = []
    runtime.generate(
        {"message_id": "m1", "text": "fictional request"},
        lambda k, v: events.append((k, v)),
    )
    assert [k for k, _ in events] == ["generation_started", "fragment", "terminal"]
    assert events[0][1] == {"operation": "encoder_extraction", "qwen_generation": False}
    reply = loads_json(events[1][1]["text"])
    assert set(reply) == {"mode", "artifact"}
    assert reply["mode"] == "encoder"
    assert reply["artifact"]["result"]["status"] == status
    assert reply["artifact"]["result"]["extraction"] == (
        {"entities": entities} if entities is not None else None
    )
    terminal = events[2][1]
    assert terminal["finish_reason"] == "stop"
    assert terminal["generation_tokens"] == 0
    assert terminal["input_tokens"] == 0
    assert terminal["elapsed_seconds"] >= 0
    # Fragment text is exactly canonical JSON of the reply.
    assert canonical_json(reply) == events[1][1]["text"].encode("utf-8")


def test_encoder_start_ack_precedes_work_not_just_its_completed_reply():
    events = []

    class Encoder:
        def extract(self, message_id, text):
            assert events[0][0] == "generation_started"
            assert events[0][1]["qwen_generation"] is False
            return encoder_artifact("ok", [])

    encoder_runtime(Encoder()).generate(
        {"message_id": "m1", "text": "fictional request"},
        lambda kind, payload: events.append((kind, payload)),
    )
    assert [kind for kind, _ in events].count("generation_started") == 1


def test_tokenizer_guard_and_start_ack_remain_active_during_chat_template(tmp_path):
    import importlib

    events = []
    weights = tmp_path / "fictional.safetensors"
    weights.write_bytes(b"not actual weights")
    original_io_open, original_os_open = io.open, os.open

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert events[0][0] == "generation_started"
            with pytest.raises(RuntimeError, match="weight payload"):
                os.open(weights, os.O_RDONLY)
            with pytest.raises(RuntimeError, match="forbidden"):
                importlib.import_module("mlx")
            return [1, 2]

    messages = [{"role": "user", "content": "fictional"}]
    tokenizer_runtime(Tokenizer()).generate(
        {"messages": messages, "messages_sha256": digest(canonical_json(messages))},
        lambda kind, payload: events.append((kind, payload)),
    )
    assert io.open is original_io_open and os.open is original_os_open


def test_extra_worker_fields_do_not_silently_change_the_contract():
    with pytest.raises(ValueError, match="exact"):
        load = {"mode": "encoder", "root": "/r", "model": None, "retry": True}
        w.load_runtime(load)
    with pytest.raises(ValueError, match="absolute"):
        w.load_runtime({"mode": "encoder", "root": ".", "model": None})
    with pytest.raises(ValueError, match="exact"):
        encoder_runtime(FakeEncoder(encoder_artifact("ok", []))).generate(
            {"message_id": "m1", "text": "fictional request", "clean_text": "repair"},
            lambda *args: None,
        )


def test_tensor_availability_probes_are_false_but_actual_imports_still_fail():
    import importlib
    import importlib.util

    original = importlib.util.find_spec
    with w._TokenizerRuntimeBoundary():
        for name in w.DENIED_RUNTIME_IMPORTS:
            assert importlib.util.find_spec(name) is None
        assert importlib.util.find_spec("json") is not None
        with pytest.raises(RuntimeError, match="forbidden"):
            importlib.import_module("mlx")
    assert importlib.util.find_spec is original


def test_encoder_generate_requires_nonblank_message_id_and_text():
    runtime = encoder_runtime(FakeEncoder(encoder_artifact("ok", [])))
    for payload in [
        {"message_id": "", "text": "x"},
        {"message_id": "m1"},
        {"message_id": "m1", "text": 3},
    ]:
        with pytest.raises(ValueError):
            runtime.generate(payload, lambda k, v: None)


# --------------------------------------------------------------------------- #
# Tokenizer generate: exact hash/flags, overlength, terminal gen 0
# --------------------------------------------------------------------------- #


def tokenizer_runtime(tokenizer, template_sha256=None):
    runtime = object.__new__(w._TokenizerRuntime)
    runtime.tokenizer = tokenizer
    runtime.template_sha256 = template_sha256 or digest("fictional template")
    return runtime


def test_tokenizer_generate_exact_flags_hash_and_reply_shape():
    tokenizer = FakeTokenizer()
    runtime = tokenizer_runtime(tokenizer)
    messages = [{"role": "user", "content": "fictional"}]
    events = []
    runtime.generate(
        {"messages": messages, "messages_sha256": digest(canonical_json(messages))},
        lambda k, v: events.append((k, v)),
    )
    assert tokenizer.calls[0][1] == dict(
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
        truncation=False,
        padding=False,
    )
    assert [k for k, _ in events] == ["generation_started", "fragment", "terminal"]
    assert events[0][1] == {"operation": "tokenizer_fit", "qwen_generation": False}
    reply = loads_json(events[1][1]["text"])
    assert reply == {
        "mode": "tokenizer",
        "messages_sha256": digest(canonical_json(messages)),
        "template_sha256": runtime.template_sha256,
        "model_receipt_sha256": w.QWEN_RECEIPT_SHA256,
        "token_ids": [7, 8],
        "token_count": 2,
        "token_ids_sha256": digest(canonical_json([7, 8])),
        "fits": True,
    }
    terminal = events[2][1]
    assert terminal["finish_reason"] == "stop"
    assert terminal["generation_tokens"] == 0
    assert terminal["input_tokens"] == 2
    assert terminal["elapsed_seconds"] >= 0


def test_tokenizer_generate_rejects_wrong_messages_hash():
    runtime = tokenizer_runtime(FakeTokenizer())
    events = []
    with pytest.raises(ValueError, match="hash mismatch"):
        runtime.generate(
            {
                "messages": [{"role": "user", "content": "x"}],
                "messages_sha256": "0" * 64,
            },
            lambda k, v: events.append((k, v)),
        )
    assert events == []


def test_tokenizer_generate_preserves_overlength_ids_and_count():
    tokenizer = FakeTokenizer([1] * 2049)
    runtime = tokenizer_runtime(tokenizer)
    messages = [{"role": "user", "content": "long"}]
    events = []
    runtime.generate(
        {"messages": messages, "messages_sha256": digest(canonical_json(messages))},
        lambda k, v: events.append((k, v)),
    )
    reply = loads_json(events[1][1]["text"])
    assert reply["fits"] is False
    assert reply["token_count"] == 2049
    assert len(reply["token_ids"]) == 2049
    assert reply["token_ids_sha256"] == digest(canonical_json([1] * 2049))
    assert events[2][1]["input_tokens"] == 2049


# --------------------------------------------------------------------------- #
# Load dispatch and mode restrictions
# --------------------------------------------------------------------------- #


def test_load_runtime_dispatch_and_unsupported_mode(monkeypatch):
    made = []

    class FakeEnc:
        def __init__(self, root, model):
            made.append(("encoder", root, model))

    class FakeTok:
        def __init__(self, root, model):
            made.append(("tokenizer", root, model))

    monkeypatch.setattr(w, "_EncoderRuntime", FakeEnc)
    monkeypatch.setattr(w, "_TokenizerRuntime", FakeTok)
    w.load_runtime({"mode": "encoder", "root": "/r", "model": None})
    w.load_runtime({"mode": "tokenizer", "root": "/r", "model": {"path": "m"}})
    assert made == [("encoder", "/r", None), ("tokenizer", "/r", {"path": "m"})]
    with pytest.raises(ValueError, match="unsupported"):
        w.load_runtime({"mode": "qwen", "root": "/r", "model": None})


def test_encoder_loader_rejects_model_spec(monkeypatch):
    monkeypatch.setattr(w, "_verify_worker_source", lambda root: {})
    with pytest.raises(ValueError, match="no model spec"):
        w._EncoderRuntime(root="/nonexistent", model={"path": "m"})


def test_tokenizer_load_orders_verify_before_deny_before_autotokenizer(
    tmp_path, monkeypatch
):
    order = []
    monkeypatch.setattr(
        w,
        "_verify_worker_source",
        lambda root: order.append("source") or {"f.py": "0" * 64},
    )
    monkeypatch.setattr(
        w,
        "verify_model",
        lambda root, spec: order.append("verify_model") or {"receipt_sha256": "0" * 64},
    )
    original_enter = w._TokenizerRuntimeBoundary.__enter__
    original_exit = w._TokenizerRuntimeBoundary.__exit__

    def enter(self):
        order.append("boundary_enter")
        return original_enter(self)

    def exit(self, *exc):
        order.append("boundary_exit")
        return original_exit(self, *exc)

    monkeypatch.setattr(w._TokenizerRuntimeBoundary, "__enter__", enter)
    monkeypatch.setattr(w._TokenizerRuntimeBoundary, "__exit__", exit)

    template = "fictional template"
    tokenizer = NS(chat_template=template)
    loaded = []

    def from_pretrained(*args, **kwargs):
        order.append("autotokenizer_load")
        loaded.append((args, kwargs))
        return tokenizer

    stubs = {"transformers": NS(AutoTokenizer=NS(from_pretrained=from_pretrained))}
    autouse_guard = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name in stubs:
            return stubs[name]
        return autouse_guard(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)
    spec = ModelSpec(
        path="models/qwen",
        receipt={"path": "receipt.json", "sha256": "0" * 64, "bytes": 0},
        template_sha256=digest(template),
    )
    runtime = w._TokenizerRuntime(str(tmp_path), spec.model_dump(mode="json"))
    assert order == [
        "source",
        "verify_model",
        "boundary_enter",
        "autotokenizer_load",
        "boundary_exit",
    ]
    assert loaded[0][1] == {"local_files_only": True, "trust_remote_code": False}
    assert runtime.identity["template_sha256"] == digest(template)
    assert runtime.identity["trust_remote_code"] is False


# --------------------------------------------------------------------------- #
# _serve wire: order, once-only, malformed, post-error stop, shutdown
# --------------------------------------------------------------------------- #


def test_serve_event_order_and_shutdown():
    events, emit = emitter()
    loaded = []

    def load(payload):
        loaded.append(payload)
        return FakeRuntime(payload)

    result = w._serve(
        iter(
            [
                line("load", {"mode": "encoder", "root": "/r", "model": None}),
                line("generate", {"message_id": "m1", "text": "hi"}, row_id="r1"),
                line("generate", {"message_id": "m2", "text": "yo"}, row_id="r2"),
                line("shutdown"),
            ]
        ),
        emit,
        load,
    )
    assert result == 0
    assert len(loaded) == 1
    assert [e["kind"] for e in events] == [
        "ready",
        "generation_started",
        "fragment",
        "terminal",
        "generation_started",
        "fragment",
        "terminal",
    ]
    assert events[0]["row_id"] is None
    assert events[0]["payload"] == {"mode": "encoder", "fictional": True}
    assert events[1]["row_id"] == "r1"
    assert events[1]["payload"] == {"operation": "encoder_op", "qwen_generation": False}
    assert events[4]["row_id"] == "r2"
    assert [e["row_id"] for e in events[2:4]] == ["r1", "r1"]
    assert [e["row_id"] for e in events[5:7]] == ["r2", "r2"]


def test_serve_rejects_repeated_load():
    events, emit = emitter()
    loaded = []

    def load(payload):
        loaded.append(payload)
        return FakeRuntime(payload)

    result = w._serve(
        iter(
            [
                line("load", {"mode": "encoder", "root": "/r", "model": None}),
                line("load", {"mode": "tokenizer", "root": "/r", "model": {}}),
            ]
        ),
        emit,
        load,
    )
    assert result == 1
    assert len(loaded) == 1
    assert events[-1]["kind"] == "error"
    assert "loaded only once" in events[-1]["payload"]["message"]
    assert "traceback" in events[-1]["payload"]


def test_serve_rejects_repeated_row():
    events, emit = emitter()
    generated = []

    class Recording(FakeRuntime):
        def generate(self, payload, emit):
            generated.append(payload)
            super().generate(payload, emit)

    def load(payload):
        return Recording(payload)

    result = w._serve(
        iter(
            [
                line("load", {"mode": "encoder", "root": "/r", "model": None}),
                line("generate", {"message_id": "m1", "text": "hi"}, row_id="r1"),
                line("generate", {"message_id": "m1", "text": "hi"}, row_id="r1"),
            ]
        ),
        emit,
        load,
    )
    assert result == 1
    assert len(generated) == 1
    assert events[-1]["kind"] == "error"
    assert events[-1]["row_id"] == "r1"
    assert "retries forbidden" in events[-1]["payload"]["message"]


def test_serve_surfaces_load_failure_and_missing_row_id():
    events, emit = emitter()

    def load(payload):
        raise ValueError(f"unsupported preparation mode: {payload.get('mode')!r}")

    result = w._serve(
        iter([line("load", {"mode": "qwen", "root": "/r", "model": None})]),
        emit,
        load,
    )
    assert result == 1
    assert events[-1]["kind"] == "error"
    assert "unsupported preparation mode" in events[-1]["payload"]["message"]

    events, emit = emitter()
    result = w._serve(
        iter(
            [
                line("load", {"mode": "encoder", "root": "/r", "model": None}),
                line("generate", {"message_id": "m1", "text": "hi"}),
            ]
        ),
        emit,
        lambda payload: FakeRuntime(payload),
    )
    assert result == 1
    assert events[-1]["kind"] == "error"
    assert "row ID" in events[-1]["payload"]["message"]


def test_serve_stops_after_error_and_never_processes_later_rows():
    events, emit = emitter()
    generated = []

    class Recording(FakeRuntime):
        def generate(self, payload, emit):
            generated.append(payload)
            super().generate(payload, emit)

    def load(payload):
        return Recording(payload)

    result = w._serve(
        iter(
            [
                line("load", {"mode": "encoder", "root": "/r", "model": None}),
                line("generate", {"message_id": "m1", "text": "hi"}, row_id="r1"),
                line("generate", {"message_id": "bad", "text": "no-row-id"}),
                line("generate", {"message_id": "m2", "text": "later"}, row_id="r2"),
            ]
        ),
        emit,
        load,
    )
    assert result == 1
    assert [payload["message_id"] for payload in generated] == ["m1"]
    assert events[-1]["kind"] == "error"
    assert events[-1]["row_id"] is None


def test_serve_requires_protocol_version():
    events, emit = emitter()
    result = w._serve(
        iter([b'{"protocol":"other","kind":"shutdown"}\n']),
        emit,
        lambda payload: FakeRuntime(payload),
    )
    assert result == 1
    assert events[-1]["kind"] == "error"
    assert "protocol version" in events[-1]["payload"]["message"]


# --------------------------------------------------------------------------- #
# main(): lifeline, dedicated protocol FD, real emit path
# --------------------------------------------------------------------------- #


def test_main_wires_lifeline_protocol_and_serve(monkeypatch):
    class Output(io.BytesIO):
        def close(self):
            pass

    output = Output()
    watched = []
    monkeypatch.setattr(sys, "argv", ["private-worker"])
    monkeypatch.setattr(
        sys,
        "stdin",
        NS(
            buffer=io.BytesIO(
                b"".join(
                    [
                        line("load", {"mode": "encoder", "root": "/r", "model": None}),
                        line(
                            "generate", {"message_id": "m1", "text": "hi"}, row_id="r1"
                        ),
                        line("shutdown"),
                    ]
                )
            )
        ),
    )
    monkeypatch.setenv("NLP3_PROTOCOL_FD", "8")
    monkeypatch.setenv("NLP3_LIFELINE_FD", "9")
    monkeypatch.setattr(w.os, "fdopen", lambda *a, **kw: output)
    original_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name == "messaging_nlp.run_supervision":
            return NS(watch_parent=lambda *args: watched.append(args))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)
    monkeypatch.setattr(w, "load_runtime", lambda payload: FakeRuntime(payload))
    assert w.main() == 0
    assert watched == [(9, 8)]
    events = [loads_json(line) for line in output.getvalue().splitlines()]
    assert [e["kind"] for e in events] == [
        "ready",
        "generation_started",
        "fragment",
        "terminal",
    ]
    assert events[0]["protocol"] == PROTOCOL_VERSION
    assert events[1]["row_id"] == "r1"


# --------------------------------------------------------------------------- #
# Source boundary and zero-Qwen guarantees
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mutation", ["bytes", "inventory"])
def test_child_installed_source_must_match_frozen_project(
    tmp_path, monkeypatch, mutation
):
    package = tmp_path / "installed"
    frozen = tmp_path / "project/src/messaging_nlp"
    package.mkdir()
    frozen.mkdir(parents=True)
    for name in ("preparation_worker.py", "__init__.py"):
        (package / name).write_bytes(b"fictional source")
        (frozen / name).write_bytes(b"fictional source")
    monkeypatch.setattr(w, "__file__", str(package / "preparation_worker.py"))
    assert set(w._verify_worker_source(tmp_path / "project")) == {
        "preparation_worker.py",
        "__init__.py",
    }
    if mutation == "bytes":
        (frozen / "__init__.py").write_bytes(b"changed source")
    else:
        (frozen / "extra.py").write_bytes(b"extra source")
    with pytest.raises(ValueError, match="source"):
        w._verify_worker_source(tmp_path / "project")


def test_zero_qwen_invocation_contract():
    # Preparation reuses the wire for supervision only: it must never import or
    # expose the Qwen generation runtime, and every operation start records
    # qwen_generation=False (asserted per-row in the generate tests above).
    assert "QwenRuntime" not in dir(w)
    assert not any("stream_generate" in name for name in dir(w))
    assert w.DENIED_RUNTIME_IMPORTS >= frozenset({"mlx", "mlx_lm", "torch"})
    assert w.TOKEN_CEILING == 2048


def test_module_import_loads_no_optional_runtime():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'mlx', 'mlx_lm', 'safetensors', 'tokenizers',
        }:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
import messaging_nlp.preparation_worker as w
assert w.TOKEN_CEILING == 2048
assert 'QwenRuntime' not in dir(w)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
