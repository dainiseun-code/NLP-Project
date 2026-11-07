"""Contract-derived fictional tests, written before adapter implementation."""

import builtins
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from messaging_nlp import qwen_worker as w
from messaging_nlp.io import canonical_json
from messaging_nlp.run_contracts import (
    PROTOCOL_VERSION,
    DecodePolicy,
    ModelSpec,
    TokenReceipt,
)


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


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Tokenizer:
    chat_template = "fictional template"

    def __init__(self, result=None):
        self.result = [7, 8] if result is None else result
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.result


@pytest.mark.parametrize(
    "invalid", [{"input_ids": [1]}, [[1]], [], [True], [1.0], [-1], "one"]
)
def test_chat_rejects_invalid_ids(invalid):
    with pytest.raises(ValueError):
        w.chat_tokens(Tokenizer(invalid), [])


def test_chat_flags_and_ceiling():
    tokenizer = Tokenizer()
    assert w.chat_tokens(tokenizer, []) == [7, 8]
    assert tokenizer.calls[0][1] == dict(
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
        truncation=False,
        padding=False,
    )
    with pytest.raises(ValueError):
        w.chat_tokens(Tokenizer([1] * 2049), [])


@pytest.mark.parametrize("mutation", ["bytes", "inventory"])
def test_child_installed_source_must_match_frozen_project_before_load(
    tmp_path, monkeypatch, mutation
):
    package = tmp_path / "installed"
    frozen = tmp_path / "project/src/messaging_nlp"
    package.mkdir()
    frozen.mkdir(parents=True)
    for name in ("qwen_worker.py", "__init__.py"):
        (package / name).write_bytes(b"fictional source")
        (frozen / name).write_bytes(b"fictional source")
    monkeypatch.setattr(w, "__file__", str(package / "qwen_worker.py"))
    assert set(w._verify_worker_source(tmp_path / "project")) == {
        "qwen_worker.py",
        "__init__.py",
    }
    if mutation == "bytes":
        (frozen / "__init__.py").write_bytes(b"changed source")
    else:
        (frozen / "extra.py").write_bytes(b"extra source")
    with pytest.raises(ValueError, match="source"):
        w._verify_worker_source(tmp_path / "project")


@pytest.mark.parametrize("expected", [None, {"wrong.py": "0" * 64}])
def test_child_requires_sealed_parent_source_hashes_before_model_open(
    monkeypatch, expected
):
    spec = ModelSpec(
        path="unused",
        receipt={"path": "unused", "sha256": "0" * 64, "bytes": 0},
        template_sha256="0" * 64,
    )
    monkeypatch.setattr(w, "version", lambda name: w.RUNTIME_VERSIONS[name])
    monkeypatch.setattr(
        w, "_verify_worker_source", lambda root: {"actual.py": "0" * 64}
    )

    def no_model(*args):
        raise AssertionError("model verification must not precede source binding")

    monkeypatch.setattr(w, "verify_model", no_model)
    with pytest.raises(ValueError, match="sealed parent"):
        w.QwenRuntime("unused", spec, DecodePolicy(), source_sha256=expected)


@pytest.fixture
def fictional_model(tmp_path, monkeypatch):
    directory = tmp_path / "fictional"
    directory.mkdir()
    content = b"fictional template"
    (directory / "chat_template.jinja").write_bytes(content)
    receipt = {
        "repo": w.QWEN_REPO,
        "revision": w.QWEN_REVISION,
        "directory": "fictional",
        "files": [
            {
                "file": "chat_template.jinja",
                "bytes": len(content),
                "sha256": digest(content),
                "verified": True,
            }
        ],
    }
    raw = canonical_json(receipt)
    (tmp_path / "receipt.json").write_bytes(raw)
    monkeypatch.setattr(w, "QWEN_RECEIPT_SHA256", digest(raw))
    spec = ModelSpec(
        path="fictional",
        receipt={"path": "receipt.json", "sha256": digest(raw), "bytes": len(raw)},
        template_sha256=digest(content),
    )
    return tmp_path, spec


def test_model_identity_and_finder(fictional_model):
    root, spec = fictional_model
    (root / "fictional/.DS_Store").write_bytes(b"ignored")
    assert w.verify_model(root, spec)["receipt_sha256"] == spec.receipt.sha256
    (root / "fictional/chat_template.jinja").write_bytes(b"changed")
    with pytest.raises(ValueError):
        w.verify_model(root, spec)


def test_known_download_bookkeeping_is_enumerated_but_never_opened(
    fictional_model, monkeypatch
):
    root, spec = fictional_model
    directory = root / "fictional"
    names = {
        ".cache/huggingface/download/chat_template.jinja.lock",
        ".cache/huggingface/download/chat_template.jinja.metadata",
        ".cache/huggingface/.gitignore",
        ".cache/huggingface/CACHEDIR.TAG",
        f".cache/huggingface/trees/{w.QWEN_REVISION}.json",
    }
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque ignored bookkeeping, not an identity authority")
    original = io.open

    def no_bookkeeping(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and ".cache" in Path(file).parts:
            raise AssertionError("download bookkeeping must not be opened")
        return original(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", no_bookkeeping)
    assert w.verify_model(root, spec)["receipt_sha256"] == spec.receipt.sha256


@pytest.mark.parametrize("kind", ["unknown", "wrong_revision", "symlink"])
def test_download_bookkeeping_exception_cannot_hide_unknown_files_or_symlinks(
    fictional_model, kind
):
    root, spec = fictional_model
    name = {
        "unknown": "download/unlisted.json",
        "wrong_revision": "trees/not-pinned.json",
        "symlink": "download/chat_template.jinja.lock",
    }[kind]
    path = root / "fictional/.cache/huggingface" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "symlink":
        path.symlink_to(root / "receipt.json")
    else:
        path.write_bytes(b"not an allowed bookkeeping name")
    with pytest.raises(ValueError):
        w.verify_model(root, spec)


def test_receipt_cannot_self_authorize(fictional_model):
    root, spec = fictional_model
    raw = (root / "receipt.json").read_bytes() + b" "
    (root / "receipt.json").write_bytes(raw)
    forged = spec.model_copy(
        update={
            "receipt": spec.receipt.model_copy(
                update={"sha256": digest(raw), "bytes": len(raw)}
            )
        }
    )
    with pytest.raises(ValueError):
        w.verify_model(root, forged)


@pytest.mark.parametrize("escape", [False, True])
def test_unexpected_files(fictional_model, escape):
    root, spec = fictional_model
    extra = root / "fictional/extra"
    if escape:
        extra.symlink_to(root / "receipt.json")
    else:
        extra.write_bytes(b"unlisted")
    with pytest.raises(ValueError):
        w.verify_model(root, spec)


def test_expected_file_symlink_and_missing_file_fail(fictional_model):
    root, spec = fictional_model
    path = root / "fictional/chat_template.jinja"
    path.unlink()
    with pytest.raises(ValueError):
        w.verify_model(root, spec)
    (root / "outside").write_bytes(b"fictional template")
    path.symlink_to(root / "outside")
    with pytest.raises(ValueError):
        w.verify_model(root, spec)


class Quantized:
    def __init__(self, bits, group_size=64, mode="affine"):
        self.bits, self.group_size, self.mode = bits, group_size, mode


def test_mixed_quantization_exact_mapping():
    nn = NS(QuantizedLinear=Quantized, QuantizedEmbedding=Quantized)
    modules = {"embed": Quantized(8), "layer": Quantized(4)}
    model = NS(named_modules=lambda: list(modules.items()))
    config = {
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "embed": {"bits": 8, "group_size": 64},
            "layer": {"bits": 4, "group_size": 64},
        }
    }
    assert w.quantization_audit(model, config, nn)["modules"]["embed"]["bits"] == 8
    modules["embed"].bits = 4
    with pytest.raises(ValueError):
        w.quantization_audit(model, config, nn)
    modules["embed"].bits = 8
    modules["extra"] = Quantized(4)
    with pytest.raises(ValueError):
        w.quantization_audit(model, config, nn)


@pytest.mark.parametrize("attribute,value", [("group_size", 128), ("mode", "mxfp4")])
def test_quantization_group_and_mode_not_only_bits(attribute, value):
    nn = NS(QuantizedLinear=Quantized, QuantizedEmbedding=Quantized)
    modules = {"a": Quantized(4), "b": Quantized(8)}
    setattr(modules["b"], attribute, value)
    config = {
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "a": {"bits": 4},
            "b": {"bits": 8},
        }
    }
    with pytest.raises(ValueError):
        w.quantization_audit(NS(named_modules=lambda: modules.items()), config, nn)


@pytest.fixture
def runtime():
    runtime = object.__new__(w.QwenRuntime)
    runtime.tokenizer = Tokenizer()
    runtime.decoding = DecodePolicy()
    runtime.model = object()
    runtime.template_sha256 = digest(b"fictional template")
    runtime.identity = {"receipt_sha256": w.QWEN_RECEIPT_SHA256}
    runtime.mx = NS(random=NS(seed=lambda seed: None), clear_cache=lambda: None)
    runtime.make_cache = lambda model: []
    runtime.make_sampler = lambda **kwargs: kwargs
    runtime.stream = lambda *args, **kwargs: iter(
        [
            NS(
                text="partial", finish_reason=None, generation_tokens=1, prompt_tokens=2
            ),
            NS(text="end", finish_reason="stop", generation_tokens=2, prompt_tokens=2),
        ]
    )
    return runtime


def payload(runtime):
    messages = [{"role": "user", "content": "fictional"}]
    receipt = TokenReceipt(
        messages_sha256=digest(canonical_json(messages)),
        template_sha256=runtime.template_sha256,
        model_receipt_sha256=w.QWEN_RECEIPT_SHA256,
        token_ids_sha256=digest(canonical_json([7, 8])),
        token_count=2,
        token_ids=[7, 8],
    )
    return {"messages": messages, "token_receipt": receipt.model_dump(mode="json")}


def test_schema_processor_is_explicit_per_call_and_audited(runtime, monkeypatch):
    from messaging_nlp import structured_decoding

    runtime.model = NS(args=NS(text_config={"vocab_size": 32}))
    schema = {"type": "object"}
    processor = NS(audit=lambda: {"postvalidation_required": True})
    monkeypatch.setattr(
        structured_decoding,
        "build_processor",
        lambda tokenizer, actual, size: (
            processor if actual == schema and size == 32 else None
        ),
    )
    original = runtime.stream
    calls = []
    runtime.stream = lambda *args, **kwargs: (
        calls.append(kwargs) or original(*args, **kwargs)
    )
    request = payload(runtime)
    request["output_schema"] = schema
    events = []
    runtime.generate(request, lambda kind, value: events.append((kind, value)))
    assert calls[0]["logits_processors"] == [processor]
    assert events[-1][1]["structured_decoding_audit"] == {
        "postvalidation_required": True
    }


def test_schema_compile_failure_still_clears_fresh_cache(runtime, monkeypatch):
    from messaging_nlp import structured_decoding

    runtime.model = NS(args=NS(text_config={"vocab_size": 32}))
    cleared = []
    runtime.mx.clear_cache = lambda: cleared.append(True)

    def fail(*args):
        raise ValueError("fictional unsupported grammar")

    monkeypatch.setattr(structured_decoding, "build_processor", fail)
    request = payload(runtime)
    request["output_schema"] = {"type": "object"}
    with pytest.raises(ValueError, match="unsupported grammar"):
        runtime.generate(request, lambda *args: None)
    assert len(cleared) == 2


@pytest.mark.parametrize(
    "field",
    [
        "messages_sha256",
        "template_sha256",
        "model_receipt_sha256",
        "token_ids_sha256",
        "token_count",
    ],
)
def test_stale_tokens_never_start(runtime, field):
    request = payload(runtime)
    request["token_receipt"][field] = 3 if field == "token_count" else "0" * 64
    events = []
    with pytest.raises(ValueError):
        runtime.generate(request, lambda kind, value: events.append((kind, value)))
    assert events == []


def test_fresh_cache_explicit_sampler_and_event_order(runtime):
    caches, calls, events = [], [], []
    runtime.make_cache = lambda model: caches.append([]) or caches[-1]
    original = runtime.stream

    def stream(*args, **kwargs):
        calls.append(kwargs)
        assert events[-1][0] == "generation_started"
        return original(*args, **kwargs)

    runtime.stream = stream
    for _ in range(2):
        runtime.generate(
            payload(runtime), lambda kind, value: events.append((kind, value))
        )
    assert caches[0] is not caches[1]
    assert calls[0]["prompt_cache"] is caches[0]
    assert calls[0]["kv_bits"] is None
    assert calls[0]["draft_model"] is None
    assert calls[0]["max_tokens"] == 1024
    assert calls[0]["logits_processors"] == []
    assert calls[0]["prefill_step_size"] == 128
    assert calls[0]["sampler"]["temp"] == 0
    assert [k for k, _ in events[:4]] == [
        "generation_started",
        "fragment",
        "fragment",
        "terminal",
    ]


def test_length_is_not_stop(runtime):
    runtime.stream = lambda *a, **kw: iter(
        [NS(text="{}", finish_reason="length", generation_tokens=256, prompt_tokens=2)]
    )
    events = []
    runtime.generate(payload(runtime), lambda k, v: events.append((k, v)))
    assert events[-1][1]["finish_reason"] == "length"


def test_missing_terminal_preserves_partial(runtime):
    runtime.stream = lambda *a, **kw: iter(
        [NS(text="partial", finish_reason=None, generation_tokens=1, prompt_tokens=2)]
    )
    events = []
    with pytest.raises(RuntimeError):
        runtime.generate(payload(runtime), lambda k, v: events.append((k, v)))
    assert [k for k, _ in events] == ["generation_started", "fragment"]


def test_runtime_exception_is_not_a_terminal_success(runtime):
    def stream(*args, **kwargs):
        yield NS(text="partial", finish_reason=None, generation_tokens=1)
        raise RuntimeError("fictional runtime failure")

    runtime.stream = stream
    events = []
    with pytest.raises(RuntimeError, match="fictional runtime"):
        runtime.generate(payload(runtime), lambda k, v: events.append((k, v)))
    assert [k for k, _ in events] == ["generation_started", "fragment"]


def test_load_uses_only_local_pinned_kwargs_and_ready_identity(tmp_path, monkeypatch):
    """All import targets below are plain stubs, never installed runtime modules."""
    (tmp_path / "fake").mkdir()
    config = {"quantization": {"bits": 4}}
    (tmp_path / "fake/config.json").write_text(json.dumps(config))
    template_sha = digest(b"fictional template")
    spec = ModelSpec(
        path="fake",
        receipt={"path": "unused", "sha256": "0" * 64, "bytes": 0},
        template_sha256=template_sha,
    )
    monkeypatch.setattr(
        w, "verify_model", lambda *a: {"receipt_sha256": w.QWEN_RECEIPT_SHA256}
    )
    monkeypatch.setattr(w, "quantization_audit", lambda *a: {"modules": "stub"})
    monkeypatch.setattr(
        w, "_verify_worker_source", lambda root: {"fictional.py": "1" * 64}
    )
    monkeypatch.setattr(w, "version", lambda name: w.RUNTIME_VERSIONS[name])
    calls = []
    mx = NS(
        set_memory_limit=lambda n: calls.append(("memory", n)),
        set_cache_limit=lambda n: calls.append(("cache", n)),
    )
    model = NS(eval=lambda: calls.append(("eval",)))

    def load(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return model, Tokenizer(), config

    stubs = {
        "mlx.core": NS(core=mx),
        "mlx.nn": NS(nn=NS()),
        "mlx_lm.generate": NS(stream_generate=object()),
        "mlx_lm.models.cache": NS(make_prompt_cache=object()),
        "mlx_lm.sample_utils": NS(make_sampler=object()),
        "mlx_lm.utils": NS(load=load),
    }
    guarded_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        return stubs[name] if name in stubs else guarded_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)
    runtime = w.QwenRuntime(
        tmp_path, spec, DecodePolicy(), source_sha256={"fictional.py": "1" * 64}
    )
    assert calls[:2] == [("memory", 3 * 1024**3), ("cache", 128 * 1024**2)]
    assert calls[2][2] == {
        "tokenizer_config": {"local_files_only": True, "trust_remote_code": False},
        "adapter_path": None,
        "lazy": False,
        "return_config": True,
        "trust_remote_code": False,
    }
    assert calls[2][1] == (str(tmp_path / "fake"),)
    assert runtime.identity["kv_bits"] is None
    assert (
        runtime.identity["decoding"]["cache_policy"]
        == "fresh_normal_cache_per_row_no_kv_quantization"
    )
    assert runtime.identity["runtime_versions"] == w.RUNTIME_VERSIONS


@pytest.mark.parametrize("forged_spec", [False, True])
def test_wrong_runtime_pins_fail_before_import_or_file_open(monkeypatch, forged_spec):
    spec = ModelSpec(
        path="unused",
        receipt={"path": "unused", "sha256": "0" * 64, "bytes": 0},
        template_sha256="0" * 64,
    )
    if forged_spec:
        spec = spec.model_copy(update={"runtime_versions": {"mlx": "arbitrary"}})
    monkeypatch.setattr(w, "version", lambda name: "wrong-version")
    with pytest.raises(ValueError, match="runtime"):
        w.QwenRuntime("nonexistent", spec, DecodePolicy())


def test_private_protocol_persistent_load_and_no_retry(monkeypatch):
    class Output(io.BytesIO):
        def close(self):
            pass

    output = Output()
    watched, generated, loaded = [], [], []

    class FakeRuntime:
        identity = {"runtime": "fictional"}

        def __init__(self, **kwargs):
            loaded.append(kwargs)

        def generate(self, payload, emit):
            generated.append(payload)
            emit("generation_started", {})
            emit("fragment", {"text": "fictional"})
            emit("terminal", {"finish_reason": "stop"})

    requests = [
        {"kind": "load", "payload": {}},
        {"kind": "generate", "row_id": "one"},
        {"kind": "generate", "row_id": "two"},
        {"kind": "generate", "row_id": "one"},
        {"kind": "generate", "row_id": "three"},
    ]
    raw = b"".join(
        canonical_json({"protocol": PROTOCOL_VERSION, **v}) + b"\n" for v in requests
    )
    monkeypatch.setattr(sys, "argv", ["private-worker"])
    monkeypatch.setattr(sys, "stdin", NS(buffer=io.BytesIO(raw)))
    monkeypatch.setenv("NLP3_PROTOCOL_FD", "8")
    monkeypatch.setenv("NLP3_LIFELINE_FD", "9")
    monkeypatch.setattr(w.os, "fdopen", lambda *a, **kw: output)
    monkeypatch.setattr(w, "QwenRuntime", FakeRuntime)
    original_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name == "messaging_nlp.run_supervision":
            return NS(watch_parent=lambda *args: watched.append(args))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)
    w.main()
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert watched == [(9, 8)]
    assert len(loaded) == 1 and len(generated) == 2
    assert events[0]["kind"] == "ready"
    assert events[-1]["kind"] == "error"
    assert events[-1]["row_id"] == "one"
    assert "retries forbidden" in events[-1]["payload"]["message"]
