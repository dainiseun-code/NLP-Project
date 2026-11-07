"""Private persistent isolated Qwen child. Importing this module loads no runtime.

Only the coordinator launches this module, after its authorization/resource gates.
Protocol travels on an inherited descriptor, never incidental library stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

from messaging_nlp.encoder_cache import portable_path
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.limits import MAX_INPUT_TOKENS
from messaging_nlp.run_contracts import (
    PROTOCOL_VERSION,
    QWEN_RECEIPT_SHA256,
    QWEN_REPO,
    QWEN_REVISION,
    DecodePolicy,
    ModelSpec,
    TokenReceipt,
    WorkerEvent,
    WorkerRequest,
)

RUNTIME_VERSIONS = {
    "mlx": "0.32.4",
    "mlx-lm": "0.32.2",
    "transformers": "5.20.2",
    "pydantic": "2.14.1",
}
SAMPLER_SETTINGS = {
    "temp": 0,
    "top_p": 0,
    "min_p": 0,
    "top_k": 0,
    "min_tokens_to_keep": 1,
    "xtc_probability": 0,
    "xtc_threshold": 0.1,
    "xtc_special_tokens": [],
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _local(root, relative):
    portable_path(relative)
    path = root / relative
    for part in [path, *path.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("model/receipt symlinks are not permitted")
    if not path.resolve().is_relative_to(root):
        raise ValueError("model path escapes root")
    return path


def _file_sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _verify_worker_source(root):
    """Bind the isolated child's installed package, not only the parent's copy."""
    package = Path(__file__).resolve().parent
    expected = Path(root).resolve(strict=True) / "src/messaging_nlp"
    actual_files = {p.relative_to(package).as_posix(): p for p in package.rglob("*.py")}
    expected_files = {
        p.relative_to(expected).as_posix(): p for p in expected.rglob("*.py")
    }
    if not actual_files or set(actual_files) != set(expected_files):
        raise ValueError("isolated worker source inventory differs from frozen project")
    checked = {}
    for name, path in actual_files.items():
        if _file_sha(path) != _file_sha(expected_files[name]):
            raise ValueError("isolated worker source bytes differ from frozen project")
        checked[name] = _file_sha(path)
    return checked


def verify_model(root, model: ModelSpec) -> dict:
    """Recheck pinned trust anchor independently of the caller's self-reported hash.

    Finder metadata is ignored without opening it. Conservatively reject even
    nonescaping symlinks; nothing authorizes following external model identities.
    """
    model = ModelSpec.model_validate(model)
    root = Path(root).resolve(strict=True)
    receipt_path = _local(root, model.receipt.path)
    raw = receipt_path.read_bytes()
    if (
        model.receipt.sha256 != QWEN_RECEIPT_SHA256
        or _sha(raw) != QWEN_RECEIPT_SHA256
        or len(raw) != model.receipt.bytes
    ):
        raise ValueError("untrusted Qwen receipt")
    receipt = json.loads(raw)
    if (
        receipt["repo"] != QWEN_REPO
        or receipt["revision"] != QWEN_REVISION
        or receipt["directory"] != model.path
    ):
        raise ValueError("Qwen repository/revision/directory mismatch")
    directory = _local(root, model.path)
    if not directory.is_dir():
        raise ValueError("missing local model directory")
    expected = {}
    for entry in receipt["files"]:
        name = entry["file"]
        portable_path(name)
        if name in expected or entry.get("verified") is not True:
            raise ValueError("invalid receipt file identity")
        expected[name] = entry
    if not expected:
        raise ValueError("empty model receipt")
    # Local hf downloads retain inert bookkeeping outside the model inventory.
    # Enumerate only this receipt's exact names/revision; never read or trust it.
    bookkeeping = {
        f".cache/huggingface/download/{name}.{suffix}"
        for name in expected
        for suffix in ("lock", "metadata")
    } | {
        ".cache/huggingface/.gitignore",
        ".cache/huggingface/CACHEDIR.TAG",
        f".cache/huggingface/trees/{QWEN_REVISION}.json",
    }
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("model symlink is forbidden")
        if path.is_dir():
            continue
        if path.name == ".DS_Store" or path.name.startswith("._"):
            continue
        name = path.relative_to(directory).as_posix()
        if name in bookkeeping:
            if not path.is_file():
                raise ValueError(
                    "download bookkeeping must be a regular non-symlink file"
                )
            continue
        actual.add(name)
        if name not in expected or not path.is_file():
            raise ValueError("unexpected model file")
        entry = expected[name]
        if path.stat().st_size != entry["bytes"] or _file_sha(path) != entry["sha256"]:
            raise ValueError(f"model file identity mismatch: {name}")
    if actual != set(expected):
        raise ValueError("missing model files")
    if expected.get("chat_template.jinja", {}).get("sha256") != model.template_sha256:
        raise ValueError("model template identity mismatch")
    return {
        "repo": QWEN_REPO,
        "revision": QWEN_REVISION,
        "receipt_sha256": QWEN_RECEIPT_SHA256,
        "files": expected,
        "path": model.path,
        "template_sha256": model.template_sha256,
    }


def chat_tokens(tokenizer, messages) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
        truncation=False,
        padding=False,
    )
    if (
        not isinstance(tokens, list)
        or not tokens
        or len(tokens) > MAX_INPUT_TOKENS
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("chat tokens must be a nonempty flat integer list <=2048")
    return tokens


def quantization_audit(model, config, nn) -> dict:
    """Compare every actual quantized module against its per-module declaration."""
    quant = config["quantization"]
    if "quantization_config" in config and config["quantization_config"] != quant:
        raise ValueError("conflicting quantization declarations")
    expected = {}
    for name, settings in quant.items():
        if isinstance(settings, dict):
            expected[name] = {
                "bits": settings.get("bits", quant["bits"]),
                "group_size": settings.get("group_size", quant["group_size"]),
                "mode": settings.get("mode", quant.get("mode", "affine")),
            }
    actual = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            actual[name] = {
                "bits": module.bits,
                "group_size": module.group_size,
                "mode": module.mode,
            }
    if not expected or actual != expected:
        raise ValueError("actual mixed quantization module/group/mode map differs")
    if {item["bits"] for item in actual.values()} != {4, 8}:
        raise ValueError("expected mixed 4/8-bit model")
    return {"modules": actual, "sha256": _sha(canonical_json(actual))}


class QwenRuntime:
    def __init__(self, root, model, decoding, source_sha256=None):
        self.decoding = DecodePolicy.model_validate(decoding)
        spec = ModelSpec.model_validate(model)
        if spec.runtime_versions != RUNTIME_VERSIONS:
            raise ValueError("model spec cannot override pinned runtime versions")
        actual_versions = {name: version(name) for name in RUNTIME_VERSIONS}
        if actual_versions != RUNTIME_VERSIONS:
            raise ValueError("installed Qwen runtime versions differ from pins")
        source = _verify_worker_source(root)
        if source != source_sha256:
            raise ValueError(
                "isolated worker source differs from sealed parent identities"
            )
        self.identity = verify_model(root, spec)
        self.identity["worker_source_sha256"] = source
        directory = Path(root).resolve(strict=True) / spec.path
        self.template_sha256 = spec.template_sha256
        # These imports are deliberately inside the load request only.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.generate import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import load

        mx.set_memory_limit(3 * 1024**3)
        mx.set_cache_limit(128 * 1024**2)
        self.mx = mx
        self.stream = stream_generate
        self.make_cache = make_prompt_cache
        self.make_sampler = make_sampler
        self.model, self.tokenizer, config = load(
            str(directory),
            tokenizer_config={"local_files_only": True, "trust_remote_code": False},
            adapter_path=None,
            lazy=False,
            return_config=True,
            trust_remote_code=False,
        )
        self.model.eval()
        retained = json.loads((directory / "config.json").read_bytes())
        if config["quantization"] != retained["quantization"]:
            raise ValueError("loader changed retained quantization map")
        audit = quantization_audit(self.model, config, nn)
        template = self.tokenizer.chat_template
        if (
            not isinstance(template, str)
            or _sha(template.encode()) != spec.template_sha256
        ):
            raise ValueError("loaded tokenizer template differs from pinned bytes")
        self.identity.update(
            {
                "runtime_versions": actual_versions,
                "quantization": audit,
                "decoding": self.decoding.model_dump(mode="json"),
                "memory_limit_bytes": 3 * 1024**3,
                "cache_limit_bytes": 128 * 1024**2,
                "local_files_only": True,
                "trust_remote_code": False,
                "adapter": None,
                "draft_model": None,
                "vision_execution": False,
                "sampler": SAMPLER_SETTINGS.copy(),
                "logits_processors": [],
                "kv_bits": None,
                "max_kv_size": None,
                "prefill_step_size": 128,
            }
        )

    def generate(self, payload, emit):
        receipt = TokenReceipt.model_validate(payload["token_receipt"])
        messages = payload["messages"]
        tokens = chat_tokens(self.tokenizer, messages)
        template = self.tokenizer.chat_template
        if (
            not isinstance(template, str)
            or _sha(template.encode()) != self.template_sha256
            or receipt.template_sha256 != self.template_sha256
            or receipt.model_receipt_sha256 != QWEN_RECEIPT_SHA256
            or self.identity["receipt_sha256"] != QWEN_RECEIPT_SHA256
            or receipt.messages_sha256 != _sha(canonical_json(messages))
            or receipt.token_ids_sha256 != _sha(canonical_json(tokens))
            or receipt.token_ids != tokens
            or receipt.token_count != len(tokens)
        ):
            raise ValueError("stale or mismatched complete token receipt")
        self.mx.clear_cache()
        self.mx.random.seed(self.decoding.seed)
        cache = self.make_cache(self.model)
        sampler = self.make_sampler(**SAMPLER_SETTINGS)
        processor = None
        started = time.monotonic()
        terminal = None
        emit("generation_started", {})
        try:
            if payload.get("output_schema") is not None:
                from messaging_nlp.structured_decoding import build_processor

                processor = build_processor(
                    self.tokenizer,
                    loads_json(payload["output_schema"])
                    if isinstance(payload["output_schema"], str)
                    else payload["output_schema"],
                    self.model.args.text_config["vocab_size"],
                )
            for response in self.stream(
                self.model,
                self.tokenizer,
                prompt=tokens,
                max_tokens=self.decoding.max_output_tokens,
                sampler=sampler,
                logits_processors=[] if processor is None else [processor],
                prompt_cache=cache,
                max_kv_size=None,
                kv_bits=None,
                draft_model=None,
                prefill_step_size=128,
            ):
                if terminal is not None:
                    raise RuntimeError("runtime yielded after terminal response")
                count = response.generation_tokens
                if (
                    type(count) is not int
                    or not 1 <= count <= self.decoding.max_output_tokens
                ):
                    raise RuntimeError("invalid runtime progress token count")
                if response.text:
                    emit(
                        "fragment", {"text": response.text, "generation_tokens": count}
                    )
                else:
                    emit("progress", {"generation_tokens": count})
                if response.finish_reason is not None:
                    if (
                        response.finish_reason not in ("stop", "length")
                        or response.prompt_tokens != len(tokens)
                        or type(response.generation_tokens) is not int
                        or not 1
                        <= response.generation_tokens
                        <= self.decoding.max_output_tokens
                    ):
                        raise RuntimeError("invalid runtime terminal metadata")
                    terminal = {
                        "finish_reason": response.finish_reason,
                        "generation_tokens": response.generation_tokens,
                        "input_tokens": len(tokens),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    if processor is not None:
                        terminal["structured_decoding_audit"] = processor.audit()
            if terminal is None:
                raise RuntimeError("generation ended without runtime terminal")
            emit("terminal", terminal)
        finally:
            del cache
            self.mx.clear_cache()


def main():
    """No public flags or substitution: inherited private protocol/lifeline only."""
    if len(sys.argv) != 1:
        raise RuntimeError("private Qwen worker takes no arguments")
    protocol_fd = int(os.environ["NLP3_PROTOCOL_FD"])
    lifeline_fd = int(os.environ["NLP3_LIFELINE_FD"])
    if protocol_fd < 3 or lifeline_fd < 3 or protocol_fd == lifeline_fd:
        raise RuntimeError("dedicated protocol and lifeline descriptors required")
    from messaging_nlp.run_supervision import watch_parent

    watch_parent(lifeline_fd, protocol_fd)
    runtime = None
    seen_rows = set()
    with os.fdopen(protocol_fd, "wb", buffering=0) as protocol:
        for line in sys.stdin.buffer:
            row_id = None

            def emit(kind, payload):
                event = WorkerEvent(kind=kind, row_id=row_id, payload=payload)
                raw = canonical_json(event.model_dump(mode="json")) + b"\n"
                view = memoryview(raw)
                while view:
                    written = protocol.write(view)
                    if not written:
                        raise BrokenPipeError("protocol descriptor closed")
                    view = view[written:]

            try:
                wire = loads_json(line.decode("utf-8"))
                if (
                    not isinstance(wire, dict)
                    or wire.get("protocol") != PROTOCOL_VERSION
                ):
                    raise ValueError("explicit current protocol version required")
                request = WorkerRequest.model_validate(wire)
                row_id = request.row_id
                if request.kind == "shutdown":
                    return
                if request.kind == "load":
                    if runtime is not None:
                        raise RuntimeError("model may be loaded only once")
                    runtime = QwenRuntime(**request.payload)
                    emit("ready", runtime.identity)
                else:
                    if runtime is None or not row_id:
                        raise RuntimeError("generate requires loaded model and row ID")
                    if row_id in seen_rows:
                        raise RuntimeError("row already dispatched; retries forbidden")
                    seen_rows.add(row_id)
                    runtime.generate(request.payload, emit)
            except Exception as exc:
                emit("error", {"message": f"{type(exc).__name__}: {exc}"})
                return 1  # Fail closed; no implicit retries/restarts or next row.


if __name__ == "__main__":
    sys.exit(main())
