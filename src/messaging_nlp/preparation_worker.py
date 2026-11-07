"""Private persistent isolated preparation child (encoder/tokenizer).

Importing this module loads no model runtime. Only the Stage 5 coordinator
launches it, after its authorization/resource gates. The wire protocol reuses
the same private WorkerRequest/WorkerEvent stream, ChildSession and parent
lifeline as the Qwen child, solely for supervision: preparation reserves zero
generations and never runs Qwen. Optional runtimes stay inside the actual mode
loaders, never at module import and never in the fictional tests.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import io
import os
import sys
import time
import traceback
from pathlib import Path

from messaging_nlp.encoder_cache import ExtractionArtifact, portable_path
from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.qwen_worker import (
    QWEN_RECEIPT_SHA256,
    QWEN_REPO,
    QWEN_REVISION,
    verify_model,
)
from messaging_nlp.run_contracts import (
    PROTOCOL_VERSION,
    ModelSpec,
    WorkerEvent,
    WorkerRequest,
)

TOKEN_CEILING = 2048

# Tokenizer mode must never pull in a tensor runtime. ``torch`` is banned even
# though the encoder child (a separate process) uses it: the two modes never
# share a process.
DENIED_RUNTIME_IMPORTS = frozenset({"mlx", "mlx_lm", "torch", "safetensors"})
WEIGHT_PAYLOAD_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".ggml",
    ".onnx",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _local(root: Path, relative: str) -> Path:
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


def _verify_worker_source(root: Path) -> dict:
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


def _deny_import(name: str) -> None:
    top = name.split(".")[0]
    if top in DENIED_RUNTIME_IMPORTS:
        raise RuntimeError(
            f"optional runtime import forbidden in tokenizer mode: {name}"
        )


def _deny_open(path) -> None:
    if isinstance(path, bytes):
        path = path.decode("utf-8", errors="surrogateescape")
    if isinstance(path, (str, os.PathLike)):
        if Path(path).name.endswith(WEIGHT_PAYLOAD_SUFFIXES):
            raise RuntimeError(
                f"weight payload open forbidden in tokenizer mode: {Path(path)}"
            )


class _TokenizerRuntimeBoundary:
    """Deny optional-runtime imports and weight-payload opens during tokenizer load.

    Installed AFTER model-free identity verification (which hashes known weight
    bytes as identity reading) and BEFORE AutoTokenizer/any optional import, so
    only config/tokenizer/template static-byte checks remain permitted.
    """

    def __enter__(self):
        if any(name.split(".")[0] in DENIED_RUNTIME_IMPORTS for name in sys.modules):
            raise RuntimeError("tensor runtime was already loaded in tokenizer mode")
        self._original_import = builtins.__import__
        self._original_open = builtins.open
        self._original_io_open = io.open
        self._original_os_open = os.open
        self._original_find_spec = importlib.util.find_spec
        self._finder = _DeniedRuntimeFinder()
        sys.meta_path.insert(0, self._finder)
        builtins.__import__ = self._guarded_import
        builtins.open = self._guarded_open
        io.open = self._guarded_io_open
        os.open = self._guarded_os_open
        importlib.util.find_spec = self._availability_spec
        return self

    def __exit__(self, *exc):
        builtins.__import__ = self._original_import
        builtins.open = self._original_open
        io.open = self._original_io_open
        os.open = self._original_os_open
        importlib.util.find_spec = self._original_find_spec
        sys.meta_path.remove(self._finder)

    def _guarded_import(self, name, *args, **kwargs):
        _deny_import(name)
        return self._original_import(name, *args, **kwargs)

    def _guarded_open(self, file, *args, **kwargs):
        _deny_open(file)
        return self._original_open(file, *args, **kwargs)

    def _guarded_io_open(self, file, *args, **kwargs):
        _deny_open(file)
        return self._original_io_open(file, *args, **kwargs)

    def _guarded_os_open(self, file, *args, **kwargs):
        _deny_open(file)
        return self._original_os_open(file, *args, **kwargs)

    def _availability_spec(self, name, *args, **kwargs):
        # Availability probes are not imports: tokenizer-only Transformers must
        # see tensor backends as unavailable, not crash while checking for them.
        # Actual importlib imports still hit the raising meta-path finder.
        if name.split(".")[0] in DENIED_RUNTIME_IMPORTS:
            return None
        return self._original_find_spec(name, *args, **kwargs)


class _DeniedRuntimeFinder:
    def find_spec(self, fullname, path=None, target=None):
        _deny_import(fullname)
        return None


def _emit_preparation(
    emit,
    operation: str,
    reply: dict,
    *,
    input_tokens: int,
    started,
    started_acknowledged=False,
):
    """Emit the supervised preparation row: start, one fragment, terminal.

    ``generation_started``/``terminal`` reuse the Qwen wire naming solely for
    supervision; they never assert Qwen work.
    """
    if not started_acknowledged:
        emit("generation_started", {"operation": operation, "qwen_generation": False})
    emit("fragment", {"text": canonical_json(reply).decode("utf-8")})
    emit(
        "terminal",
        {
            "finish_reason": "stop",
            "generation_tokens": 0,
            "input_tokens": input_tokens,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def chat_token_ids(tokenizer, messages) -> list[int]:
    """Apply the exact run chat template; preserve overlength IDs (no ceiling)."""
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
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("chat tokens must be a nonempty flat nonnegative integer list")
    return tokens


class _EncoderRuntime:
    def __init__(self, root, model):
        if model is not None:
            raise ValueError("encoder mode accepts no model spec")
        root = Path(root).resolve(strict=True)
        source = _verify_worker_source(root)
        from messaging_nlp.encoder import load_encoder

        encoder = load_encoder(root)
        self.encoder = encoder
        self.identity = {
            "mode": "encoder",
            "encoder_sha256": encoder._identity.digest(),
            "worker_source_sha256": source,
        }

    def generate(self, payload, emit):
        if type(payload) is not dict or set(payload) != {"message_id", "text"}:
            raise ValueError("encoder generate requires exact message_id/text fields")
        message_id = payload.get("message_id")
        text = payload.get("text")
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("encoder generate requires a nonblank message_id")
        if not isinstance(text, str):
            raise ValueError("encoder generate requires exact request text")
        started = time.monotonic()
        emit(
            "generation_started",
            {"operation": "encoder_extraction", "qwen_generation": False},
        )
        artifact = self.encoder.extract(message_id, text)
        # Re-validate so the reply is exactly the contract's artifact dictionary;
        # failed and valid-empty outcomes stay distinct, never collapsed.
        artifact = ExtractionArtifact.model_validate(artifact.model_dump(mode="json"))
        reply = {"mode": "encoder", "artifact": artifact.model_dump(mode="json")}
        _emit_preparation(
            emit,
            "encoder_extraction",
            reply,
            input_tokens=0,
            started=started,
            started_acknowledged=True,
        )


class _TokenizerRuntime:
    def __init__(self, root, model):
        root = Path(root).resolve(strict=True)
        source = _verify_worker_source(root)
        spec = ModelSpec.model_validate(model)
        # 1. Model-free identity verification: hashing known weight bytes is
        #    identity reading, not tensor loading, so it precedes any deny guard.
        verify_model(root, spec)
        # 2. Install the deny boundary AFTER model-free verification and BEFORE
        #    AutoTokenizer/any optional import. Only config/tokenizer/template
        #    static-byte checks remain permitted from this point on.
        boundary = _TokenizerRuntimeBoundary()
        boundary.__enter__()
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from transformers import AutoTokenizer

            directory = _local(root, spec.path)
            tokenizer = AutoTokenizer.from_pretrained(
                str(directory),
                local_files_only=True,
                trust_remote_code=False,
            )
            template = getattr(tokenizer, "chat_template", None)
            if (
                not isinstance(template, str)
                or _sha(template.encode("utf-8")) != spec.template_sha256
            ):
                raise ValueError("loaded tokenizer template differs from pinned bytes")
        finally:
            boundary.__exit__(None, None, None)
        self.tokenizer = tokenizer
        self.template_sha256 = spec.template_sha256
        self.identity = {
            "mode": "tokenizer",
            "repo": QWEN_REPO,
            "revision": QWEN_REVISION,
            "receipt_sha256": QWEN_RECEIPT_SHA256,
            "template_sha256": spec.template_sha256,
            "worker_source_sha256": source,
            "local_files_only": True,
            "trust_remote_code": False,
        }

    def generate(self, payload, emit):
        if type(payload) is not dict or set(payload) != {"messages", "messages_sha256"}:
            raise ValueError("tokenizer generate requires exact message/hash fields")
        messages = payload.get("messages")
        messages_sha256 = payload.get("messages_sha256")
        if not isinstance(messages, list):
            raise ValueError("tokenizer generate requires a messages list")
        if messages_sha256 != _sha(canonical_json(messages)):
            raise ValueError("tokenizer generate messages hash mismatch")
        started = time.monotonic()
        emit(
            "generation_started",
            {"operation": "tokenizer_fit", "qwen_generation": False},
        )
        with _TokenizerRuntimeBoundary():
            token_ids = chat_token_ids(self.tokenizer, messages)
        reply = {
            "mode": "tokenizer",
            "messages_sha256": messages_sha256,
            "template_sha256": self.template_sha256,
            "model_receipt_sha256": QWEN_RECEIPT_SHA256,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "token_ids_sha256": _sha(canonical_json(token_ids)),
            "fits": len(token_ids) <= TOKEN_CEILING,
        }
        _emit_preparation(
            emit,
            "tokenizer_fit",
            reply,
            input_tokens=len(token_ids),
            started=started,
            started_acknowledged=True,
        )


def load_runtime(payload: dict):
    """Dispatch the sole load payload {mode, root, model} to one mode loader."""
    if type(payload) is not dict or set(payload) != {"mode", "root", "model"}:
        raise ValueError("load requires exact mode/root/model fields")
    if not isinstance(payload["root"], str) or not Path(payload["root"]).is_absolute():
        raise ValueError("worker root must be an explicit absolute path")
    mode = payload.get("mode")
    root = payload.get("root")
    model = payload.get("model")
    if mode == "encoder":
        return _EncoderRuntime(root=root, model=model)
    if mode == "tokenizer":
        return _TokenizerRuntime(root=root, model=model)
    raise ValueError(f"unsupported preparation mode: {mode!r}")


def _serve(stream, emit, load_runtime):
    """Dispatch load/generate/shutdown over the private serial protocol.

    ``emit(kind, payload, row_id)`` is the raw event sink; row identity is bound
    internally. ``load_runtime(payload)`` returns an object exposing ``identity``
    and ``generate(payload, emit)``. This seam keeps the wire logic testable with
    fictional loaders; the real mode loaders are never invoked in those tests.
    """
    runtime = None
    seen_rows = set()
    for line in stream:
        row_id = None

        def bound(kind, payload):
            emit(kind, payload, row_id)

        try:
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            wire = loads_json(text)
            if not isinstance(wire, dict) or wire.get("protocol") != PROTOCOL_VERSION:
                raise ValueError("explicit current protocol version required")
            request = WorkerRequest.model_validate(wire)
            row_id = request.row_id
            if request.kind == "shutdown":
                return 0
            if request.kind == "load":
                if runtime is not None:
                    raise RuntimeError("runtime may be loaded only once")
                runtime = load_runtime(request.payload)
                bound("ready", runtime.identity)
            elif request.kind == "generate":
                if runtime is None or not row_id:
                    raise RuntimeError("generate requires loaded runtime and row ID")
                if row_id in seen_rows:
                    raise RuntimeError("row already dispatched; retries forbidden")
                seen_rows.add(row_id)
                runtime.generate(request.payload, bound)
            else:
                raise RuntimeError(f"unsupported request kind: {request.kind!r}")
        except Exception as error:
            bound(
                "error",
                {
                    "message": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
            return 1
    return 0


def main():
    """No public flags or substitution: inherited private protocol/lifeline only."""
    if len(sys.argv) != 1:
        raise RuntimeError("private preparation worker takes no arguments")
    protocol_fd = int(os.environ["NLP3_PROTOCOL_FD"])
    lifeline_fd = int(os.environ["NLP3_LIFELINE_FD"])
    if protocol_fd < 3 or lifeline_fd < 3 or protocol_fd == lifeline_fd:
        raise RuntimeError("dedicated protocol and lifeline descriptors required")
    from messaging_nlp.run_supervision import watch_parent

    watch_parent(lifeline_fd, protocol_fd)
    with os.fdopen(protocol_fd, "wb", buffering=0) as protocol:

        def emit(kind, payload, row_id=None):
            event = WorkerEvent(kind=kind, row_id=row_id, payload=payload)
            raw = canonical_json(event.model_dump(mode="json")) + b"\n"
            view = memoryview(raw)
            while view:
                written = protocol.write(view)
                if not written:
                    raise BrokenPipeError("protocol descriptor closed")
                view = view[written:]

        return _serve(sys.stdin.buffer, emit, load_runtime)


if __name__ == "__main__":
    sys.exit(main())
