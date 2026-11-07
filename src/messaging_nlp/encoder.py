"""Strict local XLM-R adapter. Importing this module opens no model assets.

Only ``load_encoder`` is the supported construction API. The underscore seams
exist for fictional software tests, not alternative study models/tokenizers.
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path

from messaging_nlp.contracts import Extraction, ExtractionResult, TechnicalFailure
from messaging_nlp.encoder_cache import (
    EncoderIdentity,
    ExtractionArtifact,
    portable_path,
    retained_identity,
    text_sha256,
)
from messaging_nlp.encoder_decode import (
    DECODER_POLICY,
    LABELS,
    OFFSET_POLICY,
    audit_policy_offsets,
    decode_bio,
    entities_from_tags,
)
from messaging_nlp.io import read_json
from messaging_nlp.io import sha256 as _sha256


def _path(root: Path, relative: str) -> Path:
    path = root
    for part in portable_path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlinked encoder asset: {relative}")
    return path


def _tree(directory: Path, files: set[str], bookkeeping: set[str] = frozenset()):
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("missing or symlinked encoder directory")
    allowed_dirs = {
        str(parent)
        for name in bookkeeping
        for parent in Path(name).parents
        if str(parent) != "."
    }
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ValueError(f"symlinked encoder asset: {relative}")
        if path.is_dir() and relative in allowed_dirs:
            continue
        if path.is_file() and relative in files | bookkeeping:
            continue  # Bookkeeping is enumerated, never opened or interpreted.
        raise ValueError(f"unexpected encoder asset: {relative}")


def _json(path: Path) -> dict:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("encoder metadata must be a JSON object")

    def hooks(item):
        if isinstance(item, dict):
            if {
                "auto_map",
                "custom_pipelines",
                "_auto_class",
                "trust_remote_code",
            } & item.keys():
                raise ValueError("custom loading hooks are forbidden")
            for child in item.values():
                hooks(child)
        elif isinstance(item, list):
            for child in item:
                hooks(child)

    hooks(value)
    return value


def _verify_assets(root: Path, identity: EncoderIdentity) -> tuple[Path, Path, dict]:
    """Verify against the caller's trusted identity, never adjacent self-claims."""
    root = Path(root).absolute()
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("encoder root must be an existing non-symlink directory")
    identity = EncoderIdentity.model_validate(identity.model_dump())
    if (
        identity.labels != list(LABELS)
        or identity.offset_policy != OFFSET_POLICY
        or identity.decoder_policy != DECODER_POLICY
    ):
        raise ValueError("encoder decoding identity mismatch")
    base = _path(root, identity.base_directory)
    component = _path(root, identity.component_directory)
    receipt_path = _path(root, identity.receipt_path)
    cache = {
        f".cache/huggingface/download/{name}.{suffix}"
        for name in identity.base_files
        for suffix in ("metadata", "lock")
    } | {
        ".cache/huggingface/.gitignore",
        ".cache/huggingface/CACHEDIR.TAG",
        f".cache/huggingface/trees/{identity.revision}.json",
    }
    # Revision is also used in a bookkeeping filename; forbid path injection.
    if (
        "/" in identity.revision
        or "\\" in identity.revision
        or identity.revision in {".", ".."}
    ):
        raise ValueError("unsafe revision")
    _tree(base, set(identity.base_files), cache)
    _tree(component, {"block-head.pt", "block-head-config.json"})
    for name, expected in identity.base_files.items():
        path = _path(root, f"{identity.base_directory}/{name}")
        if not path.is_file() or path.stat().st_size != expected.bytes:
            raise ValueError(f"missing or wrong-sized encoder asset: {name}")
        if _sha256(path) != expected.sha256:
            raise ValueError(f"encoder asset hash mismatch: {name}")
    for path, expected in (
        (receipt_path, identity.receipt_sha256),
        (component / "block-head-config.json", identity.component_config_sha256),
        (component / "block-head.pt", identity.component_sha256),
    ):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"encoder asset hash mismatch: {path.name}")
    receipt = _json(receipt_path)
    rows = receipt.get("files")
    if (
        receipt.get("repo") != identity.repo
        or receipt.get("revision") != identity.revision
        or receipt.get("directory") != identity.base_directory
        or type(receipt.get("total_bytes")) is not int
        or receipt["total_bytes"] != sum(v.bytes for v in identity.base_files.values())
        or not isinstance(rows, list)
        or len(rows) != len(identity.base_files)
    ):
        raise ValueError("encoder receipt identity mismatch")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("file") not in identity.base_files:
            raise ValueError("unexpected receipt file")
        name = row["file"]
        expected = identity.base_files[name]
        if (
            name in seen
            or type(row.get("bytes")) is not int
            or row["bytes"] != expected.bytes
            or row.get("sha256") != expected.sha256
            or row.get("verified") is not True
        ):
            raise ValueError("encoder receipt file mismatch")
        seen.add(name)
    config = _json(base / "config.json")
    tokenizer_config = _json(base / "tokenizer_config.json")
    _json(base / "tokenizer.json")
    if tokenizer_config.get("tokenizer_class") not in (
        None,
        "XLMRobertaTokenizer",
        "XLMRobertaTokenizerFast",
    ):
        raise ValueError("unexpected tokenizer class")
    if {
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "sentencepiece_model_file",
        "special_tokens_map_file",
        "added_tokens_file",
        "fast_tokenizer_files",
    } & tokenizer_config.keys():
        raise ValueError("tokenizer asset redirection is forbidden")
    if (
        config.get("model_type") != "xlm-roberta"
        or config.get("architectures") != ["XLMRobertaForMaskedLM"]
        or type(config.get("hidden_size")) is not int
        or config["hidden_size"] != identity.hidden_size
        or type(config.get("num_hidden_layers")) is not int
        or config["num_hidden_layers"] != identity.final_block_index + 1
    ):
        raise ValueError("encoder config architecture/dimension mismatch")
    metadata = _json(component / "block-head-config.json")
    expected_base = {
        "model_files_sha256": {k: v.sha256 for k, v in identity.base_files.items()},
        "config": config,
        "verified_download_identity": {
            "repo": identity.repo,
            "revision": identity.revision,
            "receipt_sha256": identity.receipt_sha256,
            "verification": "pinned_five_file_sizes_and_sha256_plus_receipt_metadata",
        },
    }
    if (
        set(metadata)
        != {
            "format",
            "labels",
            "hidden_size",
            "final_block_index",
            "settings",
            "base_model",
            "checkpoint_sha256",
        }
        or metadata["format"] != identity.format
        or metadata["labels"] != identity.labels
        or type(metadata["hidden_size"]) is not int
        or metadata["hidden_size"] != identity.hidden_size
        or type(metadata["final_block_index"]) is not int
        or metadata["final_block_index"] != identity.final_block_index
        or metadata["base_model"] != expected_base
        or metadata["checkpoint_sha256"] != identity.component_sha256
    ):
        raise ValueError("component metadata identity mismatch")
    settings = metadata["settings"]
    required_settings = {
        "max_length": 256,
        "threads": 2,
        "device": "cpu",
        "dtype": "float32",
        "offset_policy": OFFSET_POLICY,
        "decoder": DECODER_POLICY,
        "encoder_trainable": "final_block_only",
        "encoder_mode": "eval",
        "block_dropout": "disabled_eval_mode",
    }
    if not isinstance(settings, dict) or any(
        type(settings.get(k)) is not type(v) or settings.get(k) != v
        for k, v in required_settings.items()
    ):
        raise ValueError("component settings mismatch")
    return base, component, config


def _strict_state(module, state, torch):
    expected = module.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("component state keys mismatch")
    for name, tensor in state.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected[name].shape
            or tensor.dtype != expected[name].dtype
            or tensor.device.type != "cpu"
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"component state shape/dtype/value mismatch: {name}")
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("partial component load")


def _assemble(base, component: Path, identity: EncoderIdentity):
    """Same strict final-block/head assembly for retained and tiny fixtures."""
    import torch

    _configure_torch(torch, identity)
    if (
        base.config.hidden_size != identity.hidden_size
        or len(base.encoder.layer) != identity.final_block_index + 1
    ):
        raise ValueError("assembled encoder dimensions mismatch")
    state = torch.load(
        component / "block-head.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(state, dict) or set(state) != {"final_block", "head"}:
        raise ValueError("component requires exactly final_block and head")
    head = torch.nn.Linear(
        identity.hidden_size, len(LABELS), dtype=torch.float32, device="cpu"
    )
    _strict_state(
        base.encoder.layer[identity.final_block_index], state["final_block"], torch
    )
    _strict_state(head, state["head"], torch)
    for module in (base, head):
        module.to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)
    return _TorchRuntime(base, head, identity)


def _configure_torch(torch, identity):
    settings = identity.settings
    torch.set_num_threads(settings.threads)
    if torch.get_num_interop_threads() != settings.interop_threads:
        torch.set_num_interop_threads(settings.interop_threads)
    torch.use_deterministic_algorithms(True)


def _load_encoder(root: Path, identity: EncoderIdentity, *, tokenizer_factory=None):
    base_path, component, config_dict = _verify_assets(root, identity)
    import torch
    from safetensors import safe_open
    from transformers import (
        XLMRobertaConfig,
        XLMRobertaForMaskedLM,
        XLMRobertaTokenizerFast,
    )

    _configure_torch(torch, identity)
    config = XLMRobertaConfig.from_dict(config_dict)
    # The retained MLM serialization contains this unused, complete pooler pair.
    # Validate actual stored tensors before excluding ONLY these exact keys.
    pooler_shapes = {
        "roberta.pooler.dense.weight": (identity.hidden_size, identity.hidden_size),
        "roberta.pooler.dense.bias": (identity.hidden_size,),
    }
    with safe_open(base_path / "model.safetensors", framework="pt", device="cpu") as f:
        stored = set(f.keys())
        unused_pooler = stored & pooler_shapes.keys()
        if unused_pooler and unused_pooler != pooler_shapes.keys():
            raise ValueError("partial unused pooler pair")
        for name in unused_pooler:
            tensor = f.get_tensor(name)
            if (
                tuple(tensor.shape) != pooler_shapes[name]
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all().item()
            ):
                raise ValueError(f"unused pooler shape/dtype/value mismatch: {name}")
            del tensor
    with torch.device("cpu"):
        model, info = XLMRobertaForMaskedLM.from_pretrained(
            base_path,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            output_loading_info=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
    if set(info.get("unexpected_keys", [])) - unused_pooler or any(
        info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"non-strict base load: {info!r}")
    # Transformers may filter architecture-specific missing/unexpected keys.
    # Independently inspect the complete stored key set; ONLY true tied aliases
    # may be absent. No arbitrary MLM-key ignore when taking .roberta.
    stored -= unused_pooler
    expected = model.state_dict()
    missing = set(expected) - stored
    aliases = {
        "lm_head.decoder.weight": "roberta.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }
    if stored - set(expected) or missing - aliases.keys():
        raise ValueError("base safetensors key mismatch")
    for alias in missing:
        original = aliases[alias]
        if (
            original not in stored
            or expected[alias].data_ptr() != expected[original].data_ptr()
        ):
            raise ValueError("unverified tied base alias")
    runtime = _assemble(model.roberta, component, identity)
    factory = tokenizer_factory or XLMRobertaTokenizerFast.from_pretrained
    tokenizer = factory(base_path, local_files_only=True, trust_remote_code=False)
    if getattr(tokenizer, "is_fast", False) is not True:
        raise ValueError("exact-offset extraction requires a fast tokenizer")
    return Encoder._create(identity, tokenizer, runtime, config.vocab_size)


def load_encoder(root: Path) -> Encoder:
    """Load ONLY the code-pinned retained encoder (requires run authorization)."""
    return _load_encoder(root, retained_identity())


class _TorchRuntime:
    def __init__(self, base, head, identity):
        import torch

        self.torch, self.base, self.head, self.identity = torch, base, head, identity
        self._state = self._named_state()
        self._versions = {
            name: (value._version, tuple(value.shape), value.data_ptr())
            for name, value in self._state.items()
        }
        self.validate()

    def _named_state(self):
        return {
            (index, kind, name): value
            for index, module in enumerate((self.base, self.head))
            for kind, values in (
                ("parameter", module.named_parameters()),
                ("buffer", module.named_buffers()),
            )
            for name, value in values
        }

    def validate(self):
        torch = self.torch
        if (
            torch.get_num_threads() != self.identity.settings.threads
            or torch.get_num_interop_threads() != self.identity.settings.interop_threads
            or not torch.are_deterministic_algorithms_enabled()
            or getattr(self.base.config, "_attn_implementation", "eager") != "eager"
        ):
            raise RuntimeError("encoder execution settings changed")
        for module in (self.base, self.head):
            if any(child.training for child in module.modules()):
                raise RuntimeError("encoder is not in evaluation mode")
            for parameter in module.parameters():
                if (
                    parameter.requires_grad
                    or parameter.grad is not None
                    or parameter.device.type != "cpu"
                    or parameter.dtype != torch.float32
                ):
                    raise RuntimeError("encoder is not frozen CPU float32")
        current = self._named_state()
        if set(current) != set(self._state):
            raise RuntimeError("encoder state keys changed")
        for name, value in current.items():
            version, shape, pointer = self._versions[name]
            if (
                value is not self._state[name]
                or tuple(value.shape) != shape
                or value.data_ptr() != pointer
                or value.device.type != "cpu"
                or value._version != version
                or (value.is_floating_point() and value.dtype != torch.float32)
            ):
                raise RuntimeError("encoder state changed")

    def forward(self, input_ids, attention_mask):
        torch = self.torch
        self.validate()
        try:
            with torch.inference_mode():
                features = self.base(
                    input_ids=torch.tensor([input_ids], dtype=torch.long, device="cpu"),
                    attention_mask=torch.tensor(
                        [attention_mask], dtype=torch.long, device="cpu"
                    ),
                    return_dict=True,
                ).last_hidden_state
                logits = self.head(features)
                if (
                    features.requires_grad
                    or logits.requires_grad
                    or features.device.type != "cpu"
                    or logits.device.type != "cpu"
                    or features.dtype != torch.float32
                    or logits.dtype != torch.float32
                ):
                    raise RuntimeError("invalid forward execution invariants")
                return logits[0].tolist()
        finally:
            self.validate()


class Encoder:
    def __init__(self):
        raise TypeError("use load_encoder(root)")

    @classmethod
    def _create(cls, identity, tokenizer, runtime, vocab_size):
        """Private fake seam: runtime.validate(), forward(ids, mask) -> [N,9]."""
        self = object.__new__(cls)
        self._identity = EncoderIdentity.model_validate(identity.model_dump())
        self._tokenizer, self._runtime, self._vocab_size = (
            tokenizer,
            runtime,
            vocab_size,
        )
        return self

    def extract(self, message_id: str, text: str) -> ExtractionArtifact:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("extract requires a nonblank exact request text")
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("extract requires a nonblank message_id")
        diagnostics = {"text_modified": False, "forward_performed": False}
        raw = None
        code = "runtime_error"
        extraction = failure = None
        try:
            tokenized = self._tokenizer(
                text,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_attention_mask=True,
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            diagnostics["raw_tokenizer_output"] = repr(tokenized)
            code = "invalid_output"
            ids, mask, offsets, special = _validate_tokens(
                tokenized, text, self._vocab_size
            )
            diagnostics["token_count"] = len(ids)
            if len(ids) > self._identity.settings.max_tokens:
                code = "truncated"
                raise ValueError(
                    "input exceeds 256 tokens including special tokens; no forward"
                )
            policy = audit_policy_offsets(text, offsets, special)
            diagnostics["offset_policy"] = policy
            code = "runtime_error"
            self._runtime.validate()
            diagnostics["forward_performed"] = True
            try:
                logits = self._runtime.forward(ids, mask)
            finally:
                self._runtime.validate()
            raw = repr(logits)
            code = "invalid_output"
            _validate_logits(logits, len(ids))
            transformed = policy["transformed_offsets"]
            selected = [
                i
                for i, ((left, right), is_special) in enumerate(
                    zip(transformed, special, strict=True)
                )
                if not is_special and left < right
            ]
            tags = decode_bio([logits[i] for i in selected]) if selected else []
            diagnostics["decoded_token_count"] = len(selected)
            diagnostics["tags"] = tags
            entities = entities_from_tags(
                text, [transformed[i] for i in selected], tags
            )
            extraction = Extraction.model_validate(
                {"entities": entities}
            ).validate_text(text)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            diagnostics["raw_error"] = error_text
            diagnostics["traceback"] = traceback.format_exc()
            failure = TechnicalFailure(
                stage="extraction", code=code, message=error_text
            )
            if raw is None:
                raw = error_text
        result = ExtractionResult(
            message_id=message_id,
            status="failed" if failure else "ok",
            extraction=extraction,
            failure=failure,
            raw_output=raw,
            input_sha256=text_sha256(text),
            extractor_sha256=self._identity.digest(),
        )
        return ExtractionArtifact(
            text=text,
            identity=self._identity.model_copy(deep=True),
            result=result,
            diagnostics=diagnostics,
        )


def _validate_tokens(value, text, vocab_size):
    if not isinstance(value, dict) and not hasattr(value, "keys"):
        raise ValueError("tokenizer output is not a mapping")
    required = {"input_ids", "attention_mask", "offset_mapping", "special_tokens_mask"}
    if not required <= value.keys() or set(value.keys()) - required - {
        "token_type_ids"
    }:
        raise ValueError("unexpected tokenizer output keys")
    ids, mask, offsets, special = (
        value[k]
        for k in (
            "input_ids",
            "attention_mask",
            "offset_mapping",
            "special_tokens_mask",
        )
    )
    if (
        any(not isinstance(v, (list, tuple)) for v in (ids, mask, offsets, special))
        or not ids
        or not len(ids) == len(mask) == len(offsets) == len(special)
    ):
        raise ValueError("tokenizer shape mismatch")
    if any(type(v) is not int or not 0 <= v < vocab_size for v in ids):
        raise ValueError("invalid tokenizer IDs")
    if any(type(v) is not int or v != 1 for v in mask):
        raise ValueError("unexpected padding/attention mask")
    if any(type(v) is not int or v not in (0, 1) for v in special):
        raise ValueError("invalid special-token mask")
    if "token_type_ids" in value and (
        not isinstance(value["token_type_ids"], (list, tuple))
        or len(value["token_type_ids"]) != len(ids)
        or any(type(v) is not int or v != 0 for v in value["token_type_ids"])
    ):
        raise ValueError("unexpected token type IDs")
    previous_left = previous_right = 0
    for offset, is_special in zip(offsets, special, strict=True):
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or any(type(v) is not int for v in offset)
            or not 0 <= offset[0] <= offset[1] <= len(text)
        ):
            raise ValueError("invalid code-point tokenizer offset")
        left, right = offset
        if is_special and (left, right) != (0, 0):
            raise ValueError("special token owns input code points")
        if not is_special and left < right:
            # Retained semantics allow multiple Unicode/subword pieces to own
            # overlapping code points. Ordering, not disjointness, is required;
            # distinct predicted entities are checked by entities_from_tags.
            if left < previous_left or right < previous_right:
                raise ValueError("reversed tokenizer offsets")
            previous_left, previous_right = left, right
    return list(ids), list(mask), list(offsets), list(special)


def _validate_logits(logits, count):
    if not isinstance(logits, (list, tuple)) or len(logits) != count:
        raise ValueError("invalid token-classifier logits shape")
    for row in logits:
        if not isinstance(row, (list, tuple)) or len(row) != len(LABELS):
            raise ValueError("invalid token-classifier logits shape")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError("non-finite/non-numeric token-classifier logits")
