"""Bounded ByteLevel/Outlines Core decoding; not semantic output validation.

Adapted for this package from the retained feasibility pilot
``Old-version/src/messaging_nlp/feasibility/structured.py``. Only the
inference-processor responsibilities were carried over: schema
adaptation, ByteLevel vocabulary audit, guide construction and token-mask
application. Nothing imports ``Old-version`` at runtime and the pilot's audit
harness was not copied. A grammatical continuation is never a valid answer: the
existing strict validator must still accept the decoded text, and the schema
adaptation below deliberately leaves nonblank/numeric constraints to that
validator.

Optional runtime imports stay lazy. Importing this module loads neither MLX nor
``outlines_core``, so the main ``.venv`` (which has neither installed) can run
the fictional tests. A processor belongs to one ordinary, single-sequence greedy
generation: no speculative decoding, no cache reuse, no rollback and no retry.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
from collections import defaultdict

OUTLINES_CORE_VERSION = "0.2.14"

# Adapter-local resource bounds, not study policy. The pilot compiled whatever it
# was given; these keep a bad schema or width from stalling the one generation.
MAX_VOCAB_SIZE = 1 << 20
MAX_SCHEMA_NODES = 4096
MAX_SCHEMA_DEPTH = 32

_NONBLANK_PATTERN = r"\S"
_NUMERIC_RANGE_KEYWORDS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
)
# Keywords passed through to the grammar builder: structure, presence and value
# sets only. Numeric ranges and the pilot nonblank pattern are handled below.
_SCALAR_KEYWORDS = frozenset(
    {
        "$ref",
        "const",
        "description",
        "enum",
        "maxItems",
        "maxLength",
        "minItems",
        "minLength",
        "required",
        "title",
        "type",
    }
)
_NAMED_SUBSCHEMA_KEYWORDS = frozenset({"$defs", "properties"})
_SUBSCHEMA_KEYWORDS = frozenset({"additionalProperties", "anyOf", "items"})

LIMITATIONS = [
    "fixed_schema_property_order",
    "nonblank_not_enforced",
    "numeric_ranges_not_enforced",
    "no_input_dependent_semantics",
    "unicode_escape_spellings_not_fully_supported",
    "schema_defaults_not_applied",
]


def _decoding_schema(schema):
    """Copy a schema into the bounded subset Core 0.2.14 can compile.

    Returns ``(decoding_schema, transformations)``. Removals are recorded for the
    audit so an adapted grammar is never described as full-schema enforcement.
    """
    if type(schema) is not dict:
        raise TypeError("schema must be a JSON object")
    result = copy.deepcopy(schema)
    transformations = []
    visited = 0

    def visit(value, path, depth):
        nonlocal visited
        if depth > MAX_SCHEMA_DEPTH:
            raise ValueError("Schema nesting exceeds the bounded adapter depth")
        visited += 1
        if visited > MAX_SCHEMA_NODES:
            raise ValueError("Schema node count exceeds the bounded adapter limit")
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}", depth + 1)
            return
        if not isinstance(value, dict):
            # Literal enum/const members and booleans carry no keywords.
            return
        if "pattern" in value:
            if value["pattern"] != _NONBLANK_PATTERN or value.get("minLength", 0) < 1:
                raise ValueError("Only the bounded nonblank pattern can be adapted")
            transformations.append(
                {"path": path, "removed": "pattern", "value": value.pop("pattern")}
            )
        # Core 0.2.14 does not enforce numerical ranges. Make the approximation
        # explicit instead of claiming full-schema support.
        for key in _NUMERIC_RANGE_KEYWORDS:
            if key in value:
                transformations.append(
                    {"path": path, "removed": key, "value": value.pop(key)}
                )
        # ``default`` is inert for validation: JSON Schema never fills it in, so
        # dropping it cannot loosen the grammar. ``required`` still decides
        # presence; Pydantic applies defaults after decoding.
        if "default" in value:
            transformations.append(
                {"path": path, "removed": "default", "value": value.pop("default")}
            )
        for key, child in list(value.items()):
            if key in _NAMED_SUBSCHEMA_KEYWORDS:
                if type(child) is not dict:
                    raise ValueError(f"{key} must be a JSON object of subschemas")
                for name, subschema in child.items():
                    visit(subschema, f"{path}/{key}/{name}", depth + 1)
            elif key in _SUBSCHEMA_KEYWORDS:
                visit(child, f"{path}/{key}", depth + 1)
            elif key not in _SCALAR_KEYWORDS:
                raise ValueError(f"Unsupported bounded schema keyword: {key}")

    visit(result, "", 0)
    return result, transformations


def _byte_decoder():
    kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {chr(byte): byte for byte in kept}
    for index, byte in enumerate(byte for byte in range(256) if byte not in kept):
        mapping[chr(256 + index)] = byte
    return mapping


def _vocabulary(tokenizer_wrapper, vocab_size):
    """Fail-closed ByteLevel audit; returns the ordinary-token allow-list.

    Every added/control token is excluded, including ones a tokenizer does not
    label special (for example thinking or tool tags). Padded ids are counted so
    the audit can state how many rows the mask always suppresses.
    """
    inner = getattr(tokenizer_wrapper, "_tokenizer", None)
    # HF fast tokenizers themselves have a private _tokenizer (the Rust
    # backend), whereas MLX's _tokenizer is the actual HF wrapper.
    tokenizer = inner if hasattr(inner, "backend_tokenizer") else tokenizer_wrapper
    if type(vocab_size) is not int or not 0 < vocab_size <= MAX_VOCAB_SIZE:
        raise ValueError("vocab_size must be the positive bounded model logits width")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    if (
        backend.get("model", {}).get("type") != "BPE"
        or backend.get("decoder", {}).get("type") != "ByteLevel"
        or backend.get("normalizer") not in (None, {"type": "NFC"})
    ):
        raise ValueError("Only ByteLevel BPE with no normalizer or NFC is supported")

    # Decoder flags do not insert prefix spaces. Pretokenizer flags do.
    def check_pretokenizer(value):
        if isinstance(value, dict):
            if value.get("type") == "ByteLevel" and value.get(
                "add_prefix_space", False
            ):
                raise ValueError("Prefix-space ByteLevel tokenization is unsupported")
            for child in value.values():
                check_pretokenizer(child)
        elif isinstance(value, list):
            for child in value:
                check_pretokenizer(child)

    check_pretokenizer(backend.get("pre_tokenizer"))
    eos = tokenizer.eos_token_id
    stops = getattr(tokenizer_wrapper, "eos_token_ids", {eos})
    stops = {stops} if type(stops) is int else set(stops)
    if type(eos) is not int or not 0 <= eos < vocab_size or stops != {eos}:
        raise ValueError(
            "This bounded adapter requires one shared tokenizer/runtime EOS"
        )
    decoder = _byte_decoder()
    tokens = defaultdict(list)
    ordinary_ids = set()
    # Include only original BPE tokens: all added/control tokens are excluded,
    # even those the tokenizer does not label special.
    base = backend["model"]["vocab"]
    complete = tokenizer.get_vocab()
    special = set(getattr(tokenizer, "all_special_ids", ()))
    special.update(token["id"] for token in backend.get("added_tokens", ()))
    for token, token_id in base.items():
        if (
            type(token_id) is not int
            or not 0 <= token_id < vocab_size
            or complete.get(token) != token_id
            or token_id in ordinary_ids
        ):
            raise ValueError("Inconsistent or out-of-range ByteLevel vocabulary")
        if token_id == eos or token_id in special:
            continue
        try:
            raw = bytes(decoder[character] for character in token)
        except KeyError as error:
            raise ValueError(
                "Base token is not encoded with the ByteLevel alphabet"
            ) from error
        if not raw:
            raise ValueError("Empty ordinary token is unsupported")
        tokens[raw].append(token_id)
        ordinary_ids.add(token_id)
    if not ordinary_ids:
        raise ValueError("No ordinary ByteLevel token is available")
    present = set(complete.values())
    if any(type(i) is not int or not 0 <= i < vocab_size for i in present):
        raise ValueError("Tokenizer IDs exceed the model vocabulary")
    return tokenizer, eos, dict(tokens), ordinary_ids, vocab_size - len(present)


def _outlines_version():
    try:
        return importlib.metadata.version("outlines_core")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"This adapter requires outlines_core=={OUTLINES_CORE_VERSION}"
        ) from error


def _compile_index(schema, eos, tokens):
    if _outlines_version() != OUTLINES_CORE_VERSION:
        raise RuntimeError(
            f"This adapter requires outlines_core=={OUTLINES_CORE_VERSION}"
        )
    from outlines_core import Index, Vocabulary
    from outlines_core.json_schema import build_regex_from_schema

    regex = build_regex_from_schema(json.dumps(schema), r"[ \n\r\t]*")
    return Index(regex, Vocabulary(eos, tokens))


def _new_guide(index):
    from outlines_core import Guide

    return Guide(index)


def _apply_mask(logits, allowed, vocab_size):
    import mlx.core as mx

    mask = mx.zeros((vocab_size,), dtype=mx.bool_)
    mask[mx.array(allowed, dtype=mx.uint32)] = True
    return mx.where(mask[None, :], logits, mx.array(-float("inf"), dtype=logits.dtype))


class _Processor:
    """Token-level pre-sampling gate for one single-sequence generation."""

    def __init__(
        self,
        guide,
        eos,
        ordinary_ids,
        vocab_size,
        padded,
        transformations,
        tokenizer_normalizer=None,
    ):
        self.guide = guide
        self.eos = eos
        self.ordinary_ids = ordinary_ids
        self.vocab_size = vocab_size
        self.padded = padded
        self.transformations = transformations
        self.tokenizer_normalizer = tokenizer_normalizer
        self.calls = 0
        self.accepted = 0
        self.terminated = False
        self.failed = False
        self.seen = None
        self.history = None

    def __call__(self, tokens, logits):
        """Return masked logits; the caller must sample greedily from them."""
        if self.failed:
            raise RuntimeError("Structured processor previously failed; cannot resume")
        self.calls += 1
        try:
            if len(tokens.shape) != 1 or logits.shape != (1, self.vocab_size):
                raise ValueError(
                    "Only single-sequence tokens and full-width logits are supported"
                )
            size = tokens.shape[0]
            history = tokens.tolist()
            if self.seen is None:
                if size == 0:
                    raise ValueError("A nonempty prompt is required")
                # Never consume the prompt, including its last token.
                self.seen = size
            else:
                if size <= self.seen:
                    raise ValueError(
                        "Token history did not grow; no resets or rollback supported"
                    )
                if history[: self.seen] != self.history:
                    raise ValueError("Token history prefix changed")
                for token in history[self.seen :]:
                    self._accept(token)
                self.seen = size
            self.history = history
            allowed = self._allowed()
            if not allowed:
                raise RuntimeError("Grammar has no allowed continuation")
            return _apply_mask(logits, allowed, self.vocab_size)
        except Exception:
            self.failed = True
            raise

    def _allowed(self):
        if self.terminated:
            return [self.eos]
        allowed = self.guide.get_tokens()
        if any(
            token != self.eos and token not in self.ordinary_ids for token in allowed
        ):
            raise RuntimeError("Grammar attempted to allow a reserved/padded token")
        if self.eos in allowed and not self.guide.is_finished():
            raise RuntimeError("Grammar attempted to stop before completion")
        return allowed

    def _accept(self, token):
        if token not in self._allowed():
            raise RuntimeError(f"Grammar rejected generated token {token}")
        if token == self.eos:
            # Core 0.2.14 advertises EOS but cannot advance through it. Track that
            # transition ourselves and retain EOS-only lookahead masks.
            self.terminated = True
        else:
            self.guide.advance(token, return_tokens=False)
        self.accepted += 1

    def audit(self):
        """JSON-serializable counters; neither model quality nor final validation."""
        return {
            "backend": "direct_outlines_core",
            "version": OUTLINES_CORE_VERSION,
            "vocab_size": self.vocab_size,
            "padded_ids_masked": self.padded,
            "reserved_ids_masked": self.vocab_size - len(self.ordinary_ids) - 1,
            "calls": self.calls,
            "accepted": self.accepted,
            "terminated": self.terminated,
            "failed": self.failed,
            "schema_transformations": copy.deepcopy(self.transformations),
            "tokenizer_normalizer": copy.deepcopy(self.tokenizer_normalizer),
            "bounds": {
                "max_vocab_size": MAX_VOCAB_SIZE,
                "max_schema_nodes": MAX_SCHEMA_NODES,
                "max_schema_depth": MAX_SCHEMA_DEPTH,
            },
            "postvalidation_required": True,
            "limitations": list(LIMITATIONS),
        }


def build_processor(tokenizer_wrapper, schema: dict, vocab_size: int):
    """Build a fresh, token-level pre-sampling processor for one generation."""
    hf, eos, tokens, ordinary, padded = _vocabulary(tokenizer_wrapper, vocab_size)
    decoding_schema, transformations = _decoding_schema(schema)
    index = _compile_index(decoding_schema, eos, tokens)
    normalizer = json.loads(hf.backend_tokenizer.to_str()).get("normalizer")
    return _Processor(
        _new_guide(index),
        eos,
        ordinary,
        vocab_size,
        padded,
        transformations,
        normalizer,
    )
