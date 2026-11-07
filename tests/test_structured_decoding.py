"""Fictional, contract-derived tests for the bounded structured decoder.

No tokenizer, model, weight file, MLX runtime or Outlines build is loaded. The
ByteLevel audit sees a fictional tokenizer object and ``mlx``/``outlines_core``
are substituted in ``sys.modules``. Passing these tests is software evidence
only: the decoder constrains spelling, never meaning, and strict post-validation
of the decoded JSON is still required.
"""

import ast
import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from messaging_nlp import structured_decoding as sd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "messaging_nlp" / "structured_decoding.py"

EOS = 7
VOCAB_SIZE = 10
BASE_VOCAB = {"a": 0, "b": 1, "c": 2}
ADDED = [{"id": 7, "content": "<eos>"}, {"id": 8, "content": "<pad>"}]
COMPLETE = {"a": 0, "b": 1, "c": 2, "<eos>": 7, "<pad>": 8}
ABSENT = object()


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload

    def to_str(self):
        return json.dumps(self.payload)


def backend(**overrides):
    payload = {
        "model": {"type": "BPE", "vocab": dict(BASE_VOCAB)},
        "decoder": {"type": "ByteLevel"},
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "added_tokens": [dict(item) for item in ADDED],
    }
    payload.update(overrides)
    return payload


def tokenizer(
    payload=None,
    complete=None,
    eos=EOS,
    specials=(7, 8),
    eos_token_ids=ABSENT,
    inner=None,
):
    wrapper = SimpleNamespace(
        backend_tokenizer=FakeBackend(backend() if payload is None else payload),
        eos_token_id=eos,
        all_special_ids=list(specials),
        get_vocab=lambda: dict(COMPLETE if complete is None else complete),
    )
    if eos_token_ids is not ABSENT:
        wrapper.eos_token_ids = eos_token_ids
    if inner is not None:
        wrapper._tokenizer = inner
    return wrapper


def flat_schema():
    """Representation-shaped schema; not a case-specific answer key."""
    return {
        "type": "object",
        "title": "Envelope",
        "description": "ignored by the grammar builder",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["draft", "clarify", "unsupported"]},
            "recipient_contact_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "pattern": r"\S"},
            },
            "selected_event_id": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "pattern": r"\S"},
                    {"type": "null"},
                ],
                "default": None,
            },
            "draft": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "pattern": r"\S"},
                    {"type": "null"},
                ],
                "default": None,
            },
            "mentions": {
                "type": "array",
                "items": {"$ref": "#/$defs/Mention"},
            },
        },
        "required": [
            "action",
            "recipient_contact_ids",
            "selected_event_id",
            "draft",
            "mentions",
        ],
        "$defs": {
            "Mention": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "surface": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r"\S",
                        "default": "",
                    },
                    "rendering_kind": {"const": "person"},
                    "reference": {"type": "string"},
                },
                "required": ["surface", "rendering_kind", "reference"],
            }
        },
    }


class Tokens:
    def __init__(self, values, shape=None):
        self.values = list(values)
        self.shape = shape or (len(self.values),)

    def tolist(self):
        return list(self.values)


class Logits:
    def __init__(self, shape=(1, VOCAB_SIZE), dtype="float32"):
        self.shape = shape
        self.dtype = dtype


class FakeOutlines:
    """Substitutes outlines_core and its pinned version for one test."""

    def __init__(self):
        self.steps = []
        self.regex_calls = []
        self.guide = None
        self.version = sd.OUTLINES_CORE_VERSION
        self.version_error = None

    def install(self, monkeypatch):
        owner = self
        module = ModuleType("outlines_core")
        json_schema = ModuleType("outlines_core.json_schema")

        class Vocabulary:
            def __init__(self, eos, tokens):
                self.eos = eos
                self.tokens = {raw: list(ids) for raw, ids in tokens.items()}

        class Index:
            def __init__(self, regex, vocabulary):
                self.regex = regex
                self.vocabulary = vocabulary
                self.steps = [list(step) for step in owner.steps]

        class Guide:
            def __init__(self, index):
                if not isinstance(index, Index):
                    raise TypeError("Guide requires a fake Index")
                self.index = index
                self.position = 0
                self.advanced = []
                owner.guide = self

            def get_tokens(self):
                if self.position < len(self.index.steps):
                    return sorted(self.index.steps[self.position])
                return [self.index.vocabulary.eos]

            def is_finished(self):
                return self.position >= len(self.index.steps)

            def advance(self, token, return_tokens=False):
                if token not in self.get_tokens():
                    raise ValueError(f"unexpected token {token}")
                self.advanced.append(token)
                self.position += 1
                return []

        def build_regex_from_schema(schema, whitespace_pattern=""):
            owner.regex_calls.append((schema, whitespace_pattern))
            return f"regex-{len(owner.regex_calls)}"

        module.Index = Index
        module.Vocabulary = Vocabulary
        module.Guide = Guide
        module.json_schema = json_schema
        json_schema.build_regex_from_schema = build_regex_from_schema
        monkeypatch.setitem(sys.modules, "outlines_core", module)
        monkeypatch.setitem(sys.modules, "outlines_core.json_schema", json_schema)
        real_version = importlib.metadata.version

        def version(name):
            if name != "outlines_core":
                return real_version(name)
            if owner.version_error is not None:
                raise owner.version_error
            return owner.version

        monkeypatch.setattr(importlib.metadata, "version", version)


@pytest.fixture
def outlines(monkeypatch):
    fake = FakeOutlines()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def masks(monkeypatch):
    calls = []

    def fake_mask(logits, allowed, vocab_size):
        calls.append((list(allowed), vocab_size))
        return "MASKED"

    monkeypatch.setattr(sd, "_apply_mask", fake_mask)
    return calls


@pytest.fixture
def built(outlines):
    def make(steps, wrapper=None, schema=None, vocab_size=VOCAB_SIZE):
        outlines.steps = [list(step) for step in steps]
        return sd.build_processor(
            wrapper or tokenizer(),
            flat_schema() if schema is None else schema,
            vocab_size,
        )

    return make


def transformations(schema=None):
    _, recorded = sd._decoding_schema(flat_schema() if schema is None else schema)
    return {(row["path"], row["removed"]) for row in recorded}


# --- lazy optional imports -------------------------------------------------


def test_module_import_loads_no_optional_runtime():
    assert not {"mlx", "outlines_core"}.intersection(sys.modules)


def test_optional_imports_are_nested_in_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    assert not {"mlx", "outlines_core"}.intersection(top_level)
    nested = set()
    for node in ast.walk(tree):
        if node in tree.body:
            continue
        if isinstance(node, ast.Import):
            nested.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            nested.add((node.module or "").split(".")[0])
    assert {"mlx", "outlines_core"} <= nested


# --- schema adaptation -----------------------------------------------------


def test_defaults_are_dropped_and_recorded():
    removed = transformations()
    assert ("/properties/selected_event_id", "default") in removed
    assert ("/properties/draft", "default") in removed
    assert ("/$defs/Mention/properties/surface", "default") in removed
    decoding, _ = sd._decoding_schema(flat_schema())
    assert "default" not in decoding["properties"]["draft"]
    assert "default" not in decoding["$defs"]["Mention"]["properties"]["surface"]


def test_transformations_are_inert_and_copied():
    original = flat_schema()
    decoding, recorded = sd._decoding_schema(original)
    assert recorded
    json.dumps(recorded)
    # The caller's schema is never mutated by an adaptation that failed or passed.
    assert original["properties"]["draft"]["default"] is None
    assert decoding is not original


def test_required_enum_const_and_references_survive():
    decoding, _ = sd._decoding_schema(flat_schema())
    assert decoding["required"] == [
        "action",
        "recipient_contact_ids",
        "selected_event_id",
        "draft",
        "mentions",
    ]
    assert decoding["properties"]["action"]["enum"] == [
        "draft",
        "clarify",
        "unsupported",
    ]
    assert decoding["$defs"]["Mention"]["properties"]["rendering_kind"] == {
        "const": "person"
    }
    assert decoding["properties"]["mentions"]["items"] == {"$ref": "#/$defs/Mention"}
    assert decoding["properties"]["selected_event_id"]["anyOf"][1] == {"type": "null"}
    assert decoding["additionalProperties"] is False
    assert decoding["properties"]["recipient_contact_ids"]["maxItems"] == 8


def test_nonblank_pattern_is_removed_but_min_length_stays():
    decoding, recorded = sd._decoding_schema(flat_schema())
    items = decoding["properties"]["recipient_contact_ids"]["items"]
    assert "pattern" not in items
    assert items["minLength"] == 1
    assert {
        "path": "/properties/recipient_contact_ids/items",
        "removed": "pattern",
        "value": r"\S",
    } in recorded


def test_title_and_description_are_tolerated():
    decoding, _ = sd._decoding_schema(flat_schema())
    assert decoding["title"] == "Envelope"
    assert decoding["description"] == "ignored by the grammar builder"


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("format", "email"),
        ("oneOf", []),
        ("uniqueItems", True),
        ("$comment", "not supported"),
        ("deprecated", True),
    ],
)
def test_unexpected_schema_keywords_are_refused(keyword, value):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", keyword: value}},
    }
    with pytest.raises(ValueError, match="Unsupported bounded schema keyword"):
        sd._decoding_schema(schema)


def test_other_nonblank_patterns_are_refused():
    schema = {"type": "string", "minLength": 1, "pattern": "^[a-z]+$"}
    with pytest.raises(ValueError, match="nonblank pattern"):
        sd._decoding_schema(schema)


def test_nonblank_pattern_without_min_length_is_refused():
    schema = {"type": "string", "pattern": r"\S"}
    with pytest.raises(ValueError, match="nonblank pattern"):
        sd._decoding_schema(schema)


def test_numeric_ranges_are_removed_and_recorded():
    schema = {
        "type": "object",
        "properties": {
            "start": {
                "type": "integer",
                "minimum": 0,
                "maximum": 99,
                "exclusiveMinimum": -1,
                "exclusiveMaximum": 100,
            }
        },
    }
    decoding, recorded = sd._decoding_schema(schema)
    assert decoding["properties"]["start"] == {"type": "integer"}
    removed = {(row["path"], row["removed"]) for row in recorded}
    assert removed == {
        ("/properties/start", "minimum"),
        ("/properties/start", "maximum"),
        ("/properties/start", "exclusiveMinimum"),
        ("/properties/start", "exclusiveMaximum"),
    }


def test_additional_properties_subschema_is_visited():
    schema = {
        "type": "object",
        "additionalProperties": {"type": "string", "default": "x"},
    }
    decoding, recorded = sd._decoding_schema(schema)
    assert decoding["additionalProperties"] == {"type": "string"}
    assert ("/additionalProperties", "default") in {
        (row["path"], row["removed"]) for row in recorded
    }
    bad = {
        "type": "object",
        "additionalProperties": {"type": "string", "format": "email"},
    }
    with pytest.raises(ValueError, match="Unsupported bounded schema keyword"):
        sd._decoding_schema(bad)
    assert sd._decoding_schema({"type": "object", "additionalProperties": False})[
        0
    ] == {
        "type": "object",
        "additionalProperties": False,
    }


def test_schema_must_be_an_object():
    for value in (None, [], "object", 3):
        with pytest.raises(TypeError, match="schema must be a JSON object"):
            sd._decoding_schema(value)


def test_schema_depth_is_bounded():
    schema = {"type": "string"}
    for _ in range(sd.MAX_SCHEMA_DEPTH + 1):
        schema = {"anyOf": [schema]}
    with pytest.raises(ValueError, match="bounded adapter depth"):
        sd._decoding_schema(schema)


def test_schema_node_count_is_bounded():
    properties = {
        f"field_{index}": {"type": "string"} for index in range(sd.MAX_SCHEMA_NODES + 1)
    }
    with pytest.raises(ValueError, match="bounded adapter limit"):
        sd._decoding_schema({"type": "object", "properties": properties})


# --- ByteLevel vocabulary audit -------------------------------------------


def test_plain_bytelevel_bpe_is_accepted():
    hf, eos, tokens, ordinary, padded = sd._vocabulary(tokenizer(), VOCAB_SIZE)
    assert eos == EOS
    assert ordinary == {0, 1, 2}
    assert tokens == {b"a": [0], b"b": [1], b"c": [2]}
    assert padded == VOCAB_SIZE - len(COMPLETE)
    assert hf is not None


@pytest.mark.parametrize("vocab_size", [0, -1, True, 1.5, "10", sd.MAX_VOCAB_SIZE + 1])
def test_vocab_size_must_be_a_bounded_positive_int(vocab_size):
    with pytest.raises(ValueError, match="vocab_size"):
        sd._vocabulary(tokenizer(), vocab_size)


def test_multiple_eos_is_refused():
    with pytest.raises(ValueError, match="one shared tokenizer/runtime EOS"):
        sd._vocabulary(tokenizer(eos_token_ids={7, 9}), VOCAB_SIZE)


def test_runtime_eos_may_be_a_plain_int():
    sd._vocabulary(tokenizer(eos_token_ids=EOS), VOCAB_SIZE)


@pytest.mark.parametrize("eos", [None, "7", 42])
def test_eos_must_be_an_in_range_int(eos):
    with pytest.raises(ValueError, match="one shared tokenizer/runtime EOS"):
        sd._vocabulary(tokenizer(eos=eos), VOCAB_SIZE)


def test_non_bytelevel_decoder_is_refused():
    with pytest.raises(ValueError, match="ByteLevel BPE"):
        sd._vocabulary(tokenizer(backend(decoder={"type": "Metaspace"})), VOCAB_SIZE)


def test_non_bpe_model_is_refused():
    payload = backend(model={"type": "WordPiece", "vocab": dict(BASE_VOCAB)})
    with pytest.raises(ValueError, match="ByteLevel BPE"):
        sd._vocabulary(tokenizer(payload), VOCAB_SIZE)


@pytest.mark.parametrize(
    "normalizer", [{"type": "NFKC"}, {"type": "NFD"}, {"type": "Lowercase"}]
)
def test_non_nfc_normalizer_is_refused(normalizer):
    with pytest.raises(ValueError, match="ByteLevel BPE"):
        sd._vocabulary(tokenizer(backend(normalizer=normalizer)), VOCAB_SIZE)


def test_nfc_normalizer_is_accepted():
    sd._vocabulary(tokenizer(backend(normalizer={"type": "NFC"})), VOCAB_SIZE)


def test_prefix_space_pretokenizer_is_refused():
    payload = backend(pre_tokenizer={"type": "ByteLevel", "add_prefix_space": True})
    with pytest.raises(ValueError, match="Prefix-space ByteLevel"):
        sd._vocabulary(tokenizer(payload), VOCAB_SIZE)
    nested = backend(
        pre_tokenizer={
            "type": "Sequence",
            "pretokenizers": [
                {"type": "ByteLevel", "add_prefix_space": False},
                {"type": "ByteLevel", "add_prefix_space": True},
            ],
        }
    )
    with pytest.raises(ValueError, match="Prefix-space ByteLevel"):
        sd._vocabulary(tokenizer(nested), VOCAB_SIZE)


def test_decoder_prefix_space_flag_is_not_pretokenization():
    payload = backend(decoder={"type": "ByteLevel", "add_prefix_space": True})
    _, _, _, ordinary, _ = sd._vocabulary(tokenizer(payload), VOCAB_SIZE)
    assert ordinary == {0, 1, 2}


def test_added_tokens_are_never_ordinary():
    payload = backend()
    payload["added_tokens"] = [*ADDED, {"id": 9, "content": "<|thinking|>"}]
    complete = {**COMPLETE, "<|thinking|>": 9}
    _, _, _, ordinary, padded = sd._vocabulary(
        tokenizer(payload, complete=complete), VOCAB_SIZE
    )
    assert ordinary == {0, 1, 2}
    assert padded == VOCAB_SIZE - len(complete)


def test_base_token_outside_the_bytelevel_alphabet_is_refused():
    payload = backend(model={"type": "BPE", "vocab": {"a": 0, " ": 1}})
    with pytest.raises(ValueError, match="ByteLevel alphabet"):
        sd._vocabulary(tokenizer(payload, complete={"a": 0, " ": 1}), VOCAB_SIZE)


def test_empty_ordinary_token_is_refused():
    payload = backend(model={"type": "BPE", "vocab": {"": 0}})
    with pytest.raises(ValueError, match="Empty ordinary token"):
        sd._vocabulary(tokenizer(payload, complete={"": 0}), VOCAB_SIZE)


def test_inconsistent_vocabulary_is_refused():
    payload = backend(model={"type": "BPE", "vocab": {"a": 0}})
    with pytest.raises(ValueError, match="Inconsistent"):
        sd._vocabulary(tokenizer(payload, complete={"a": 1}), VOCAB_SIZE)


def test_out_of_range_token_id_is_refused():
    payload = backend(model={"type": "BPE", "vocab": {"a": VOCAB_SIZE}})
    with pytest.raises(ValueError, match="Inconsistent"):
        sd._vocabulary(tokenizer(payload, complete={"a": VOCAB_SIZE}), VOCAB_SIZE)


def test_tokenizer_ids_beyond_the_model_are_refused():
    complete = {**COMPLETE, "<extra>": VOCAB_SIZE + 5}
    with pytest.raises(ValueError, match="exceed the model vocabulary"):
        sd._vocabulary(tokenizer(complete=complete), VOCAB_SIZE)


def test_special_ids_are_excluded_from_the_ordinary_set():
    complete = {**COMPLETE, "<cls>": 4}
    payload = backend()
    payload["added_tokens"] = [*ADDED, {"id": 4, "content": "<cls>"}]
    _, _, _, ordinary, padded = sd._vocabulary(
        tokenizer(payload, complete=complete, specials=(4, 7, 8)), VOCAB_SIZE
    )
    assert ordinary == {0, 1, 2}
    assert padded == VOCAB_SIZE - len(complete)


# --- processor construction ------------------------------------------------


def test_build_returns_one_processor_with_an_audit(built):
    processor = built([[0], [1]])
    assert callable(processor)
    audit = processor.audit()
    assert audit["backend"] == "direct_outlines_core"
    assert audit["version"] == sd.OUTLINES_CORE_VERSION
    assert audit["vocab_size"] == VOCAB_SIZE
    assert audit["padded_ids_masked"] == VOCAB_SIZE - len(COMPLETE)
    assert audit["reserved_ids_masked"] == VOCAB_SIZE - 3 - 1
    assert audit["calls"] == 0
    assert audit["accepted"] == 0
    assert audit["terminated"] is False
    assert audit["failed"] is False
    assert audit["tokenizer_normalizer"] is None
    json.dumps(audit)
    assert "mlx" not in sys.modules


def test_grammar_is_compiled_from_the_adapted_schema(built, outlines):
    built([[0]])
    schema_text, whitespace_pattern = outlines.regex_calls[0]
    sent = json.loads(schema_text)
    assert whitespace_pattern == r"[ \n\r\t]*"
    assert "pattern" not in sent["properties"]["recipient_contact_ids"]["items"]
    assert "default" not in sent["properties"]["draft"]
    assert sent["properties"]["action"]["enum"] == ["draft", "clarify", "unsupported"]
    assert sent["required"] == flat_schema()["required"]
    assert sent["$defs"]["Mention"]["properties"]["rendering_kind"] == {
        "const": "person"
    }


def test_audit_records_the_schema_transformations(built):
    audit = built([[0]]).audit()
    recorded = {
        (row["path"], row["removed"]) for row in audit["schema_transformations"]
    }
    assert ("/properties/draft", "default") in recorded
    assert ("/properties/recipient_contact_ids/items", "pattern") in recorded


def test_audit_states_what_the_grammar_does_not_enforce(built):
    audit = built([[0]]).audit()
    assert audit["postvalidation_required"] is True
    for name in (
        "fixed_schema_property_order",
        "nonblank_not_enforced",
        "numeric_ranges_not_enforced",
        "no_input_dependent_semantics",
        "unicode_escape_spellings_not_fully_supported",
        "schema_defaults_not_applied",
    ):
        assert name in audit["limitations"]
    assert not {"valid", "passed", "quality", "score"}.intersection(audit)


def test_audit_is_a_snapshot(built):
    processor = built([[0]])
    first = processor.audit()
    first["schema_transformations"].append({"path": "/x", "removed": "default"})
    first["limitations"].append("tampered")
    second = processor.audit()
    assert {"path": "/x", "removed": "default"} not in second["schema_transformations"]
    assert "tampered" not in second["limitations"]


def test_version_pin_is_enforced(built, outlines):
    outlines.version = "0.2.13"
    with pytest.raises(RuntimeError, match="outlines_core==0.2.14"):
        built([[0]])


def test_missing_outlines_package_is_reported(built, outlines):
    outlines.version_error = importlib.metadata.PackageNotFoundError("outlines_core")
    with pytest.raises(RuntimeError, match="outlines_core==0.2.14"):
        built([[0]])


def test_build_refuses_a_schema_that_is_not_an_object(built):
    with pytest.raises(TypeError, match="schema must be a JSON object"):
        built([[0]], schema=["not", "a", "schema"])


def test_build_refuses_multiple_runtime_eos(built):
    with pytest.raises(ValueError, match="one shared tokenizer/runtime EOS"):
        built([[0]], wrapper=tokenizer(eos_token_ids={7, 9}))


def test_build_records_the_tokenizer_normalizer(built):
    processor = built([[0]], wrapper=tokenizer(backend(normalizer={"type": "NFC"})))
    assert processor.audit()["tokenizer_normalizer"] == {"type": "NFC"}


def test_wrapped_inner_tokenizer_is_the_audited_backend(built):
    inner = tokenizer(backend(normalizer={"type": "NFC"}), eos_token_ids={7})
    wrapper = tokenizer(inner=inner)
    assert "eos_token_ids" not in vars(wrapper) or wrapper.eos_token_ids == {7}
    processor = built([[0]], wrapper=wrapper)
    assert processor.audit()["tokenizer_normalizer"] == {"type": "NFC"}


# --- processor behaviour ---------------------------------------------------


def test_prompt_tokens_are_never_consumed(built, masks):
    processor = built([[0], [1]])
    assert processor(Tokens([10, 11]), Logits()) == "MASKED"
    assert processor.audit()["accepted"] == 0
    assert masks[-1] == ([0], VOCAB_SIZE)


def test_continuation_is_accepted_in_order(built, masks):
    processor = built([[0], [1]])
    processor(Tokens([10, 11]), Logits())
    processor(Tokens([10, 11, 0]), Logits())
    processor(Tokens([10, 11, 0, 1]), Logits())
    assert processor.audit()["accepted"] == 2
    assert masks[-1] == ([EOS], VOCAB_SIZE)


def test_rollback_or_repeat_is_refused(built, masks):
    processor = built([[0], [1]])
    processor(Tokens([10, 11]), Logits())
    with pytest.raises(ValueError, match="did not grow"):
        processor(Tokens([10, 11]), Logits())
    assert processor.audit()["failed"] is True


def test_changed_prefix_is_refused(built, masks):
    processor = built([[0], [1]])
    processor(Tokens([10, 11]), Logits())
    with pytest.raises(ValueError, match="prefix changed"):
        processor(Tokens([10, 99, 0]), Logits())


def test_empty_prompt_is_refused(built, masks):
    with pytest.raises(ValueError, match="nonempty prompt"):
        built([[0]])(Tokens([]), Logits())


def test_only_single_sequence_full_width_logits_are_supported(built, masks):
    narrow = built([[0], [1]])
    with pytest.raises(ValueError, match="single-sequence"):
        narrow(Tokens([10, 11]), Logits(shape=(1, VOCAB_SIZE - 1)))
    batch = built([[0], [1]])
    with pytest.raises(ValueError, match="single-sequence"):
        batch(Tokens([10, 11], shape=(1, 2)), Logits())


def test_invalid_generated_token_is_refused(built, masks):
    processor = built([[0], [1]])
    processor(Tokens([10, 11]), Logits())
    with pytest.raises(RuntimeError, match="rejected generated token 1"):
        processor(Tokens([10, 11, 1]), Logits())


def test_padded_or_reserved_ids_can_never_be_allowed(built, masks):
    for token in sorted({3, 4, 5, 6, 8, 9}):
        processor = built([[token]])
        with pytest.raises(RuntimeError, match="reserved/padded"):
            processor(Tokens([10]), Logits())
        assert processor.audit()["failed"] is True


def test_stopping_before_completion_is_refused(built, masks):
    processor = built([[0, EOS]])
    with pytest.raises(RuntimeError, match="before completion"):
        processor(Tokens([10]), Logits())


def test_empty_continuation_is_refused(built, masks):
    processor = built([[]])
    with pytest.raises(RuntimeError, match="no allowed continuation"):
        processor(Tokens([10]), Logits())


def test_single_eos_terminates_and_then_allows_only_eos(built, masks):
    processor = built([[0], [1]])
    processor(Tokens([10, 11]), Logits())
    processor(Tokens([10, 11, 0]), Logits())
    processor(Tokens([10, 11, 0, 1]), Logits())
    assert masks[-1] == ([EOS], VOCAB_SIZE)
    processor(Tokens([10, 11, 0, 1, EOS]), Logits())
    audit = processor.audit()
    assert audit["terminated"] is True
    assert audit["accepted"] == 3
    assert masks[-1] == ([EOS], VOCAB_SIZE)
    with pytest.raises(RuntimeError, match="rejected generated token 0"):
        processor(Tokens([10, 11, 0, 1, EOS, 0]), Logits())


def test_failed_processor_cannot_resume(built, masks):
    processor = built([[0]])
    processor(Tokens([10, 11]), Logits())
    with pytest.raises(RuntimeError, match="rejected generated token"):
        processor(Tokens([10, 11, 1]), Logits())
    with pytest.raises(RuntimeError, match="previously failed"):
        processor(Tokens([10, 11, 1, 0]), Logits())


# --- mask application ------------------------------------------------------


class FakeArray:
    def __init__(self, values, dtype=None):
        self.values = values
        self.dtype = dtype

    @property
    def shape(self):
        if isinstance(self.values, list):
            if self.values and isinstance(self.values[0], list):
                return (len(self.values), len(self.values[0]))
            return (len(self.values),)
        return ()

    def __setitem__(self, key, value):
        assert isinstance(key, FakeArray)
        for index in key.values:
            self.values[index] = value

    def __getitem__(self, key):
        assert key == (None, slice(None))
        return FakeArray([list(self.values)], self.dtype)


class FakeMx:
    bool_ = "bool_"
    uint32 = "uint32"

    def zeros(self, shape, dtype=None):
        return FakeArray([False] * shape[0], dtype)

    def array(self, values, dtype=None):
        if isinstance(values, (list, tuple)):
            return FakeArray(list(values), dtype)
        return FakeArray(values, dtype)

    def where(self, condition, on_true, on_false):
        row = condition.values[0]
        return FakeArray(
            [
                [
                    value if row[index] else float("-inf")
                    for index, value in enumerate(line)
                ]
                for line in on_true.values
            ],
            on_true.dtype,
        )


@pytest.fixture
def fake_mlx(monkeypatch):
    module = ModuleType("mlx")
    core = FakeMx()
    module.core = core
    monkeypatch.setitem(sys.modules, "mlx", module)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return core


def test_apply_mask_keeps_only_allowed_ids(fake_mlx):
    logits = FakeArray([[0.1, 0.2, 0.3, 0.4]], "float32")
    masked = sd._apply_mask(logits, [0, 2], 4)
    assert masked.values == [[0.1, float("-inf"), 0.3, float("-inf")]]


def test_processor_masks_through_the_real_mask_path(built, fake_mlx):
    processor = built([[0], [1]])
    logits = FakeArray([[0.1, 0.2, 0.3] + [0.5] * (VOCAB_SIZE - 3)], "float32")
    masked = processor(Tokens([10, 11]), logits)
    assert masked.values == [[0.1] + [float("-inf")] * (VOCAB_SIZE - 1)]
