"""Fictional adapter tests: no Torch, pretrained tokenizer, or retained assets."""

import copy
import hashlib
import inspect
import json
import re

import pytest

from messaging_nlp import encoder
from messaging_nlp.encoder_cache import EncoderIdentity, FileIdentity, retained_identity
from messaging_nlp.encoder_decode import LABELS


class FakeTokenizer:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.error:
            raise self.error
        return copy.deepcopy(self.output)


class FakeRuntime:
    def __init__(self, logits=None, error=None):
        self.logits, self.error = logits, error
        self.calls = []
        self.validations = 0

    def validate(self):
        self.validations += 1

    def forward(self, ids, mask):
        self.calls.append((ids, mask))
        if self.error:
            raise self.error
        if self.logits is not None:
            return self.logits
        return [[5.0] + [0.0] * 8 for _ in ids]


def tokens(offsets):
    return {
        "input_ids": [0] + [7] * len(offsets) + [2],
        "attention_mask": [1] * (len(offsets) + 2),
        "offset_mapping": [(0, 0), *offsets, (0, 0)],
        "special_tokens_mask": [1] + [0] * len(offsets) + [1],
    }


def adapter(tokenizer, runtime=None):
    runtime = runtime or FakeRuntime()
    return encoder.Encoder._create(retained_identity(), tokenizer, runtime, 32), runtime


def test_exact_noisy_surface_kwargs_offsets_and_identity():
    text = "👓  Li\u0301ra?!\t7"
    left, right = text.index("L"), text.index("?")
    output = tokens([(left, right)])
    logits = [[0.0] * 9 for _ in range(3)]
    logits[1][1] = 11.0
    tokenizer = FakeTokenizer(output)
    model, runtime = adapter(tokenizer, FakeRuntime(logits))
    artifact = model.extract("fiction-1", text)
    assert tokenizer.calls == [
        (
            text,
            {
                "return_offsets_mapping": True,
                "return_special_tokens_mask": True,
                "return_attention_mask": True,
                "add_special_tokens": True,
                "truncation": False,
                "padding": False,
            },
        )
    ]
    assert artifact.text == text
    assert artifact.result.status == "ok"
    entity = artifact.result.extraction.entities[0]
    assert (entity.start, entity.end, entity.text) == (left, right, "Li\u0301ra")
    assert artifact.result.input_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert artifact.result.extractor_sha256 == retained_identity().digest()
    assert len(runtime.calls) == 1
    assert runtime.validations == 2
    assert artifact.diagnostics["offset_policy"]["text_modified"] is False


@pytest.mark.parametrize("count", [256, 257])
def test_ceiling_includes_specials_and_never_forwards_over_limit(count):
    text = "x" * (count - 2)
    tokenizer = FakeTokenizer(tokens([(i, i + 1) for i in range(len(text))]))
    model, runtime = adapter(tokenizer)
    artifact = model.extract("ceiling", text)
    assert artifact.diagnostics["token_count"] == count
    if count == 256:
        assert artifact.result.status == "ok"
        assert len(runtime.calls) == 1
    else:
        assert artifact.result.failure.code == "truncated"
        assert artifact.result.extraction is None
        assert runtime.calls == []
        assert artifact.diagnostics["forward_performed"] is False
        assert artifact.result.raw_output


def test_empty_entities_are_success_not_exception_and_no_retry():
    tokenizer = FakeTokenizer(tokens([(0, 4)]))
    model, _ = adapter(tokenizer)
    result = model.extract("empty", "fern").result
    assert result.status == "ok"
    assert result.extraction.entities == []
    assert result.failure is None
    broken = FakeTokenizer(error=RuntimeError("fictional tokenizer boom"))
    model, runtime = adapter(broken)
    artifact = model.extract("broken", "fern")
    assert artifact.result.failure.code == "runtime_error"
    assert artifact.result.extraction is None
    assert artifact.result.raw_output == "RuntimeError: fictional tokenizer boom"
    assert "fictional tokenizer boom" in artifact.diagnostics["raw_error"]
    assert "RuntimeError" in artifact.diagnostics["traceback"]
    assert len(broken.calls) == 1
    assert runtime.calls == []


def test_forward_exception_preserves_error_and_checks_invariants_afterward():
    tokenizer = FakeTokenizer(tokens([(0, 4)]))
    model, runtime = adapter(
        tokenizer, FakeRuntime(error=RuntimeError("forward exploded"))
    )
    artifact = model.extract("bad-forward", "fern")
    assert artifact.result.failure.code == "runtime_error"
    assert artifact.result.raw_output == "RuntimeError: forward exploded"
    assert len(runtime.calls) == 1
    assert runtime.validations == 2


@pytest.mark.parametrize(
    "change",
    [
        {"offset_mapping": [(0, 0), (0, 5), (0, 0)]},
        {"offset_mapping": [(0, 0), (-1, 2), (0, 0)]},
        {"offset_mapping": [(0, 0), (True, 2), (0, 0)]},
        {"special_tokens_mask": [1, 2, 1]},
        {"special_tokens_mask": [True, 0, 1]},
        {"special_tokens_mask": [1, 1, 1]},
        {"attention_mask": [1, 0, 1]},
        {"attention_mask": [1, 1]},
        {"input_ids": [0, 32, 2]},
        {"input_ids": [0, True, 2]},
        {"input_ids": [[0, 7, 2]]},
        {"offset_mapping": [(1, 1), (0, 4), (0, 0)]},
        {"token_type_ids": [0, 1, 0]},
    ],
)
def test_invalid_tokenizer_output_cannot_become_empty_success(change):
    value = tokens([(0, 4)])
    value.update(change)
    model, runtime = adapter(FakeTokenizer(value))
    artifact = model.extract("invalid-token", "fern")
    assert artifact.result.failure.code == "invalid_output"
    assert artifact.result.extraction is None
    assert artifact.diagnostics["raw_error"]
    assert not runtime.calls


@pytest.mark.parametrize(
    "logits",
    [
        [],
        [[0.0] * 9],
        [[0.0] * 8] * 3,
        [[float("nan")] * 9] * 3,
        [[float("inf")] * 9] * 3,
        [["5"] * 9] * 3,
        [[True] * 9] * 3,
    ],
)
def test_bad_logits_are_failed_and_raw_output_is_preserved(logits):
    model, runtime = adapter(FakeTokenizer(tokens([(0, 4)])), FakeRuntime(logits))
    artifact = model.extract("bad-logits", "fern")
    assert artifact.result.failure.code == "invalid_output"
    assert artifact.result.raw_output == repr(logits)
    assert artifact.result.extraction is None
    assert len(runtime.calls) == 1


def test_invalid_bio_is_not_repaired(monkeypatch):
    monkeypatch.setattr(encoder, "decode_bio", lambda logits: [2])  # I-PERSON at start
    model, _ = adapter(FakeTokenizer(tokens([(0, 4)])))
    artifact = model.extract("bad-bio", "fern")
    assert artifact.result.failure.code == "invalid_output"
    assert artifact.result.extraction is None
    assert artifact.diagnostics["tags"] == [2]


def test_zero_width_and_special_rows_excluded_only_after_offset_policy():
    tokenizer = FakeTokenizer(tokens([(0, 0), (0, 1)]))
    model, _ = adapter(tokenizer)
    artifact = model.extract("zero", "x")
    assert artifact.result.status == "ok"
    assert artifact.diagnostics["token_count"] == 4
    assert len(artifact.diagnostics["offset_policy"]["transformed_offsets"]) == 4
    assert artifact.diagnostics["decoded_token_count"] == 1


def test_no_active_token_rows_are_a_valid_empty_extraction():
    model, _ = adapter(FakeTokenizer(tokens([])))
    artifact = model.extract("no-active-rows", "🪶")
    assert artifact.result.status == "ok"
    assert artifact.result.extraction.entities == []
    assert artifact.diagnostics["decoded_token_count"] == 0


def test_overlapping_unicode_pieces_can_merge_within_one_entity():
    logits = [[0.0] * 9 for _ in range(4)]
    logits[1][1] = 8.0
    logits[2][2] = 8.0
    model, _ = adapter(FakeTokenizer(tokens([(0, 1), (0, 1)])), FakeRuntime(logits))
    artifact = model.extract("shared-codepoint", "🪶")
    assert artifact.result.status == "ok"
    assert artifact.result.extraction.entities[0].text == "🪶"


def test_returned_artifact_cannot_mutate_live_identity():
    model, _ = adapter(FakeTokenizer(tokens([(0, 4)])))
    artifact = model.extract("first", "fern")
    artifact.identity.labels[0] = "modified"
    artifact.identity.base_files.clear()
    second = model.extract("second", "fern")
    assert second.identity == retained_identity()


def test_public_loader_has_no_identity_or_factory_switch(monkeypatch, tmp_path):
    assert list(inspect.signature(encoder.load_encoder).parameters) == ["root"]
    calls = []
    monkeypatch.setattr(
        encoder, "_load_encoder", lambda root, identity: calls.append((root, identity))
    )
    encoder.load_encoder(tmp_path)
    assert calls[0][0] == tmp_path
    assert calls[0][1] == retained_identity()
    with pytest.raises(TypeError):
        encoder.Encoder()


@pytest.fixture
def fake_assets(tmp_path):
    """Only invented bytes/JSON; this fixture never reads a retained asset."""
    base = tmp_path / "assets/base"
    component = tmp_path / "assets/component"
    base.mkdir(parents=True)
    component.mkdir()
    config = {
        "model_type": "xlm-roberta",
        "architectures": ["XLMRobertaForMaskedLM"],
        "hidden_size": 4,
        "num_hidden_layers": 1,
    }
    contents = {
        "config.json": json.dumps(config).encode(),
        "model.safetensors": b"fictional safetensors placeholder\x00\xff",
        "sentencepiece.bpe.model": b"fictional tokenizer placeholder",
        "tokenizer.json": b'{"fictional": true}',
        "tokenizer_config.json": b"{}",
    }
    for name, data in contents.items():
        (base / name).write_bytes(data)
    files = {
        name: FileIdentity(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for name, data in contents.items()
    }
    receipt = {
        "repo": "fictional/test",
        "revision": "fictional-revision",
        "directory": "assets/base",
        "total_bytes": sum(f.bytes for f in files.values()),
        "files": [
            {"file": name, "bytes": item.bytes, "sha256": item.sha256, "verified": True}
            for name, item in files.items()
        ],
    }
    receipt_path = tmp_path / "assets/receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    checkpoint = component / "block-head.pt"
    checkpoint.write_bytes(b"fictional component, not Torch data\x00")
    checkpoint_digest = encoder._sha256(checkpoint)
    metadata = {
        "format": "nlp3-xlmr-final-block-bio-head-v1",
        "labels": list(LABELS),
        "hidden_size": 4,
        "final_block_index": 0,
        "checkpoint_sha256": checkpoint_digest,
        "settings": {
            "max_length": 256,
            "threads": 2,
            "device": "cpu",
            "dtype": "float32",
            "offset_policy": encoder.OFFSET_POLICY,
            "decoder": encoder.DECODER_POLICY,
            "encoder_trainable": "final_block_only",
            "encoder_mode": "eval",
            "block_dropout": "disabled_eval_mode",
        },
        "base_model": {
            "model_files_sha256": {k: v.sha256 for k, v in files.items()},
            "config": config,
            "verified_download_identity": {
                "repo": "fictional/test",
                "revision": "fictional-revision",
                "receipt_sha256": encoder._sha256(receipt_path),
                "verification": "pinned_five_file_sizes_and_sha256_plus_receipt_metadata",
            },
        },
    }
    metadata_path = component / "block-head-config.json"
    metadata_path.write_text(json.dumps(metadata))
    identity = EncoderIdentity(
        repo="fictional/test",
        revision="fictional-revision",
        base_directory="assets/base",
        base_files=files,
        receipt_path="assets/receipt.json",
        receipt_sha256=encoder._sha256(receipt_path),
        component_directory="assets/component",
        component_config_sha256=encoder._sha256(metadata_path),
        component_sha256=checkpoint_digest,
        hidden_size=4,
        final_block_index=0,
        labels=list(LABELS),
    )
    return tmp_path, identity, base, component


def test_fictional_asset_verification_and_chunked_byte_hash(fake_assets, tmp_path):
    root, identity, base, component = fake_assets
    found_base, found_component, config = encoder._verify_assets(root, identity)
    assert (found_base, found_component, config["hidden_size"]) == (base, component, 4)
    path = tmp_path / "chunk-test.bin"
    data = b"\x00\xff\r\n" * (1024 * 1024 + 3)
    path.write_bytes(data)
    assert encoder._sha256(path) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "target",
    [
        "assets/base/model.safetensors",
        "assets/base/tokenizer.json",
        "assets/component/block-head.pt",
        "assets/component/block-head-config.json",
        "assets/receipt.json",
    ],
)
def test_mutable_adjacent_bytes_cannot_redefine_identity(fake_assets, target):
    root, identity, _, _ = fake_assets
    path = root / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|sized"):
        encoder._verify_assets(root, identity)


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
def test_same_size_base_file_corruption_fails_only_the_hash_check(fake_assets, name):
    root, identity, _, _ = fake_assets
    # Positive control with the untouched fictional fixture: unrelated fixture
    # damage cannot be mistaken for the corruption under test.
    encoder._verify_assets(root, identity)
    expected = identity.base_files[name]
    path = root / identity.base_directory / name
    original = path.read_bytes()
    assert len(original) == expected.bytes
    assert expected.sha256 == hashlib.sha256(original).hexdigest()
    corrupted = bytearray(original)
    corrupted[len(corrupted) // 2] ^= 0x01
    path.write_bytes(corrupted)
    # The byte length (and therefore the size precheck) is unchanged, so only
    # the content digest can reject this file. The trusted expected identity is
    # still the caller's original one; it is not rebound to the corrupt bytes.
    assert path.stat().st_size == expected.bytes
    assert path.read_bytes() != original
    assert identity.base_files[name] == expected
    with pytest.raises(
        ValueError, match=re.escape(f"encoder asset hash mismatch: {name}")
    ):
        encoder._verify_assets(root, identity)


@pytest.mark.parametrize(
    "unexpected",
    [
        "extra.bin",
        "custom.py",
        ".cache/huggingface/download/code.py",
    ],
)
def test_unexpected_assets_fail_closed(fake_assets, unexpected):
    root, identity, base, _ = fake_assets
    path = base / unexpected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fictional")
    with pytest.raises(ValueError, match="unexpected"):
        encoder._verify_assets(root, identity)


def test_symlinked_asset_rejected_even_when_bytes_match(fake_assets):
    root, identity, base, _ = fake_assets
    file = base / "tokenizer.json"
    destination = root / "same-bytes.json"
    destination.write_bytes(file.read_bytes())
    file.unlink()
    file.symlink_to(destination)
    with pytest.raises(ValueError, match="symlink"):
        encoder._verify_assets(root, identity)


def test_bookkeeping_preserved_without_reading_contents(fake_assets, monkeypatch):
    root, identity, base, _ = fake_assets
    bookkeeping = base / ".cache/huggingface/download/model.safetensors.metadata"
    bookkeeping.parent.mkdir(parents=True)
    bookkeeping.write_bytes(b"opaque bookkeeping, not model input")
    original = encoder._sha256

    def hash_without_cache(path):
        assert ".cache" not in path.parts
        return original(path)

    monkeypatch.setattr(encoder, "_sha256", hash_without_cache)
    encoder._verify_assets(root, identity)
    assert bookkeeping.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"labels": list(reversed(LABELS))},
        {"hidden_size": 5},
        {"final_block_index": True},
        {"checkpoint_sha256": "f" * 64},
        {"base_model": {}},
        {"extra_component": "forbidden"},
        {"settings": {}},
        {"auto_map": {"AutoModel": "malicious.py"}},
    ],
)
def test_known_metadata_hash_does_not_bypass_semantic_contract(fake_assets, change):
    root, identity, _, component = fake_assets
    path = component / "block-head-config.json"
    metadata = json.loads(path.read_text())
    metadata.update(change)
    path.write_text(json.dumps(metadata))
    # A trusted tiny fixture may specify different hashes, but not different
    # labels/assembly rules. The public loader cannot accept this identity.
    updated = identity.model_copy(
        update={"component_config_sha256": encoder._sha256(path)}
    )
    with pytest.raises(ValueError):
        encoder._verify_assets(root, updated)


@pytest.mark.parametrize(
    "metadata",
    [
        {"auto_map": {"AutoTokenizer": "external.Custom"}},
        {"tokenizer_class": "ExternalTokenizer"},
        {"tokenizer_file": "/outside/tokenizer.json"},
    ],
)
def test_tokenizer_hooks_and_asset_redirection_fail_before_runtime_import(
    fake_assets, metadata
):
    root, identity, base, component = fake_assets
    path = base / "tokenizer_config.json"
    path.write_text(json.dumps(metadata))
    files = dict(identity.base_files)
    files[path.name] = FileIdentity(
        bytes=path.stat().st_size, sha256=encoder._sha256(path)
    )
    receipt_path = root / identity.receipt_path
    receipt = json.loads(receipt_path.read_text())
    receipt["total_bytes"] = sum(item.bytes for item in files.values())
    for row in receipt["files"]:
        item = files[row["file"]]
        row.update(bytes=item.bytes, sha256=item.sha256)
    receipt_path.write_text(json.dumps(receipt))
    component_path = component / "block-head-config.json"
    component_metadata = json.loads(component_path.read_text())
    component_metadata["base_model"]["model_files_sha256"] = {
        name: item.sha256 for name, item in files.items()
    }
    receipt_digest = encoder._sha256(receipt_path)
    component_metadata["base_model"]["verified_download_identity"]["receipt_sha256"] = (
        receipt_digest
    )
    component_path.write_text(json.dumps(component_metadata))
    updated = identity.model_copy(
        update={
            "base_files": files,
            "receipt_sha256": receipt_digest,
            "component_config_sha256": encoder._sha256(component_path),
        }
    )
    with pytest.raises(ValueError, match="hooks|class|redirection"):
        encoder._verify_assets(root, updated)


@pytest.mark.parametrize("data", [b'{"a": 1, "a": 2}', b'{"a": NaN}', b'{"a": 1e9999}'])
def test_ambiguous_or_nonfinite_metadata_is_rejected(tmp_path, data):
    path = tmp_path / "fictional.json"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        encoder._json(path)
