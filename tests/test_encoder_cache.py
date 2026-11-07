"""Cache requirements: exact bytes, external authority and retained failures."""

import builtins
import copy
import importlib
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from messaging_nlp.contracts import Extraction, ExtractionResult, TechnicalFailure
from messaging_nlp.encoder_cache import (
    EncoderIdentity,
    EncoderSettings,
    ExtractionArtifact,
    portable_path,
    read_artifact,
    retained_identity,
    text_sha256,
    write_artifact,
)
from messaging_nlp.io import canonical_json


def artifact(*, text="Ask Zoë 🛰️ and Zoe\u0308", failed=False):
    identity = retained_identity()
    result = ExtractionResult(
        message_id="fictional",
        status="failed" if failed else "ok",
        extraction=None if failed else Extraction(entities=[]),
        failure=TechnicalFailure(
            stage="extraction", code="runtime_error", message="raw failure"
        )
        if failed
        else None,
        raw_output="RuntimeError: raw failure" if failed else "[]",
        input_sha256=text_sha256(text),
        extractor_sha256=identity.digest(),
    )
    return ExtractionArtifact(
        text=text,
        identity=identity,
        result=result,
        diagnostics={"text_modified": False, "unresolved_marks": []},
    )


@pytest.mark.parametrize("failed", [False, True])
def test_saved_empty_success_and_failure_remain_distinct(tmp_path, failed):
    value = artifact(failed=failed)
    path = tmp_path / "result.json"
    write_artifact(path, value)
    restored = read_artifact(
        path, message_id="fictional", text=value.text, identity=value.identity
    )
    assert restored == value
    assert restored.result.status == ("failed" if failed else "ok")
    assert restored.result.raw_output == value.result.raw_output
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_artifact(path, artifact(failed=not failed))
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "text", ["Ask Zoë 🛰️ and Zoë", "Ask Zoë 🛰️ and Zoe\u0308 ", "ask Zoë 🛰️ and Zoe\u0308"]
)
def test_exact_noisy_text_invalidates_reuse(tmp_path, text):
    value = artifact()
    path = tmp_path / "result.json"
    write_artifact(path, value)
    with pytest.raises(ValueError, match="stale"):
        read_artifact(path, message_id="fictional", text=text, identity=value.identity)


def test_request_id_is_bound_even_for_identical_text(tmp_path):
    value = artifact()
    path = tmp_path / "result.json"
    write_artifact(path, value)
    with pytest.raises(ValueError, match="stale"):
        read_artifact(
            path, message_id="other", text=value.text, identity=value.identity
        )


@pytest.mark.parametrize(
    "field",
    [
        "component_sha256",
        "component_config_sha256",
        "receipt_sha256",
        "revision",
        "hidden_size",
        "final_block_index",
    ],
)
def test_external_identity_is_authority_not_artifact_itself(tmp_path, field):
    value = artifact()
    row = value.model_dump()
    row["identity"][field] = {
        "revision": "other",
        "hidden_size": 16,
        "final_block_index": 0,
    }.get(field, "a" * 64)
    changed = EncoderIdentity.model_validate(row["identity"])
    row["result"]["extractor_sha256"] = changed.digest()
    path = tmp_path / "rewritten.json"
    path.write_bytes(canonical_json(row))
    with pytest.raises(ValueError, match="stale"):
        read_artifact(
            path, message_id="fictional", text=value.text, identity=value.identity
        )


def test_tokenizer_byte_identity_invalidates_reuse(tmp_path):
    value = artifact()
    row = value.model_dump()
    row["identity"]["base_files"]["tokenizer.json"]["sha256"] = "b" * 64
    row["result"]["extractor_sha256"] = EncoderIdentity.model_validate(
        row["identity"]
    ).digest()
    path = tmp_path / "rewritten.json"
    path.write_bytes(canonical_json(row))
    with pytest.raises(ValueError, match="stale"):
        read_artifact(
            path, message_id="fictional", text=value.text, identity=value.identity
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"offset_policy": "other"},
        {"decoder_policy": "greedy"},
        {"labels": list(reversed(retained_identity().labels))},
        {"settings": {"max_tokens": 255}},
        {"settings": {"threads": 1}},
        {"settings": {"truncation": True}},
        {"settings": {"dtype": "float16"}},
        {"settings": {"device": "cuda"}},
    ],
)
def test_undeclared_policy_settings_and_labels_rejected(changes):
    row = retained_identity().model_dump()
    row.update(changes)
    with pytest.raises(ValueError):
        EncoderIdentity.model_validate(row)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(text="different"),
        lambda row: row["result"].update(input_sha256="a" * 64),
        lambda row: row["result"].update(extractor_sha256="a" * 64),
        lambda row: row.update(schema_version=True),
        lambda row: row.update(unknown="hidden"),
        lambda row: row["result"].update(status="ok", extraction=None),
        lambda row: row["result"].update(
            extraction={
                "entities": [{"type": "PERSON", "text": "wrong", "start": 0, "end": 3}]
            }
        ),
    ],
)
def test_malformed_cache_rejected_without_overwrite(tmp_path, mutation):
    value = artifact()
    row = copy.deepcopy(value.model_dump())
    mutation(row)
    path = tmp_path / "bad.json"
    path.write_bytes(canonical_json(row))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        read_artifact(
            path, message_id="fictional", text=value.text, identity=value.identity
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "data", [b"{", b'{"schema_version":1,"schema_version":1}', b'{"diagnostics":NaN}']
)
def test_invalid_json_never_regenerates_or_overwrites(tmp_path, data):
    path = tmp_path / "bad.json"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        read_artifact(
            path, message_id="fictional", text="input", identity=retained_identity()
        )
    assert path.read_bytes() == data


def test_missing_cache_is_explicit_no_execution(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_artifact(
            tmp_path / "absent",
            message_id="fictional",
            text="input",
            identity=retained_identity(),
        )
    assert list(tmp_path.iterdir()) == []


def test_write_fsyncs_file_and_parent(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("messaging_nlp.encoder_cache.os.fsync", calls.append)
    write_artifact(tmp_path / "saved.json", artifact())
    assert len(calls) == 2


def test_write_requires_caller_created_parent_directory(tmp_path):
    destination = tmp_path / "approved-run" / "nested" / "result.json"
    with pytest.raises(FileNotFoundError):
        write_artifact(destination, artifact())
    # The cache writer must not invent directories on the caller's behalf.
    assert not destination.exists()
    assert not destination.parent.exists()
    assert not (tmp_path / "approved-run").exists()
    assert list(tmp_path.iterdir()) == []
    # Directory creation is an explicit caller decision; only then does the
    # exclusive write succeed.
    destination.parent.mkdir(parents=True)
    write_artifact(destination, artifact())
    assert destination.is_file()


@pytest.mark.parametrize(
    "path",
    [
        "/absolute",
        "../outside",
        "models/../outside",
        "models//base",
        "models/./base",
        "models\\base",
        ".",
        "",
    ],
)
def test_receipt_asset_paths_are_explicit_portable_relative(path):
    with pytest.raises(ValueError, match="portable"):
        portable_path(path)


def test_retained_anchor_construction_opens_no_assets(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("asset opened")

    monkeypatch.setattr(builtins, "open", blocked)
    value = retained_identity()
    assert value.hidden_size == 768
    assert value.final_block_index == 11
    assert value.receipt_path == "models/receipts/xlmr.json"
    assert (
        value.component_config_sha256
        == "1448aafb1814250f8efc49c3891e244342031fc5bc980a09a84f85558fb1886e"
    )


def test_encoder_imports_block_optional_runtimes_and_archive():
    script = """
import importlib.abc, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','mlx','tokenizers','safetensors'} or 'Old-version' in fullname:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
models = pathlib.Path('models').resolve()
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if path == models or models in path.parents:
            raise AssertionError('forbidden model asset open: ' + str(path))
sys.addaudithook(guard)
import messaging_nlp.encoder
import messaging_nlp.encoder_decode
import messaging_nlp.encoder_cache
from messaging_nlp.encoder_cache import retained_identity
assert len(retained_identity().labels) == 9
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_json_diagnostics_reject_nonfinite_values():
    value = artifact().model_dump()
    value["diagnostics"] = {"hidden": float("nan")}
    with pytest.raises((ValueError, ValidationError)):
        ExtractionArtifact.model_validate(value)


def test_identity_digest_binds_exact_canonical_bytes():
    import hashlib

    value = retained_identity()
    assert (
        value.digest() == hashlib.sha256(canonical_json(value.model_dump())).hexdigest()
    )
    assert (
        json.loads(canonical_json(value.model_dump()))["settings"]
        == EncoderSettings().model_dump()
    )
    assert importlib.util.find_spec("messaging_nlp.encoder_cache") is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 256.0),
        ("threads", 2.0),
        ("interop_threads", 2.0),
        ("deterministic", 1),
        ("truncation", 0),
    ],
)
def test_settings_do_not_alias_booleans_and_numbers(field, value):
    with pytest.raises(ValueError):
        EncoderSettings.model_validate({field: value})


def test_identity_version_is_not_boolean_or_float():
    for version in (True, 1.0):
        row = retained_identity().model_dump()
        row["schema_version"] = version
        with pytest.raises(ValueError):
            EncoderIdentity.model_validate(row)


def test_cache_is_portable_and_does_not_open_identity_paths(tmp_path):
    value = artifact(failed=True)
    path = tmp_path / "relocated" / "retained-failure.json"
    path.parent.mkdir()
    # No models/receipt directories exist in the relocated artifact tree.
    write_artifact(path, value)
    assert (
        read_artifact(
            path, message_id="fictional", text=value.text, identity=retained_identity()
        )
        == value
    )


def test_nested_mutation_is_revalidated_before_exclusive_write(tmp_path):
    value = artifact()
    value.identity.labels.reverse()
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_artifact(path, value)
    assert not path.exists()
