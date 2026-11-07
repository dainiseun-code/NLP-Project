"""Check public data against the original manifest, without private provenance.

This does not verify the omitted files or establish study-result correctness.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data/stage1-v1"
PRIVATE_PREFIX = "evaluator/provenance/"


def verify_published_data(export):
    manifest = json.loads((export / "selection-manifest.json").read_text())
    expected = {
        name: item
        for name, item in manifest["files"].items()
        if not name.startswith(PRIVATE_PREFIX)
    }
    actual = {
        path.relative_to(export).as_posix()
        for path in export.rglob("*")
        if path.is_file()
        and path != export / "selection-manifest.json"
        and not path.relative_to(export).as_posix().startswith(PRIVATE_PREFIX)
    }
    assert actual == set(expected), "public data file coverage differs"
    for name, receipt in expected.items():
        path = export / name
        assert not path.is_symlink() and path.resolve().is_relative_to(export.resolve())
        data = path.read_bytes()
        assert len(data) == receipt["bytes"], f"size differs: {name}"
        assert hashlib.sha256(data).hexdigest() == receipt["sha256"], name
        if path.suffix == ".jsonl":
            lines = data.splitlines()
            assert len(lines) == receipt["rows"], f"row count differs: {name}"
            assert all(isinstance(json.loads(line), dict) for line in lines)
        elif path.suffix == ".json":
            json.loads(data)


def test_published_data_match_available_original_manifest_bindings():
    verify_published_data(EXPORT)


def test_public_check_does_not_claim_to_verify_private_provenance(tmp_path):
    export = tmp_path / "stage1-v1"
    shutil.copytree(EXPORT, export)
    withheld = export / PRIVATE_PREFIX / "private-record.json"
    withheld.parent.mkdir(parents=True, exist_ok=True)
    withheld.write_text("{}")
    verify_published_data(export)


@pytest.mark.parametrize("mutation", ["missing", "changed", "extra"])
def test_public_data_check_rejects_mutation(tmp_path, mutation):
    export = tmp_path / "stage1-v1"
    shutil.copytree(EXPORT, export)
    target = export / "public/requests.jsonl"
    if mutation == "missing":
        target.unlink()
    elif mutation == "changed":
        target.write_bytes(target.read_bytes() + b"\n")
    else:
        (export / "unexpected.json").write_text("{}")
    with pytest.raises(AssertionError):
        verify_published_data(export)
