from __future__ import annotations

import hashlib

import pytest

from messaging_nlp.io import (
    canonical_json,
    loads_json,
    read_json,
    read_jsonl,
    sha256,
)
from messaging_nlp.paths import ensure_output_disjoint


def test_loads_json_rejects_duplicate_keys_at_every_depth():
    with pytest.raises(ValueError, match="duplicate JSON key: message_id"):
        loads_json('{"message_id": 1, "message_id": 2}')

    with pytest.raises(ValueError, match="duplicate JSON key: value"):
        loads_json('{"outer": [{"nested": {"value": 1, "value": 2}}]}')


@pytest.mark.parametrize(
    "text",
    [
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
        '{"value": 1e999}',
        '{"value": [0, {"deep": -1e999}]}',
    ],
)
def test_loads_json_rejects_nonfinite_numbers(text):
    with pytest.raises(ValueError, match="nonfinite JSON number"):
        loads_json(text)


@pytest.mark.parametrize("reader", [read_json, read_jsonl])
def test_readers_reject_invalid_utf8(tmp_path, reader):
    path = (
        tmp_path / "invalid.jsonl"
        if reader is read_jsonl
        else tmp_path / "invalid.json"
    )
    path.write_bytes(b'{"text": "\xff"}')

    with pytest.raises(UnicodeDecodeError):
        reader(path)


def test_read_json_accepts_utf8_and_rechecks_parsed_numbers(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text('{"text": "café", "count": 2}', encoding="utf-8")
    assert read_json(valid) == {"text": "café", "count": 2}

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": 1e999}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite JSON number"):
        read_json(nonfinite)


@pytest.mark.parametrize(
    "content",
    [
        "\n",
        " \t\n",
        '{"valid": true}\n\n{"also_valid": true}\n',
        "null\n",
        "[]\n",
        '"scalar"\n',
        "1\n",
    ],
)
def test_read_jsonl_rejects_blank_and_scalar_rows(tmp_path, content):
    path = tmp_path / "rows.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        read_jsonl(path)


def test_read_jsonl_returns_only_object_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_empty_jsonl_represents_an_empty_table(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    assert read_jsonl(path) == []


def test_jsonl_preserves_unicode_line_separators_and_crlf_rows(tmp_path):
    path = tmp_path / "unicode.jsonl"
    # U+2028/U+2029 are legal within a JSON string, not JSONL row boundaries.
    # A broad str.splitlines() would split this exact noisy request incorrectly.
    path.write_bytes('{"text":"É\u2028lo\u2029die 👩🏽‍💻"}\r\n'.encode("utf-8"))
    assert read_jsonl(path) == [{"text": "É\u2028lo\u2029die 👩🏽‍💻"}]


def test_canonical_json_is_sorted_compact_utf8_and_finite():
    value = {"z": "café", "a": [2, 1]}

    assert canonical_json(value) == b'{"a":[2,1],"z":"caf\xc3\xa9"}'
    with pytest.raises(ValueError):
        canonical_json({"value": float("inf")})


def test_sha256_streams_content_larger_than_one_chunk(tmp_path):
    content = bytes(range(256)) * 5_000
    path = tmp_path / "large.bin"
    path.write_bytes(content)

    assert sha256(path) == hashlib.sha256(content).hexdigest()


def test_output_symlink_inside_protected_tree_is_rejected(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)

    with pytest.raises(ValueError, match="overlaps protected path"):
        ensure_output_disjoint(alias / "new-output", [protected])


def test_protected_symlink_to_output_ancestor_is_rejected(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    alias = tmp_path / "protected-alias"
    alias.symlink_to(output, target_is_directory=True)

    with pytest.raises(ValueError, match="overlaps protected path"):
        ensure_output_disjoint(output / "new-output", [alias])


@pytest.mark.parametrize(
    ("output", "protected"),
    [
        ("output/new", "output"),
        ("workspace", "workspace/input"),
    ],
)
def test_output_rejects_both_overlap_directions(tmp_path, output, protected):
    with pytest.raises(ValueError, match="overlaps protected path"):
        ensure_output_disjoint(tmp_path / output, [tmp_path / protected])


def test_disjoint_output_is_allowed(tmp_path):
    ensure_output_disjoint(
        tmp_path / "runs" / "new-run",
        [tmp_path / "data", tmp_path / "reports" / "evidence.json"],
    )


def test_output_equal_to_protected_path_is_rejected(tmp_path):
    path = tmp_path / "retained.json"
    with pytest.raises(ValueError, match="overlaps protected path"):
        ensure_output_disjoint(path, [path])


def test_path_prefix_is_not_ancestry_and_protected_iterables_work(tmp_path):
    ensure_output_disjoint(
        tmp_path / "database" / "new",
        (tmp_path / name for name in ("data", "models")),
    )
