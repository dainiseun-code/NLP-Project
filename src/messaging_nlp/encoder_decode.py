"""Pure BIO decoding and source-text offset policy for the Stage 2 encoder.

This module is a self-contained extract of allowlisted archived logic: the
label order and valid-BIO decoder from ``encoder_head.py`` and the
``whitespace_edges_unowned_marks_v1`` policy from ``feasibility/encoder.py``.
It imports only the standard library and numpy: no Torch/Transformers, archived
training code, gold annotations or model state is loaded.

The decoder maximizes emission score over valid BIO paths and never consults
annotations. The offset policy is a boundary projection, not lossless token
decoding and not an entity-recognition quality claim.
"""

import unicodedata

import numpy as np

ENTITY_TYPES = ("PERSON", "DATE", "EVENT", "TOPIC")
LABELS = ("O",) + tuple(
    f"{prefix}-{kind}" for kind in ENTITY_TYPES for prefix in ("B", "I")
)
OFFSET_POLICY = "whitespace_edges_unowned_marks_v1"
DECODER_POLICY = "maximum_score_valid_BIO_path_no_gold"

_LIMITATION = "Boundary projection only; not lossless token decoding or NER quality"


def _as_sequence(value, *, name):
    """Reject strings/mappings and materialize an ordered token sequence."""
    if isinstance(value, (str, bytes, bytearray, dict, set, frozenset)) or not hasattr(
        value, "__iter__"
    ):
        raise ValueError(f"{name} must be a sequence")
    try:
        return list(value)
    except TypeError:
        raise ValueError(f"{name} must be a sequence") from None


def _code_point(value, *, what):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what}: non-integer code-point index")
    return int(value)


def _offset_pair(pair, *, what):
    if not isinstance(pair, (list, tuple, np.ndarray)):
        raise ValueError(f"{what} pair is malformed")
    if len(pair) != 2:
        raise ValueError(f"{what} pair is malformed")
    return _code_point(pair[0], what=what), _code_point(pair[1], what=what)


def _special_flag(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError("special_tokens_mask entries must be 0/1 integers or booleans")


def decode_bio(logits):
    """Return the maximum-score valid BIO path, one label index per token.

    Every retained BIO transition has equal weight, so the score is the sum of
    the per-token emissions. Labels starting with ``I-`` cannot start a
    sequence and may only follow ``B-``/``I-`` of the same type. Ties are
    deterministic: numpy argmax keeps the lowest label index.
    """
    try:
        raw = np.asarray(logits)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid token-classifier logits") from error
    if raw.dtype.kind not in "fiu":
        raise ValueError("invalid token-classifier logits")
    scores = raw.astype(np.float64, copy=False)
    if (
        scores.ndim != 2
        or scores.shape[1] != len(LABELS)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("invalid token-classifier logits")
    if scores.shape[0] == 0:
        return []

    transitions = np.zeros((len(LABELS), len(LABELS)))
    for current, label in enumerate(LABELS):
        if label.startswith("I-"):
            transitions[:, current] = -np.inf
            for previous, prior in enumerate(LABELS):
                if prior in ("B-" + label[2:], label):
                    transitions[previous, current] = 0.0

    values = scores[0].copy()
    for index, label in enumerate(LABELS):
        if label.startswith("I-"):
            values[index] = -np.inf
    parents = []
    for emissions in scores[1:]:
        candidates = values[:, None] + transitions
        parent = candidates.argmax(axis=0)
        parents.append(parent)
        values = candidates[parent, np.arange(len(LABELS))] + emissions
    path = [int(values.argmax())]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    return list(reversed(path))


def entities_from_tags(text, offsets, tags):
    """Build exact source spans from predicted BIO groups without repairs.

    ``offsets``/``tags`` must already exclude special and zero-width tokens, as
    the archived caller did. A multi-token entity may bridge whitespace only;
    any other unrepresented code point, entity overlap or bad continuation is an
    explicit failure. Returned ``text`` values are exact ``text[start:end]``
    slices with no normalization or sibling repair.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str of Unicode code points")
    offsets = _as_sequence(offsets, name="offsets")
    tags = _as_sequence(tags, name="tags")
    if len(offsets) != len(tags):
        raise ValueError("offset/tag length mismatch")

    result, current, previous_end = [], None, -1
    for pair, index in zip(offsets, tags):
        left, right = _offset_pair(pair, what="invalid prediction offset")
        if not 0 <= left < right <= len(text):
            raise ValueError("invalid prediction offset range")
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise ValueError("unknown BIO label index")
        index = int(index)
        if not 0 <= index < len(LABELS):
            raise ValueError("unknown BIO label index")

        label = LABELS[index]
        if label == "O":
            if current is not None:
                previous_end = current["end"]
                result.append(current)
                current = None
            continue
        prefix, kind = label.split("-", 1)
        if prefix == "B":
            if current is not None:
                previous_end = current["end"]
                result.append(current)
                current = None
            if left < previous_end:
                raise ValueError("predicted spans overlap")
            current = {"type": kind, "start": left, "end": right}
            continue
        if current is None or current["type"] != kind:
            raise ValueError("invalid BIO continuation")
        if left > current["end"] and not text[current["end"] : left].isspace():
            raise ValueError("entity crosses unrepresented non-whitespace")
        # Retained behavior: several subword tokens can own the same code point
        # (notably byte/Unicode token pieces). Merge within one BIO entity;
        # distinct predicted entities must still never overlap.
        current["end"] = max(current["end"], right)
    if current is not None:
        result.append(current)

    for entity in result:
        entity["text"] = text[entity["start"] : entity["end"]]
    return result


def audit_policy_offsets(text, offsets, special_mask):
    """Apply ``whitespace_edges_unowned_marks_v1`` without changing the text.

    Strip only whitespace at token edges. Recover an unowned contiguous Unicode
    ``M*`` run only at its unique base owner's token end, when no competing
    ownership exists anywhere in the base/mark chain. Symbols may own variation
    selectors; ``Sk`` skin-tone modifiers are never recovered. Every decision is
    reported; unresolved marks record why ownership could not be assigned.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str of Unicode code points")
    offsets = _as_sequence(offsets, name="offsets")
    special_mask = _as_sequence(special_mask, name="special_mask")
    if len(offsets) != len(special_mask):
        raise ValueError("Offset/mask length mismatch")

    transformed = []
    owners = [set() for _ in text]
    changes = []
    for index, (pair, mask) in enumerate(zip(offsets, special_mask)):
        left, right = _offset_pair(pair, what="invalid tokenizer offset")
        if not 0 <= left <= right <= len(text):
            raise ValueError("Tokenizer offset is not a code-point boundary")
        special = _special_flag(mask)
        original = [left, right]
        if not special:
            for position in range(left, right):
                owners[position].add(index)
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
        transformed.append([left, right])
        if original != [left, right]:
            changes.append(
                {
                    "token_index": index,
                    "action": "trim_edge_whitespace",
                    "reason": "whitespace_only_token_edges",
                    "before": original,
                    "after": [left, right],
                }
            )

    unresolved = []
    position = 0
    while position < len(text):
        if owners[position] or not unicodedata.category(text[position]).startswith("M"):
            position += 1
            continue
        start = position
        while (
            position < len(text)
            and not owners[position]
            and unicodedata.category(text[position]).startswith("M")
        ):
            position += 1
        base = start - 1
        while base >= 0 and unicodedata.category(text[base]).startswith("M"):
            base -= 1
        reason = None
        if (
            base < 0
            or unicodedata.category(text[base])[0] not in "LNS"
            or (unicodedata.category(text[base]) == "Sk")
        ):
            reason = "unsupported_or_missing_base"
        elif len(owners[base]) != 1:
            reason = (
                "ambiguous_base_ownership" if owners[base] else "missing_base_ownership"
            )
        else:
            index = next(iter(owners[base]))
            left, right = transformed[index]
            if not left <= base < right or right != start:
                reason = "not_adjacent_to_base_token_end"
            elif any(owners[p] - {index} for p in range(base, start)):
                reason = "ambiguous_mark_chain_ownership"
            else:
                transformed[index] = [left, position]
                changes.append(
                    {
                        "token_index": index,
                        "action": "recover_unowned_marks",
                        "reason": "adjacent_marks_with_unique_base_owner",
                        "base_codepoint": base,
                        "mark_span": [start, position],
                        "before": [left, right],
                        "after": [left, position],
                    }
                )
        if reason:
            unresolved.append({"mark_span": [start, position], "reason": reason})

    return {
        "name": OFFSET_POLICY,
        "transformed_offsets": transformed,
        "changes": changes,
        "unresolved_marks": unresolved,
        "text_modified": False,
        "limitation": _LIMITATION,
    }
