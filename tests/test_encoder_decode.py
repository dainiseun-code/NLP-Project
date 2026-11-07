"""Independent behavioral tests for the pure BIO decoder and offset policy.

These tests encode the Stage 2 contract directly and do not import archived
training code. They are software checks only; no model output is observed.
"""

import itertools
import subprocess
import sys
import unicodedata

import numpy as np
import pytest

from messaging_nlp import encoder_decode as decoder

# Declared independently of the implementation so label-order drift fails.
CONTRACT_LABELS = (
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-DATE",
    "I-DATE",
    "B-EVENT",
    "I-EVENT",
    "B-TOPIC",
    "I-TOPIC",
)
INDEX = {label: position for position, label in enumerate(CONTRACT_LABELS)}


def tag(label):
    return INDEX[label]


def valid_bio(path):
    """Independent valid-BIO predicate: I- must follow B-/I- of the same type."""
    previous = None
    for index in path:
        label = CONTRACT_LABELS[index]
        if label.startswith("I-"):
            kind = label[2:]
            if previous not in (f"B-{kind}", f"I-{kind}"):
                return False
        previous = label
    return True


def test_retained_within_entity_shared_codepoint_tokens_are_merged():
    text = "🛰️"
    assert decoder.entities_from_tags(
        text,
        [(0, 1), (0, 1), (1, 2)],
        [tag("B-TOPIC"), tag("I-TOPIC"), tag("I-TOPIC")],
    ) == [{"type": "TOPIC", "text": text, "start": 0, "end": 2}]


def path_score(logits, path):
    return float(sum(logits[position, label] for position, label in enumerate(path)))


def brute_force(logits):
    """Enumerate every valid path; report the best total and all tying paths."""
    scores = np.asarray(logits, dtype=np.float64)
    best = -np.inf
    winners = []
    for path in itertools.product(range(len(CONTRACT_LABELS)), repeat=len(scores)):
        if not valid_bio(path):
            continue
        total = path_score(scores, path)
        if total > best:
            best, winners = total, [path]
        elif total == best:
            winners.append(path)
    return best, winners


def spans(text, pieces):
    """Return exact code-point offsets for successive literal pieces."""
    offsets, cursor = [], 0
    for piece in pieces:
        start = text.index(piece, cursor)
        offsets.append((start, start + len(piece)))
        cursor = start + len(piece)
    return offsets


# ---------------------------------------------------------------------------
# Declared constants
# ---------------------------------------------------------------------------


def test_declared_constants_match_the_stage2_contract():
    assert decoder.LABELS == CONTRACT_LABELS
    assert decoder.ENTITY_TYPES == ("PERSON", "DATE", "EVENT", "TOPIC")
    assert decoder.OFFSET_POLICY == "whitespace_edges_unowned_marks_v1"
    assert decoder.DECODER_POLICY == "maximum_score_valid_BIO_path_no_gold"


# ---------------------------------------------------------------------------
# Global optimum and deterministic ties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "length"),
    [
        (0, 1),
        (817, 1),
        (5, 2),
        (11, 2),
        (42, 3),
        (2026, 3),
    ],
)
def test_decoder_matches_brute_force_global_optimum(seed, length):
    logits = np.random.default_rng(seed).normal(size=(length, len(CONTRACT_LABELS)))
    path = decoder.decode_bio(logits)
    assert len(path) == length
    assert all(type(index) is int for index in path)
    assert valid_bio(path)
    best, winners = brute_force(logits)
    assert path_score(logits, path) == pytest.approx(best)
    # Continuous random scores have no ties, so the brute-force optimum is unique.
    assert tuple(path) == winners[0]


def test_decoder_never_starts_inside_an_entity_even_when_that_logit_is_highest():
    logits = np.zeros((1, len(CONTRACT_LABELS)))
    logits[0, tag("I-PERSON")] = 100.0
    logits[0, tag("B-DATE")] = 2.0
    assert decoder.decode_bio(logits) == [tag("B-DATE")]

    long_logits = np.zeros((2, len(CONTRACT_LABELS)))
    long_logits[:, tag("I-PERSON")] = 100.0
    path = decoder.decode_bio(long_logits)
    assert path[0] not in (tag("I-PERSON"),)
    assert valid_bio(path)
    assert path[1] == tag("I-PERSON")


def test_decoder_argmax_ties_are_deterministic_first_index():
    assert decoder.decode_bio(np.zeros((3, len(CONTRACT_LABELS)))) == [0, 0, 0]

    logits = np.zeros((1, len(CONTRACT_LABELS)))
    logits[0, tag("B-PERSON")] = 5.0
    logits[0, tag("B-DATE")] = 5.0
    assert decoder.decode_bio(logits) == [tag("B-PERSON")]

    repeated = np.zeros((2, len(CONTRACT_LABELS)))
    assert decoder.decode_bio(repeated) == decoder.decode_bio(repeated)


# ---------------------------------------------------------------------------
# Input validation and empty batches
# ---------------------------------------------------------------------------


def test_empty_batch_is_valid_and_returns_no_path():
    assert decoder.decode_bio(np.zeros((0, len(CONTRACT_LABELS)))) == []


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 4)),
        np.zeros((0, len(CONTRACT_LABELS) - 1)),
        np.zeros(len(CONTRACT_LABELS)),
        np.zeros((1, len(CONTRACT_LABELS), 1)),
        np.zeros((1,), dtype=object),
        np.array([["x"] * len(CONTRACT_LABELS)]),
        np.array([["0.5"] * len(CONTRACT_LABELS)]),
        np.array([[True] * len(CONTRACT_LABELS)]),
        np.array(0.0),
    ],
)
def test_wrong_logit_shapes_and_types_are_rejected(bad):
    with pytest.raises(ValueError, match="logits"):
        decoder.decode_bio(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_logits_are_rejected(value):
    logits = np.zeros((2, len(CONTRACT_LABELS)))
    logits[1, tag("B-EVENT")] = value
    with pytest.raises(ValueError, match="logits"):
        decoder.decode_bio(logits)


def test_decoder_accepts_plain_sequences_without_mutating_numpy_input():
    assert decoder.decode_bio([[0.0] * len(CONTRACT_LABELS)]) == [0]
    assert decoder.decode_bio(np.zeros((1, len(CONTRACT_LABELS)), dtype=np.int64)) == [
        0
    ]

    logits = np.zeros((2, len(CONTRACT_LABELS)))
    before = logits.copy()
    decoder.decode_bio(logits)
    assert np.array_equal(logits, before)


# ---------------------------------------------------------------------------
# Exact spans: repeated names, emoji, decomposed marks
# ---------------------------------------------------------------------------


def test_repeated_names_and_multi_token_spans_are_exact_original_slices():
    text = "Nerina met Nerina near the pottery class"
    offsets = spans(
        text, ["Nerina", "met", "Nerina", "near", "the", "pottery", "class"]
    )
    tags = [
        tag(label)
        for label in (
            "B-PERSON",
            "O",
            "B-PERSON",
            "O",
            "O",
            "B-EVENT",
            "I-EVENT",
        )
    ]
    assert decoder.entities_from_tags(text, offsets, tags) == [
        {"type": "PERSON", "start": 0, "end": 6, "text": "Nerina"},
        {"type": "PERSON", "start": 11, "end": 17, "text": "Nerina"},
        {"type": "EVENT", "start": 27, "end": 40, "text": "pottery class"},
    ]


def test_emoji_entities_use_code_point_offsets_not_grapheme_guesses():
    text = "🛰️ Nerina met Nerina near 🪴."
    offsets = spans(text, ["🛰️", "Nerina", "met", "Nerina", "near", "🪴", "."])
    tags = [
        tag(label)
        for label in (
            "B-TOPIC",
            "O",
            "O",
            "B-PERSON",
            "O",
            "B-EVENT",
            "O",
        )
    ]
    result = decoder.entities_from_tags(text, offsets, tags)
    assert len("🛰️") == 2
    assert result[0] == {"type": "TOPIC", "start": 0, "end": 2, "text": "🛰️"}
    assert result[1] == {"type": "PERSON", "start": 14, "end": 20, "text": "Nerina"}
    assert result[2] == {"type": "EVENT", "start": 26, "end": 27, "text": "🪴"}


def test_decomposed_marks_are_recovered_but_never_normalized():
    text = "Zoe\u0308 met Zoe\u0308"
    offsets = [(0, 2), (2, 3), (5, 8), (9, 11), (11, 12)]
    audit = decoder.audit_policy_offsets(text, offsets, [0] * len(offsets))
    assert audit["transformed_offsets"] == [
        [0, 2],
        [2, 4],
        [5, 8],
        [9, 11],
        [11, 13],
    ]
    tags = [
        tag(label)
        for label in (
            "B-PERSON",
            "I-PERSON",
            "O",
            "B-PERSON",
            "I-PERSON",
        )
    ]
    result = decoder.entities_from_tags(text, audit["transformed_offsets"], tags)
    assert [entity["text"] for entity in result] == ["Zoe\u0308", "Zoe\u0308"]
    assert result[0]["text"] == text[0:4]
    assert len(result[0]["text"]) == 4
    assert result[0]["text"] != unicodedata.normalize("NFC", result[0]["text"])


def test_empty_extraction_is_success():
    assert decoder.entities_from_tags("", [], []) == []
    assert decoder.entities_from_tags("no entities here", [], []) == []


# ---------------------------------------------------------------------------
# Whitespace bridges and invalid entity structure
# ---------------------------------------------------------------------------


def test_whitespace_bridge_merges_but_non_whitespace_gap_fails():
    assert decoder.entities_from_tags(
        "a b", [(0, 1), (2, 3)], [tag("B-PERSON"), tag("I-PERSON")]
    ) == [{"type": "PERSON", "start": 0, "end": 3, "text": "a b"}]
    assert decoder.entities_from_tags(
        "a\tb", [(0, 1), (2, 3)], [tag("B-PERSON"), tag("I-PERSON")]
    ) == [{"type": "PERSON", "start": 0, "end": 3, "text": "a\tb"}]
    with pytest.raises(ValueError, match="non-whitespace"):
        decoder.entities_from_tags(
            "a!b", [(0, 1), (2, 3)], [tag("B-PERSON"), tag("I-PERSON")]
        )


@pytest.mark.parametrize(
    ("offsets", "tags", "match"),
    [
        ([(0, 4)], [tag("I-PERSON")], "continuation"),
        ([(0, 2), (2, 4)], [tag("B-PERSON"), tag("I-DATE")], "continuation"),
        ([(0, 4)], [9], "BIO label"),
        ([(0, 4)], [-1], "BIO label"),
        ([(0, 4)], [1.0], "BIO label"),
        ([(0, 4)], [True], "BIO label"),
        ([(0, 4)], ["B-PERSON"], "BIO label"),
        ([(0, 0)], [tag("O")], "offset"),
        ([(3, 1)], [tag("O")], "offset"),
        ([(0, 5)], [tag("O")], "offset"),
        ([(-1, 2)], [tag("O")], "offset"),
        ([(0.0, 4)], [tag("O")], "offset"),
        ([(False, 4)], [tag("O")], "offset"),
        ([[0]], [tag("O")], "offset"),
        ([(0, 4, 5)], [tag("O")], "offset"),
        ([(0, 4), (0, 4)], [tag("B-PERSON"), tag("B-PERSON")], "overlap"),
        ([(1, 4), (0, 1)], [tag("B-PERSON"), tag("B-PERSON")], "overlap"),
    ],
)
def test_invalid_entity_inputs_fail_explicitly(offsets, tags, match):
    with pytest.raises(ValueError, match=match):
        decoder.entities_from_tags("Maya", offsets, tags)


def test_entity_inputs_accept_numpy_integer_arrays():
    result = decoder.entities_from_tags(
        "a b",
        np.array([[0, 1], [2, 3]], dtype=np.int64),
        np.array([tag("B-PERSON"), tag("I-PERSON")], dtype=np.int64),
    )
    assert result == [{"type": "PERSON", "start": 0, "end": 3, "text": "a b"}]


def test_entity_length_mismatch_and_non_text_fail_explicitly():
    with pytest.raises(ValueError, match="length"):
        decoder.entities_from_tags("Maya", [(0, 4)], [])
    with pytest.raises(ValueError, match="text"):
        decoder.entities_from_tags(b"Maya", [(0, 4)], [tag("O")])


# ---------------------------------------------------------------------------
# Offset policy: trimming, zero-width and special tokens
# ---------------------------------------------------------------------------


def test_policy_trims_edge_whitespace_without_touching_the_text():
    text = " 🪴 Nerina \tmet Nerina "
    offsets = [[0, 2], [2, 11], [11, 14], [14, 21]]
    before = [list(pair) for pair in offsets]
    audit = decoder.audit_policy_offsets(text, offsets, [0] * len(offsets))
    assert audit["transformed_offsets"] == [[1, 2], [3, 9], [11, 14], [15, 21]]
    assert audit["text_modified"] is False
    assert offsets == before
    assert [change["action"] for change in audit["changes"]] == [
        "trim_edge_whitespace"
    ] * 3
    assert all(
        change["reason"] == "whitespace_only_token_edges" for change in audit["changes"]
    )
    assert audit["unresolved_marks"] == []
    assert audit["name"] == decoder.OFFSET_POLICY
    assert audit["limitation"]


def test_policy_keeps_whitespace_only_and_zero_width_tokens_as_empty_slices():
    text = "a b"
    audit = decoder.audit_policy_offsets(
        text, [(0, 1), (1, 2), (1, 1), (2, 3)], [0, 0, 0, 0]
    )
    assert audit["transformed_offsets"] == [[0, 1], [2, 2], [1, 1], [2, 3]]
    changes = {change["token_index"]: change for change in audit["changes"]}
    assert changes[1]["before"] == [1, 2]
    assert changes[1]["after"] == [2, 2]
    assert 2 not in changes


def test_policy_never_trims_or_owns_special_tokens():
    text = " a "
    audit = decoder.audit_policy_offsets(text, [(0, 1), (1, 2), (2, 3)], [0, 1, 0])
    assert audit["transformed_offsets"] == [[1, 1], [1, 2], [3, 3]]
    assert {change["token_index"] for change in audit["changes"]} == {0, 2}
    assert audit["unresolved_marks"] == []


def test_policy_ignores_special_ownership_but_keeps_special_offsets():
    text = "a\u0301 "
    offsets = [[0, 2], [0, 1], [2, 3], [0, 0]]
    audit = decoder.audit_policy_offsets(text, offsets, [1, 0, 0, 1])
    assert audit["transformed_offsets"] == [[0, 2], [0, 2], [3, 3], [0, 0]]
    recovery = [
        change
        for change in audit["changes"]
        if change["action"] == "recover_unowned_marks"
    ]
    assert [change["token_index"] for change in recovery] == [1]


def test_policy_accepts_boolean_mask_entries():
    audit = decoder.audit_policy_offsets("a", [(0, 1)], [True])
    assert audit["transformed_offsets"] == [[0, 1]]


# ---------------------------------------------------------------------------
# Offset policy: combining-mark recovery and diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "offsets", "expected"),
    [
        ("Zoe\u0308", [(0, 2), (2, 3)], [[0, 2], [2, 4]]),
        ("e\u0301lan", [(0, 1), (2, 5)], [[0, 2], [2, 5]]),
        ("a\u0301\u0308", [(0, 1)], [[0, 3]]),
        ("🛰️", [(0, 1)], [[0, 2]]),
        ("木\U000e0100", [(0, 1)], [[0, 2]]),
    ],
)
def test_policy_recovers_only_adjacent_marks_with_a_unique_base_owner(
    text, offsets, expected
):
    audit = decoder.audit_policy_offsets(text, offsets, [0] * len(offsets))
    assert audit["transformed_offsets"] == expected
    assert audit["unresolved_marks"] == []
    change = audit["changes"][-1]
    assert change["action"] == "recover_unowned_marks"
    assert change["reason"] == "adjacent_marks_with_unique_base_owner"
    assert type(change["base_codepoint"]) is int
    assert change["base_codepoint"] == change["mark_span"][0] - 1
    assert change["mark_span"][0] == change["before"][1]
    assert change["after"] == expected[change["token_index"]]


@pytest.mark.parametrize(
    ("text", "offsets", "reason"),
    [
        ("a\u0301", [(0, 1), (0, 1)], "ambiguous_base_ownership"),
        ("a\u0301\u0308", [(0, 2), (1, 2)], "ambiguous_mark_chain_ownership"),
        ("a\u0301b", [(0, 3), (0, 1)], None),
        ("\u0301", [(0, 0)], "unsupported_or_missing_base"),
        ("a\u0301", [(0, 0)], "missing_base_ownership"),
        ("a\u0301\u0308", [(0, 1), (1, 2)], "not_adjacent_to_base_token_end"),
    ],
)
def test_policy_never_steals_owned_marks_or_guesses_ambiguous_ownership(
    text, offsets, reason
):
    audit = decoder.audit_policy_offsets(text, offsets, [0] * len(offsets))
    assert audit["transformed_offsets"] == [list(pair) for pair in offsets]
    assert audit["changes"] == []
    if reason is None:
        assert audit["unresolved_marks"] == []
    else:
        assert [entry["reason"] for entry in audit["unresolved_marks"]] == [reason]


def test_policy_distinguishes_variation_selectors_from_skin_tone_modifiers():
    # U+FE0F is Mn and may be recovered onto a symbol base.
    audit = decoder.audit_policy_offsets("🛰️", [(0, 1)], [0])
    assert audit["transformed_offsets"] == [[0, 2]]
    assert audit["changes"][-1]["reason"] == "adjacent_marks_with_unique_base_owner"

    # U+1F3FD is Sk: never a recoverable mark, and not reported as an M* orphan.
    audit = decoder.audit_policy_offsets("👍🏽", [(0, 1)], [0])
    assert audit["transformed_offsets"] == [[0, 1]]
    assert audit["changes"] == []
    assert audit["unresolved_marks"] == []

    # A variation selector after an Sk modifier has no supported base.
    audit = decoder.audit_policy_offsets("👍🏽️", [(0, 2)], [0])
    assert audit["transformed_offsets"] == [[0, 2]]
    assert audit["changes"] == []
    assert audit["unresolved_marks"][0]["reason"] == "unsupported_or_missing_base"


def test_recovered_mark_chain_diagnostics_keep_original_base_index():
    text = "x a\u0301\u0308"
    audit = decoder.audit_policy_offsets(text, [(0, 1), (2, 4)], [0, 0])
    assert audit["transformed_offsets"] == [[0, 1], [2, 5]]
    assert audit["changes"][-1]["base_codepoint"] == 2
    assert audit["changes"][-1]["mark_span"] == [4, 5]
    assert audit["text_modified"] is False


# ---------------------------------------------------------------------------
# Offset policy: validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "offsets", "mask", "match"),
    [
        ("a", [(0, 1)], [], "length"),
        ("a", [(0, 2)], [0], "code-point"),
        ("a", [(-1, 1)], [0], "code-point"),
        ("a", [(1, 0)], [0], "code-point"),
        ("a", [[0]], [0], "offset"),
        ("a", [None], [0], "offset"),
        ("a", [(0.0, 1)], [0], "offset"),
        ("a", [(False, 1)], [0], "offset"),
        ("a", [(0, 1)], [2], "special"),
        ("a", [(0, 1)], [0.0], "special"),
        ("a", [(0, 1)], ["0"], "special"),
        (b"a", [(0, 1)], [0], "text"),
    ],
)
def test_policy_rejects_malformed_inputs_explicitly(text, offsets, mask, match):
    with pytest.raises(ValueError, match=match):
        decoder.audit_policy_offsets(text, offsets, mask)


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


def test_decoder_module_imports_no_model_or_archive_code():
    code = (
        "import sys\n"
        "import messaging_nlp.encoder_decode as module\n"
        "blocked = ('torch', 'transformers', 'tokenizers', 'safetensors')\n"
        "models = sorted(name for name in sys.modules "
        "if name.split('.')[0] in blocked)\n"
        "archived = sorted(name for name in sys.modules "
        "if name in ('messaging_nlp.encoder_head', "
        "'messaging_nlp.feasibility.encoder', 'messaging_nlp.encoder_block'))\n"
        "print(models, archived)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[] []"
