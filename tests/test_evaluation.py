"""Independent fictional Stage 4B kernel tests; no models or research inputs."""

import hashlib
import json

import pytest

from messaging_nlp.contracts import (
    Decision,
    Entity,
    ExtractionResult,
    RunRow,
)
from messaging_nlp.evaluation import (
    evidence_coverage,
    normalize_human_review,
    paired_summary,
    score_extraction,
    score_structure,
)

# ---------------------------------------------------------------------------
# Small independent fixture builders (fictional only).
# ---------------------------------------------------------------------------


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_digest(payload):
    # Independent spelling of the persisted canonical JSON request identity.
    return digest(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def entity(type, text, start, end):
    return Entity(type=type, text=text, start=start, end=end)


def extraction_result(text, entities):
    return ExtractionResult.model_validate(
        {
            "message_id": "fictional-request",
            "status": "ok",
            "extraction": {"entities": [e.model_dump() for e in entities]},
            "failure": None,
            "raw_output": "[]",
            "input_sha256": digest(text),
            "extractor_sha256": "e" * 64,
        }
    )


def draft_decision(recipient_ids=None, selected_event_id=None, facts=None):
    recipients = list(recipient_ids) if recipient_ids is not None else ["zoe"]
    facts = list(facts or [])
    fact_sources = {source for fact in facts for source in fact["source_record_ids"]}
    sources = sorted({"src-1"} | fact_sources)
    return Decision.model_validate(
        {
            "action": "draft",
            "selection": {
                "recipient_contact_ids": recipients,
                "selected_event_id": selected_event_id,
                "facts": facts,
                "source_record_ids": sources,
            },
            "draft": "Hi Zoë",
            "clarification": None,
            "reason": None,
        }
    )


def clarify_decision():
    return Decision.model_validate(
        {
            "action": "clarify",
            "selection": None,
            "draft": None,
            "clarification": "Which Zoë?",
            "reason": "ambiguous recipient",
        }
    )


def unsupported_decision():
    return Decision.model_validate(
        {
            "action": "unsupported",
            "selection": None,
            "draft": None,
            "clarification": None,
            "reason": "draft-only assistant cannot send",
        }
    )


def run_row(decision=None, status="ok"):
    text = "Ask Ren about supper"
    request = {
        "message_id": "fictional-request",
        "text": text,
        "owner_id": "owner",
        "request_time": "2026-10-27T10:00:00+00:00",
        "time_zone": "UTC",
        "context_snapshot_id": "snapshot",
    }
    extraction = None
    if status == "ok":
        extraction = {
            "message_id": "fictional-request",
            "status": "ok",
            "extraction": {"entities": []},
            "failure": None,
            "raw_output": "[]",
            "input_sha256": digest(text),
            "extractor_sha256": "e" * 64,
        }
    return RunRow.model_validate(
        {
            "schema_version": 1,
            "run_id": "fictional-run",
            "condition": "B",
            "request": request,
            "request_sha256": request_digest(request),
            "context_sha256": "c" * 64,
            "preprocessing_sha256": "d" * 64 if status == "ok" else None,
            "extraction_result": extraction,
            "candidates": {
                "contact_ids": ["zoe"],
                "event_ids": [],
                "source_record_ids": ["src-1"],
            }
            if status == "ok"
            else None,
            "status": status,
            "decision": decision,
            "failure": None
            if status == "ok"
            else {"stage": "generation", "code": "timeout", "message": "fictional"},
            "raw_output": None if status == "unexecuted" else "raw",
            "attempt": 1,
        }
    )


def draft_requirement(recipient_ids=None, normalized_facts=None):
    return {
        "expected_action": "draft",
        "recipient_contact_ids": list(recipient_ids)
        if recipient_ids is not None
        else ["zoe"],
        "normalized_facts": list(normalized_facts or []),
    }


def clarify_requirement():
    # recipient_contact_ids is eventual-draft metadata, never a score target.
    return {
        "expected_action": "clarify",
        "recipient_contact_ids": ["zoe"],
        "normalized_facts": [],
    }


EVENT_FACT = {
    "kind": "selected_event",
    "event_id": "e04",
    "source_record_ids": ["src-event"],
}

START_FACT = {
    "kind": "event_start",
    "source_value": "2026-11-03T16:00:00-07:00",
    "instant_utc": "2026-11-03T23:00:00+00:00",
    "local_value": "2026-11-03T16:00:00-07:00",
    "time_zone": "America/Vancouver",
    "source_record_ids": ["src-event"],
}


def paired_row(scenario, variant, condition, family, success):
    return {
        "scenario_id": scenario,
        "family_id": family,
        "variant_type": variant,
        "condition": condition,
        "success": success,
    }


# ---------------------------------------------------------------------------
# normalize_human_review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["pending", "pending_not_human_annotation"])
def test_known_pending_statuses_normalize(value):
    assert normalize_human_review(value) == "pending"


@pytest.mark.parametrize(
    "value",
    ["completed", "human_annotated", "approved", "", "Pending", "pending-not-human"],
)
def test_unknown_human_review_statuses_fail_closed(value):
    with pytest.raises(ValueError):
        normalize_human_review(value)


# ---------------------------------------------------------------------------
# score_extraction
# ---------------------------------------------------------------------------


def test_exact_unicode_spans_match_on_codepoints():
    text = "Zoe\u0308 tmrw"
    gold = [
        entity("PERSON", "Zoe\u0308", 0, 4),
        entity("DATE", "tmrw", 5, 9),
    ]
    scored = score_extraction(text, gold, extraction_result(text, gold))
    assert scored["status"] == "measured"
    assert (scored["tp"], scored["fp"], scored["fn"]) == (2, 0, 0)
    assert scored["precision"] == scored["recall"] == scored["f1"] == 1.0


def test_noisy_span_produces_false_positive_and_negative():
    text = "Zoe tmrw"
    gold = [entity("PERSON", "Zoe", 0, 3)]
    predicted = [entity("PERSON", "Zoe ", 0, 4)]
    scored = score_extraction(text, gold, extraction_result(text, predicted))
    assert scored["status"] == "measured"
    assert (scored["tp"], scored["fp"], scored["fn"]) == (0, 1, 1)
    assert scored["precision"] == scored["recall"] == scored["f1"] == 0.0


def test_type_mismatch_does_not_match():
    text = "Zoe"
    gold = [entity("PERSON", "Zoe", 0, 3)]
    predicted = [entity("TOPIC", "Zoe", 0, 3)]
    scored = score_extraction(text, gold, extraction_result(text, predicted))
    assert scored["status"] == "measured"
    assert (scored["tp"], scored["fp"], scored["fn"]) == (0, 1, 1)


def test_valid_empty_is_measured_but_failure_is_unmeasured():
    text = "hello"
    scored_empty = score_extraction(text, [], extraction_result(text, []))
    assert scored_empty["status"] == "measured"
    assert (scored_empty["tp"], scored_empty["fp"], scored_empty["fn"]) == (0, 0, 0)
    assert (
        scored_empty["precision"] == scored_empty["recall"] == scored_empty["f1"] == 1.0
    )

    failed = ExtractionResult.model_validate(
        {
            "message_id": "fictional-request",
            "status": "failed",
            "extraction": None,
            "failure": {"stage": "extraction", "code": "timeout", "message": "boom"},
            "raw_output": None,
            "input_sha256": digest(text),
            "extractor_sha256": "e" * 64,
        }
    )
    for scored in (
        score_extraction(text, [], failed),
        score_extraction(text, [], None),
    ):
        assert scored["status"] == "unmeasured"
        for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
            assert scored[key] is None


def test_empty_prediction_against_nonempty_gold_scores_zero():
    text = "Zoe"
    gold = [entity("PERSON", "Zoe", 0, 3)]
    scored = score_extraction(text, gold, extraction_result(text, []))
    assert (scored["tp"], scored["fp"], scored["fn"]) == (0, 0, 1)
    assert scored["recall"] == 0.0
    assert scored["precision"] == 0.0
    assert scored["f1"] == 0.0


def test_mismatched_gold_slice_fails_closed():
    with pytest.raises(ValueError):
        score_extraction("abcd", [entity("PERSON", "ZZZ", 0, 3)], None)


def test_mismatched_predicted_slice_fails_closed():
    text = "abcd"
    gold = [entity("PERSON", "ab", 0, 2)]
    result = ExtractionResult.model_validate(
        {
            "message_id": "fictional-request",
            "status": "ok",
            "extraction": {
                "entities": [{"type": "PERSON", "text": "XX", "start": 0, "end": 2}]
            },
            "failure": None,
            "raw_output": "[]",
            "input_sha256": digest(text),
            "extractor_sha256": "e" * 64,
        }
    )
    with pytest.raises(ValueError):
        score_extraction(text, gold, result)


# ---------------------------------------------------------------------------
# score_structure
# ---------------------------------------------------------------------------


def test_draft_action_and_recipient_pass():
    row = run_row(decision=draft_decision(recipient_ids=["zoe"]))
    scored = score_structure(row, draft_requirement(recipient_ids=["zoe"]))
    assert scored == {
        "action": "pass",
        "recipient": "pass",
        "selected_event": "not_applicable",
        "event_start": "not_applicable",
    }


def test_false_clarification_fails_action_and_recipient():
    row = run_row(decision=clarify_decision())
    scored = score_structure(row, draft_requirement(recipient_ids=["zoe"]))
    assert scored["action"] == "fail"
    assert scored["recipient"] == "fail"


def test_wrong_action_fails():
    row = run_row(decision=unsupported_decision())
    assert score_structure(row, draft_requirement())["action"] == "fail"


def test_non_draft_recipient_is_not_applicable():
    row = run_row(decision=clarify_decision())
    scored = score_structure(row, clarify_requirement())
    assert scored["action"] == "pass"
    assert scored["recipient"] == "not_applicable"
    assert scored["selected_event"] == "not_applicable"
    assert scored["event_start"] == "not_applicable"


def test_missing_or_failed_output_is_unmeasured():
    requirement = draft_requirement(
        recipient_ids=["zoe"], normalized_facts=[EVENT_FACT, START_FACT]
    )
    expected = {
        "action": "unmeasured",
        "recipient": "unmeasured",
        "selected_event": "unmeasured",
        "event_start": "unmeasured",
    }
    assert score_structure(None, requirement) == expected
    assert (
        score_structure(run_row(decision=None, status="failed"), requirement)
        == expected
    )


def test_selected_event_id_and_start_pass():
    facts = [
        {
            "field": "start",
            "value": "2026-11-03T16:00:00-07:00",
            "source_record_ids": ["src-event"],
        }
    ]
    row = run_row(
        decision=draft_decision(
            recipient_ids=["zoe"], selected_event_id="e04", facts=facts
        )
    )
    requirement = draft_requirement(
        recipient_ids=["zoe"], normalized_facts=[EVENT_FACT, START_FACT]
    )
    scored = score_structure(row, requirement)
    assert scored["selected_event"] == "pass"
    assert scored["event_start"] == "pass"


def test_wrong_selected_event_id_fails():
    row = run_row(
        decision=draft_decision(recipient_ids=["zoe"], selected_event_id="e99")
    )
    requirement = draft_requirement(
        recipient_ids=["zoe"], normalized_facts=[EVENT_FACT, START_FACT]
    )
    scored = score_structure(row, requirement)
    assert scored["selected_event"] == "fail"
    assert scored["event_start"] == "fail"  # no start fact at all


def test_event_start_instant_matches_across_equivalent_offsets():
    # Same absolute instant written with a different numeric offset.
    facts = [
        {
            "field": "start",
            "value": "2026-11-03T15:00:00-08:00",
            "source_record_ids": ["src-event"],
        }
    ]
    row = run_row(
        decision=draft_decision(
            recipient_ids=["zoe"], selected_event_id="e04", facts=facts
        )
    )
    requirement = draft_requirement(
        recipient_ids=["zoe"], normalized_facts=[EVENT_FACT, START_FACT]
    )
    assert score_structure(row, requirement)["event_start"] == "pass"


def test_wrong_event_start_instant_fails():
    facts = [
        {
            "field": "start",
            "value": "2026-11-04T16:00:00-07:00",
            "source_record_ids": ["src-event"],
        }
    ]
    row = run_row(
        decision=draft_decision(
            recipient_ids=["zoe"], selected_event_id="e04", facts=facts
        )
    )
    requirement = draft_requirement(
        recipient_ids=["zoe"], normalized_facts=[EVENT_FACT, START_FACT]
    )
    assert score_structure(row, requirement)["event_start"] == "fail"


def test_wrong_draft_recipient_fails():
    row = run_row(decision=draft_decision(recipient_ids=["ren"]))
    scored = score_structure(row, draft_requirement(recipient_ids=["zoe"]))
    assert scored["action"] == "pass"
    assert scored["recipient"] == "fail"


# ---------------------------------------------------------------------------
# evidence_coverage
# ---------------------------------------------------------------------------


def test_complete_when_alternative_is_subset():
    result = evidence_coverage(["a", "b", "c"], [["a", "b"], ["x"]])
    assert result == {"status": "measured", "complete": True, "best_fraction": 1.0}


def test_incomplete_reports_best_fraction():
    result = evidence_coverage(["a"], [["a", "b"], ["c"]])
    assert result["status"] == "measured"
    assert result["complete"] is False
    assert result["best_fraction"] == 0.5


def test_empty_alternative_is_valid_and_complete():
    assert evidence_coverage([], [[]])["complete"] is True
    result = evidence_coverage(["a"], [["x"], []])
    assert result["complete"] is True
    assert result["best_fraction"] == 1.0


def test_missing_candidate_result_is_unmeasured():
    assert evidence_coverage(None, [["a"]]) == {
        "status": "unmeasured",
        "complete": None,
        "best_fraction": None,
    }


def test_empty_candidates_is_measured_incomplete():
    result = evidence_coverage([], [["a"]])
    assert result["status"] == "measured"
    assert result["complete"] is False
    assert result["best_fraction"] == 0.0


# ---------------------------------------------------------------------------
# paired_summary
# ---------------------------------------------------------------------------


def test_duplicate_scenario_variant_condition_rejected():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "baseline", "f1", True),
    ]
    with pytest.raises(ValueError):
        paired_summary(rows)


def test_inconsistent_variant_family_linkage_rejected():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f2", True),
    ]
    with pytest.raises(ValueError):
        paired_summary(rows)


def test_inconsistent_scenario_family_linkage_rejected():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "typo", "baseline", "f2", True),
    ]
    with pytest.raises(ValueError):
        paired_summary(rows)


@pytest.mark.parametrize("bad", [1, 0, "yes", "no", 1.0])
def test_invalid_success_type_rejected(bad):
    with pytest.raises(ValueError):
        paired_summary([paired_row("s1", "clean", "baseline", "f1", bad)])


def test_condition_counts_and_rates_do_not_label_unmeasured_zero():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f1", False),
        paired_row("s2", "clean", "baseline", "f2", None),
        paired_row("s2", "clean", "B", "f2", True),
    ]
    summary = paired_summary(rows, repetitions=10, seed=42)
    baseline = summary["conditions"]["baseline"]
    assert (baseline["planned"], baseline["measured"]) == (2, 1)
    assert (baseline["successful"], baseline["unmeasured"]) == (1, 1)
    assert baseline["credited_success_coverage"] == 0.5
    assert baseline["measured_rate"] == 1.0
    assisted = summary["conditions"]["B"]
    assert (assisted["planned"], assisted["measured"]) == (2, 2)
    assert assisted["successful"] == 1
    assert assisted["credited_success_coverage"] == 0.5
    assert assisted["measured_rate"] == 0.5


def test_contrast_reports_complete_and_incomplete_pairs():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f1", False),
        paired_row("s2", "clean", "baseline", "f2", True),
        paired_row("s2", "clean", "B", "f2", None),
    ]
    summary = paired_summary(rows, repetitions=100, seed=42)
    contrast = summary["contrasts"]["baseline_vs_B"]
    assert contrast["complete_pairs"] == 1
    assert contrast["incomplete_pairs"] == 1
    assert contrast["mean_difference"] == 1.0
    assert contrast["represented_families"] == 1
    assert contrast["ci_lower"] is None and contrast["ci_upper"] is None


def test_no_complete_pairs_gives_null_estimate_and_interval():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f1", None),
        paired_row("s2", "clean", "baseline", "f2", None),
        paired_row("s2", "clean", "B", "f2", True),
    ]
    contrast = paired_summary(rows, repetitions=10, seed=42)["contrasts"][
        "baseline_vs_B"
    ]
    assert contrast["complete_pairs"] == 0
    assert contrast["incomplete_pairs"] == 2
    assert contrast["mean_difference"] is None
    assert contrast["ci_lower"] is None and contrast["ci_upper"] is None


def test_family_bootstrap_produces_bounded_interval():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f1", True),
        paired_row("s2", "clean", "baseline", "f2", False),
        paired_row("s2", "clean", "B", "f2", True),
    ]
    contrast = paired_summary(rows, repetitions=2000, seed=42)["contrasts"][
        "baseline_vs_B"
    ]
    assert contrast["complete_pairs"] == 2
    assert contrast["represented_families"] == 2
    assert contrast["mean_difference"] == pytest.approx(-0.5)
    assert contrast["ci_lower"] is not None and contrast["ci_upper"] is not None
    assert -1.0 <= contrast["ci_lower"] <= contrast["ci_upper"] <= 1.0


def test_fixed_seed_is_reproducible():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f1", False),
        paired_row("s2", "clean", "baseline", "f2", True),
        paired_row("s2", "clean", "B", "f2", True),
    ]
    assert paired_summary(rows, repetitions=500, seed=42) == paired_summary(
        rows, repetitions=500, seed=42
    )


def test_three_conditions_produce_three_shared_contrasts():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f1", True),
        paired_row("s1", "clean", "A+B", "f1", True),
        paired_row("s2", "clean", "baseline", "f2", False),
        paired_row("s2", "clean", "B", "f2", True),
        paired_row("s2", "clean", "A+B", "f2", True),
    ]
    summary = paired_summary(rows, repetitions=20, seed=42)
    assert set(summary["conditions"]) == {"baseline", "B", "A+B"}
    assert set(summary["contrasts"]) == {
        "baseline_vs_B",
        "baseline_vs_A+B",
        "B_vs_A+B",
    }


def test_empty_rows_yield_empty_summary():
    summary = paired_summary([], repetitions=10, seed=42)
    assert summary["conditions"] == {}
    assert summary["contrasts"] == {}


def test_both_unmeasured_still_counts_as_incomplete_planned_pair():
    rows = [
        paired_row("s", "asr", "baseline", "f", None),
        paired_row("s", "asr", "B", "f", None),
    ]
    contrast = paired_summary(rows)["contrasts"]["baseline_vs_B"]
    assert contrast["complete_pairs"] == 0
    assert contrast["incomplete_pairs"] == 1
    assert contrast["mean_difference"] is None


def test_unapproved_comparison_arm_fails_closed():
    with pytest.raises(ValueError, match="three-arm"):
        paired_summary([paired_row("s", "clean", "A-only", "f", True)])


def test_empty_extraction_cannot_be_scored_against_a_different_request_text():
    result = extraction_result("one exact noisy request", [])
    with pytest.raises(ValueError, match="exact input"):
        score_extraction("another request", [], result)


def test_retained_roster_pending_spelling_is_not_human_review():
    assert normalize_human_review("pending_not_human_annotated") == "pending"


def test_no_declared_support_alternative_does_not_prove_no_support_needed():
    assert evidence_coverage([], []) == {
        "status": "measured",
        "complete": False,
        "best_fraction": 0.0,
    }
    assert evidence_coverage([], [[]])["complete"] is True


def test_pair_seen_only_in_third_arm_is_still_planned_incomplete_work():
    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f1", False),
        paired_row("s2", "asr", "A+B", "f2", None),
    ]
    contrast = paired_summary(rows)["contrasts"]["baseline_vs_B"]
    assert contrast["complete_pairs"] == 1
    assert contrast["incomplete_pairs"] == 1


def test_undefined_cluster_draws_are_omitted_and_the_usable_count_is_disclosed():
    import random

    rows = [
        paired_row("s1", "clean", "baseline", "f1", True),
        paired_row("s1", "clean", "B", "f1", False),
        paired_row("s2", "typo", "baseline", "f2", False),
        paired_row("s2", "typo", "B", "f2", True),
        paired_row("s3", "asr", "baseline", "f3", None),
        paired_row("s3", "asr", "B", "f3", None),
    ]
    rng = random.Random(42)
    draws = [[rng.choice(["f1", "f2", "f3"]) for _ in range(3)] for _ in range(2000)]
    expected = sum(any(family != "f3" for family in draw) for draw in draws)
    contrast = paired_summary(rows)["contrasts"]["baseline_vs_B"]
    assert contrast["bootstrap_samples"] == expected
    assert 0 < expected < 2000
    assert -1 <= contrast["ci_lower"] <= contrast["ci_upper"] <= 1
