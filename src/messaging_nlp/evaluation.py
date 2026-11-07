"""Stage 4B pure scoring kernels (model-free, fictional-safe).

These kernels consume in-memory contracts and plain dictionaries only. They
never import optional model runtimes, read research data, load weights, or
compute study-quality values; the orchestrator owns the blinded-review bindings,
the actual reproduction integration and final acceptance. Semantic correctness
(book roles, prose meaning, authorship, true blocker/reason) is intentionally
NOT inferred here: structure scoring only compares explicitly representable,
deterministic facts and leaves the rest to the frozen semantic-review protocol.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter

from messaging_nlp.contracts import Entity, ExtractionResult, RunRow, clock

# Canonical human-review status vocabulary (Stage 1 decisions, normalized here).
_PENDING = "pending"
_KNOWN_PENDING_STATUSES = frozenset(
    {"pending", "pending_not_human_annotation", "pending_not_human_annotated"}
)

# Fixed condition presentation order; unknown labels sort after these.
_CANONICAL_CONDITIONS = ("baseline", "B", "A+B")


def normalize_human_review(value: str) -> str:
    """Normalize a retained human-review label to the frozen status vocabulary.

    The three retained ``pending``/``pending_not_human_annotation``/
    ``pending_not_human_annotated`` spellings are recognized; all
    mean "not human-annotated" and normalize to ``"pending"``. Any other value,
    including empty strings and historical spellings not enumerated here, fails
    closed with :class:`ValueError` rather than being mistaken for completion.
    """
    if value in _KNOWN_PENDING_STATUSES:
        return _PENDING
    raise ValueError(f"unknown human-review status: {value!r}")


def _validate_spans(text: str, entities: list[Entity], label: str) -> None:
    """Reject spans that do not slice the exact input text (fail closed)."""
    for entity in entities:
        if entity.start < 0 or entity.end <= entity.start or entity.end > len(text):
            raise ValueError(f"{label} span out of bounds: {entity.start}:{entity.end}")
        if text[entity.start : entity.end] != entity.text:
            raise ValueError(
                f"{label} span text mismatch for type {entity.type!r}: "
                f"{entity.start}:{entity.end}"
            )


def score_extraction(
    text: str, gold: list[Entity], result: ExtractionResult | None
) -> dict:
    """Score one extraction against exact Unicode (start, end, type) gold spans.

    Returns a JSON-compatible dictionary with keys:

    - ``status``: ``"measured"`` or ``"unmeasured"``. A missing or failed
      prediction (``None`` or ``status != "ok"``) is *unmeasured* with null
      counts/rates, NOT an empty extraction. A valid empty extraction
      (``status == "ok"`` with no entities) is measured.
    - ``tp``, ``fp``, ``fn``: integer counts (or ``None`` when unmeasured).
      A prediction matches gold on the exact ``(start, end, type)`` tuple;
      ``text`` is not part of the match key, but every gold and predicted span
      must slice the supplied ``text`` exactly or this function raises
      :class:`ValueError` (fail closed).
    - ``precision``, ``recall``, ``f1``: floats (or ``None`` when unmeasured).
      Valid both-empty gives ``tp=fp=fn=0`` and all rates ``1.0``. Other
      undefined denominators follow the ``zero_division=0`` convention, so an
      empty prediction against nonempty gold scores recall/precision ``0.0``.

    Baseline is not scored and B/A+B share one extraction, so the orchestrator
    calls this once per request for the shared B/A+B result.
    """
    _validate_spans(text, gold, "gold")
    if (
        result is not None
        and result.input_sha256 != hashlib.sha256(text.encode()).hexdigest()
    ):
        raise ValueError("extraction result differs from exact input text")
    if result is None or result.status != "ok" or result.extraction is None:
        return {
            "status": "unmeasured",
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    predicted = list(result.extraction.entities)
    _validate_spans(text, predicted, "predicted")

    gold_bag = Counter((e.start, e.end, e.type) for e in gold)
    predicted_bag = Counter((e.start, e.end, e.type) for e in predicted)
    tp = sum((gold_bag & predicted_bag).values())
    fp = len(predicted) - tp
    fn = len(gold) - tp

    if tp == 0 and fp == 0 and fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )

    return {
        "status": "measured",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _expected_start_instant(fact: dict):
    """Resolve the authoritative absolute instant of an event-start requirement."""
    instant = fact.get("instant_utc")
    if instant is not None:
        return clock(instant)
    return clock(fact["source_value"])


def score_structure(row: RunRow | None, requirement: dict) -> dict:
    """Score deterministic decision structure against one requirement.

    Returns a JSON-compatible dictionary with four top-level fields, each one of
    the strings ``"pass"``, ``"fail"``, ``"unmeasured"`` or ``"not_applicable"``:

    - ``action``: pass iff the saved ``decision.action`` equals
      ``expected_action``. A false clarification (a clarification where the
      expected action is not ``clarify``) fails here. Missing/invalid output is
      ``unmeasured``, never a correct action.
    - ``recipient``: draft-recipient equality. Only applicable when
      ``expected_action == "draft"``; otherwise ``not_applicable`` regardless of
      the eventual-draft ``recipient_contact_ids`` metadata. A missing expected
      draft selection fails.
    - ``selected_event``: exact selected-event ID equality, from the
      ``selected_event`` normalized fact. ``not_applicable`` when the
      requirement has no selected event.
    - ``event_start``: explicit event-start instant equality, from the
      ``event_start`` normalized fact, compared on the absolute UTC instant so
      equivalent offsets match. ``not_applicable`` when the requirement has no
      event start.

    Only these explicitly representable normalized facts are deterministic.
    Book roles, prose meaning, authorship and true blocker/reason require
    semantic review and are never inferred here.
    """
    has_output = row is not None and row.status == "ok"
    expected_action = requirement["expected_action"]
    facts = requirement.get("normalized_facts") or []
    expected_event = next((f for f in facts if f.get("kind") == "selected_event"), None)
    expected_start = next((f for f in facts if f.get("kind") == "event_start"), None)

    if not has_output:
        action = "unmeasured"
    elif row.decision.action == expected_action:
        action = "pass"
    else:
        action = "fail"

    if expected_action != "draft":
        recipient = "not_applicable"
    elif not has_output:
        recipient = "unmeasured"
    else:
        selection = row.decision.selection
        if row.decision.action != "draft" or selection is None:
            recipient = "fail"
        elif set(selection.recipient_contact_ids) == set(
            requirement["recipient_contact_ids"]
        ):
            recipient = "pass"
        else:
            recipient = "fail"

    if expected_event is None:
        selected_event = "not_applicable"
    elif not has_output:
        selected_event = "unmeasured"
    else:
        selection = row.decision.selection
        if row.decision.action != "draft" or selection is None:
            selected_event = "fail"
        elif selection.selected_event_id == expected_event["event_id"]:
            selected_event = "pass"
        else:
            selected_event = "fail"

    if expected_start is None:
        event_start = "not_applicable"
    elif not has_output:
        event_start = "unmeasured"
    else:
        selection = row.decision.selection
        if row.decision.action != "draft" or selection is None:
            event_start = "fail"
        else:
            expected_instant = _expected_start_instant(expected_start)
            start_facts = [f for f in selection.facts if f.field == "start"]
            event_start = (
                "pass"
                if any(clock(f.value) == expected_instant for f in start_facts)
                else "fail"
            )

    return {
        "action": action,
        "recipient": recipient,
        "selected_event": selected_event,
        "event_start": event_start,
    }


def evidence_coverage(
    candidate_record_ids: list[str] | None, alternatives: list[list[str]]
) -> dict:
    """Measure how completely candidate evidence covers the support alternatives.

    Returns ``{"status", "complete", "best_fraction"}``:

    - ``status``: ``"measured"`` or ``"unmeasured"``. A missing candidate result
      (``None``) is unmeasured with null ``complete``/``best_fraction``.
    - ``complete``: ``True`` iff any sufficient source set (alternative) is a
      subset of the candidate IDs. An empty alternative is valid: it needs no
      additional records and therefore makes coverage complete.
    - ``best_fraction``: the best proportion of any alternative's required
      records present in the candidates (``1.0`` for a satisfied/empty
      alternative), reported separately from the all-or-nothing ``complete``.
    """
    if candidate_record_ids is None:
        return {"status": "unmeasured", "complete": None, "best_fraction": None}
    candidates = set(candidate_record_ids)
    complete = False
    best_fraction = 0.0
    for alternative in alternatives:
        required = set(alternative)
        if required <= candidates:
            complete = True
        fraction = 1.0 if not required else len(required & candidates) / len(required)
        best_fraction = max(best_fraction, fraction)
    return {
        "status": "measured",
        "complete": complete,
        "best_fraction": best_fraction,
    }


def _condition_sort_key(condition: str) -> tuple[int, object]:
    try:
        return (0, _CANONICAL_CONDITIONS.index(condition))
    except ValueError:
        return (1, condition)


def _percentile_linear(sorted_values: list[float], q: float) -> float:
    """Return the ``q`` quantile using numpy's ``linear`` interpolation."""
    count = len(sorted_values)
    if count == 0:
        raise ValueError("cannot compute a percentile of an empty distribution")
    if count == 1:
        return sorted_values[0]
    position = (count - 1) * q
    lower = int(position)
    fraction = position - lower
    if lower >= count - 1:
        return sorted_values[count - 1]
    return sorted_values[lower] * (1 - fraction) + sorted_values[lower + 1] * fraction


def _paired_contrast(
    scenario_variant_rows: dict[tuple[str, str], dict[str, bool | None]],
    scenario_variant_family: dict[tuple[str, str], str],
    first: str,
    second: str,
    shared_resamples: list[list[str]],
) -> dict:
    resolved: list[tuple[str, int]] = []
    incomplete = 0
    for pair, by_condition in scenario_variant_rows.items():
        first_present = first in by_condition
        second_present = second in by_condition
        first_measured = first_present and by_condition[first] is not None
        second_measured = second_present and by_condition[second] is not None
        if first_measured and second_measured:
            difference = (1 if by_condition[first] else 0) - (
                1 if by_condition[second] else 0
            )
            resolved.append((scenario_variant_family[pair], difference))
        else:
            incomplete += 1

    result = {
        "condition_a": first,
        "condition_b": second,
        "complete_pairs": len(resolved),
        "incomplete_pairs": incomplete,
        "represented_families": 0,
        "bootstrap_samples": 0,
        "mean_difference": None,
        "ci_lower": None,
        "ci_upper": None,
    }
    if not resolved:
        return result

    result["mean_difference"] = sum(d for _, d in resolved) / len(resolved)
    represented = sorted({family for family, _ in resolved})
    result["represented_families"] = len(represented)
    if len(represented) < 2:
        # Fewer than two represented families => interval is null; the point
        # estimate over the resolved pairs remains reported.
        return result

    family_sum: Counter[str] = Counter()
    family_count: Counter[str] = Counter()
    for family, difference in resolved:
        family_sum[family] += difference
        family_count[family] += 1

    bootstrap: list[float] = []
    for resample in shared_resamples:
        weights = Counter(resample)
        total = 0.0
        denominator = 0
        for family in represented:
            weight = weights.get(family, 0)
            if weight:
                total += weight * family_sum[family]
                denominator += weight * family_count[family]
        if denominator:
            bootstrap.append(total / denominator)

    if bootstrap:
        result["bootstrap_samples"] = len(bootstrap)
        ordered = sorted(bootstrap)
        result["ci_lower"] = _percentile_linear(ordered, 0.025)
        result["ci_upper"] = _percentile_linear(ordered, 0.975)
    return result


def paired_summary(
    rows: list[dict], *, repetitions: int = 2000, seed: int = 42
) -> dict:
    """Summarize paired condition outcomes with a shared family-cluster bootstrap.

    ``rows`` are plain dictionaries with keys ``scenario_id``, ``family_id``,
    ``variant_type``, ``condition`` and ``success`` (``bool`` or ``None``).
    Duplicate ``(scenario_id, variant_type, condition)`` triples and
    inconsistent family linkage (a scenario or scenario/variant mapped to more
    than one family) are rejected with :class:`ValueError`.

    Returns a JSON-compatible dictionary:

    - ``conditions``: per condition ``planned``, ``measured``, ``successful``
      and ``unmeasured`` counts, ``credited_success_coverage``
      (``successful / planned``) and ``measured_rate``
      (``successful / measured``, ``None`` when nothing was measured).
      Unexecuted/unmeasured work is never labelled a measured zero.
    - ``contrasts``: one entry per unordered condition pair, keyed
      ``"<condition_a>_vs_<condition_b>"``, with ``condition_a``/``condition_b``,
      ``complete_pairs`` (same scenario/variant resolved on both sides),
      ``incomplete_pairs``, ``represented_families``, ``mean_difference``
      (mean of ``success[condition_a] - success[condition_b]`` over resolved
      pairs) and the bootstrap ``ci_lower``/``ci_upper``.
    - ``repetitions`` and ``seed`` echo the bootstrap settings.

    The interval is a cluster bootstrap over families: families are sampled
    uniformly with replacement (size = number of families), every paired
    scenario/variant row within a sampled family is preserved, and the same
    resamples are shared across all contrasts. It is a percentile-95% interval
    (2.5-97.5, linear interpolation) over ``repetitions`` resamples from
    ``random.Random(seed)``. No complete pairs => null estimate and interval;
    fewer than two represented families => null interval. Five source families
    limit uncertainty interpretation, not an independence guarantee.
    """
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    required = {"scenario_id", "family_id", "variant_type", "condition", "success"}
    seen_triples: set[tuple[str, str, str]] = set()
    scenario_variant_family: dict[tuple[str, str], str] = {}
    scenario_family: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not a mapping")
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} missing keys: {sorted(missing)}")
        if row["success"] is not None and type(row["success"]) is not bool:
            raise ValueError(f"row {index} success must be bool or None")
        if row["condition"] not in _CANONICAL_CONDITIONS:
            raise ValueError("condition is outside the frozen three-arm comparison")
        triple = (row["scenario_id"], row["variant_type"], row["condition"])
        if triple in seen_triples:
            raise ValueError(f"duplicate scenario/variant/condition: {triple!r}")
        seen_triples.add(triple)
        pair = (row["scenario_id"], row["variant_type"])
        previous_family = scenario_variant_family.setdefault(pair, row["family_id"])
        if previous_family != row["family_id"]:
            raise ValueError(
                f"inconsistent family linkage for scenario/variant {pair!r}"
            )
        previous_scenario_family = scenario_family.setdefault(
            row["scenario_id"], row["family_id"]
        )
        if previous_scenario_family != row["family_id"]:
            raise ValueError(
                f"inconsistent family linkage for scenario {row['scenario_id']!r}"
            )

    conditions = sorted({row["condition"] for row in rows}, key=_condition_sort_key)
    families = sorted({row["family_id"] for row in rows})

    condition_summaries = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        planned = len(condition_rows)
        measured = sum(1 for row in condition_rows if row["success"] is not None)
        successful = sum(1 for row in condition_rows if row["success"] is True)
        unmeasured = planned - measured
        condition_summaries[condition] = {
            "planned": planned,
            "measured": measured,
            "successful": successful,
            "unmeasured": unmeasured,
            "credited_success_coverage": successful / planned if planned else None,
            "measured_rate": successful / measured if measured else None,
        }

    scenario_variant_rows: dict[tuple[str, str], dict[str, bool | None]] = {}
    for row in rows:
        pair = (row["scenario_id"], row["variant_type"])
        scenario_variant_rows.setdefault(pair, {})[row["condition"]] = row["success"]

    random_source = random.Random(seed)
    shared_resamples = (
        [
            [random_source.choice(families) for _ in range(len(families))]
            for _ in range(repetitions)
        ]
        if families
        else []
    )

    contrasts = {}
    for first_index in range(len(conditions)):
        for second_index in range(first_index + 1, len(conditions)):
            first = conditions[first_index]
            second = conditions[second_index]
            key = f"{first}_vs_{second}"
            contrasts[key] = _paired_contrast(
                scenario_variant_rows,
                scenario_variant_family,
                first,
                second,
                shared_resamples,
            )

    return {
        "conditions": condition_summaries,
        "contrasts": contrasts,
        "repetitions": repetitions,
        "seed": seed,
    }
