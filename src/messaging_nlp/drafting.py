"""Post-selection mechanical checks; never repair or certify prose semantics."""

from __future__ import annotations

import re

from messaging_nlp.application_types import Mention, OutputEnvelope, RequestAnchor
from messaging_nlp.context import PublicBundle
from messaging_nlp.contracts import CandidateEvidence, Request
from messaging_nlp.temporal import TimezoneBinding, request_zone
from messaging_nlp.temporal_language import (
    RENDERING_POLICY as RENDERING_POLICY,
)
from messaging_nlp.temporal_language import (
    clauses,
    current_start_clauses,
    local_instant,
    parse_clock,
    parse_date,
    parse_datetime,
    proposal_clocks,
    quotation_spans,
    scan_temporals,
)


def _zone(request: Request, binding: TimezoneBinding):
    if not isinstance(binding, TimezoneBinding):
        raise ValueError("trusted timezone binding is mandatory")
    try:
        return request_zone(request, binding=binding)
    except OSError as error:
        raise ValueError("trusted timezone artifact unavailable") from error


def precompute_anchors(
    request: Request, *, binding: TimezoneBinding
) -> list[RequestAnchor]:
    """Deterministic exact-span request anchors; output text cannot create one."""
    zone = _zone(request, binding)
    local_now = local_instant(request.request_time, zone=zone)
    pending = []
    for start, end in quotation_spans(request.text):
        pending.append(
            {
                "kind": "quotation",
                "start": start,
                "end": end,
                "text": request.text[start:end],
                "value": request.text[start + 1 : end - 1],
                "scope_start": start,
                "scope_end": end,
            }
        )
    for start, end, occurrence in proposal_clocks(request.text, local_now=local_now):
        hour, minute = occurrence.value
        pending.append(
            {
                "kind": "proposal_time",
                "start": occurrence.start,
                "end": occurrence.end,
                "text": request.text[occurrence.start : occurrence.end],
                "value": f"{hour:02d}:{minute:02d}",
                "scope_start": start,
                "scope_end": end,
            }
        )
    return [
        RequestAnchor(anchor_id=f"request-anchor-{index:04d}", **fields)
        for index, fields in enumerate(sorted(pending, key=lambda item: item["start"]))
    ]


def _quote_inner(surface: str) -> str:
    if quotation_spans(surface) != [(0, len(surface))]:
        raise ValueError("faithful_quote requires one complete paired quotation")
    return surface[1:-1]


def _validate_flat_draft(
    decision,
    *,
    bundle: PublicBundle,
    anchors: list[RequestAnchor],
    zone,
) -> None:
    """Bounded temporal/event check for the flat model-facing decision.

    The model still owns recipient/event/action choice. This check only rejects
    a draft that contradicts the selected event's displayed start or asserts a
    concrete time/date with no selected event. Proposal clocks and faithful
    quotations remain provenance, not current-schedule claims.
    """
    selection = decision.selection
    draft = decision.draft
    if selection is None or draft is None:
        raise ValueError("draft requires selection and text")
    local_now = local_instant(bundle.request.request_time, zone=zone)
    # A faithful quotation proves provenance, not the current schedule. Mask the
    # quoted bytes before scanning so their source wording is not reinterpreted.
    scanned = list(draft)
    for start, end in quotation_spans(draft):
        scanned[start:end] = " " * (end - start)
    scanned = "".join(scanned)
    occurrences = scan_temporals(scanned, local_now=local_now)
    proposal_ranges = [
        (occurrence.start, occurrence.end)
        for _, _, occurrence in proposal_clocks(scanned, local_now=local_now)
    ]
    # A normal draft often addresses the recipient before the proposal ("Ren,
    # could we move supper to four pm?"). Treat that whole clause as a proposal
    # too; the clock is a suggestion, not a current-schedule assertion.
    for start, end, fragment in clauses(scanned, local_now=local_now):
        stripped = re.sub(
            r"^\s*(?:dear\s+)?[\w'’.-]+(?:[ \t]+[\w'’.-]+){0,3},\s*",
            "",
            fragment,
            flags=re.IGNORECASE,
        )
        if re.match(
            r"(?:could|can|would|should)\b.*\b(?:move|work|instead|propose)\b",
            stripped,
            re.IGNORECASE,
        ):
            proposal_ranges.append((start, end))
    request_proposals = set()
    for anchor in anchors:
        if anchor.kind == "proposal_time":
            hour, minute = anchor.value.split(":")
            request_proposals.add((int(hour), int(minute)))
    event = None
    if selection.selected_event_id is not None:
        event = next(
            (
                row
                for row in bundle.context.calendar
                if row.event_id == selection.selected_event_id
            ),
            None,
        )
        if event is None:
            raise ValueError("selected event is absent from eligible context")
    event_start = local_instant(event.start, zone=zone) if event is not None else None
    for occurrence in occurrences:
        if any(
            start <= occurrence.start and occurrence.end <= end
            for start, end in proposal_ranges
        ):
            continue
        if occurrence.kind == "clock" and occurrence.value in request_proposals:
            continue
        if event_start is None:
            raise ValueError("draft asserts a time/date without a selected event")
        if occurrence.kind == "clock":
            valid = occurrence.value == (event_start.hour, event_start.minute)
        elif occurrence.kind == "date":
            valid = occurrence.value == event_start.date()
        else:
            raise ValueError("unsupported flat temporal occurrence")
        if not valid:
            raise ValueError("draft time/date contradicts the selected event start")

    current_clauses = current_start_clauses(bundle.request.text, local_now=local_now)
    if current_clauses:
        if event is None or event.status != "scheduled":
            raise ValueError(
                "current scheduled start requires a selected scheduled event"
            )
        if not any(
            occurrence.kind == "clock"
            and occurrence.value == (event_start.hour, event_start.minute)
            for occurrence in occurrences
        ):
            raise ValueError(
                "current scheduled start requires the selected start clock"
            )
        for clause in current_clauses:
            for token in scan_temporals(clause, local_now=local_now):
                if token.kind == "clock" and token.value != (
                    event_start.hour,
                    event_start.minute,
                ):
                    raise ValueError(
                        "explicit request clock conflicts with selected current start"
                    )


def validate_envelope(
    envelope: OutputEnvelope,
    *,
    bundle: PublicBundle,
    candidates: CandidateEvidence,
    anchors: list[RequestAnchor],
    binding: TimezoneBinding,
) -> None:
    """Validate strict internal references, exact surfaces and bounded temporal prose.

    IDs must already have been strictly resolved by the envelope parser. The bundle
    must come from the eligible dataset entrypoint. There is intentionally no
    arbitrary-prose semantic verdict and no fallback/repair/clarification mutation.
    """
    zone = _zone(bundle.request, binding)
    # Strict contracts are rechecked without changing any bytes or content. Frozen
    # Pydantic models contain mutable lists; callers cannot bypass validation by
    # appending an invalid selection after initial parsing.
    OutputEnvelope.model_validate(envelope.model_dump())
    # Authenticate the precomputed anchors locally. Never create an anchor from
    # output wording or rerun the anchor-creation/preparation entrypoint. The sealed
    # byte receipt authenticates the roster; these checks also protect direct users.
    local_now = local_instant(bundle.request.request_time, zone=zone)
    request_text = bundle.request.text
    if len({a.anchor_id for a in anchors}) != len(anchors):
        raise ValueError("duplicate request anchor identity")
    request_scopes = {(a, b) for a, b, _ in clauses(request_text, local_now=local_now)}
    for anchor in anchors:
        RequestAnchor.model_validate(anchor.model_dump())
        if not (
            0
            <= anchor.scope_start
            <= anchor.start
            < anchor.end
            <= anchor.scope_end
            <= len(request_text)
        ):
            raise ValueError("request anchor bounds differ from original request")
        if request_text[anchor.start : anchor.end] != anchor.text:
            raise ValueError("request anchor text differs from original request")
        if anchor.kind == "quotation":
            if (
                (anchor.start, anchor.end) not in quotation_spans(request_text)
                or (anchor.scope_start, anchor.scope_end) != (anchor.start, anchor.end)
                or _quote_inner(anchor.text) != anchor.value
            ):
                raise ValueError("request quotation anchor is not exact")
        else:
            if (anchor.scope_start, anchor.scope_end) not in request_scopes:
                raise ValueError(
                    "proposal anchor is outside a recognized request clause"
                )
            scoped = request_text[anchor.scope_start : anchor.scope_end]
            tokens = proposal_clocks(scoped, local_now=local_now)
            if not any(
                anchor.scope_start + token.start == anchor.start
                and anchor.scope_start + token.end == anchor.end
                and f"{token.value[0]:02d}:{token.value[1]:02d}" == anchor.value
                for _, _, token in tokens
            ):
                raise ValueError("proposal anchor differs from original request scope")
    bundle.validate_decision(envelope.decision, candidates)
    if envelope.decision.action != "draft":
        if envelope.mentions:
            raise ValueError("non-draft mentions must be empty")
        return
    if not envelope.mentions:
        _validate_flat_draft(
            envelope.decision, bundle=bundle, anchors=anchors, zone=zone
        )
        return
    decision = envelope.decision
    selection = decision.selection
    draft = decision.draft
    if selection is None or draft is None:
        raise ValueError("draft requires selection and text")
    anchor_map = {anchor.anchor_id: anchor for anchor in anchors}
    positions: list[tuple[int, int, Mention]] = []
    start_expressed = False
    faithful_ranges = []
    proposal_occurrences = proposal_clocks(draft, local_now=local_now)
    for mention in envelope.mentions:
        surface = mention.surface
        # Count overlapping occurrences too: str.count would miss e.g. 'ana' in
        # 'banana'. No occurrence guessing is allowed, even for correct clocks.
        starts = [i for i in range(len(draft)) if draft.startswith(surface, i)]
        if len(starts) != 1:
            raise ValueError("mention surface must occur exactly once (representation)")
        start, end = starts[0], starts[0] + len(surface)
        if any(start < b and a < end for a, b, _ in positions):
            raise ValueError("mention surfaces overlap (representation)")
        positions.append((start, end, mention))
        if mention.anchor_id is not None:
            anchor = anchor_map.get(mention.anchor_id)
            if anchor is None:
                raise ValueError("unknown request anchor")
            if (
                mention.rendering_kind == "faithful_quote"
                and anchor.kind == "quotation"
            ):
                if surface != anchor.text or _quote_inner(surface) != anchor.value:
                    raise ValueError("request quotation is not exact")
                faithful_ranges.append((start, end))
            elif (
                mention.rendering_kind == "proposal_clock"
                and anchor.kind == "proposal_time"
            ):
                hour, minute = parse_clock(surface)
                if f"{hour:02d}:{minute:02d}" != anchor.value:
                    raise ValueError("proposal clock differs from request anchor")
                if not any(
                    token.start == start and token.end == end
                    for _, _, token in proposal_occurrences
                ):
                    raise ValueError(
                        "proposal anchor requires a locally recognized proposal clause"
                    )
            else:
                raise ValueError("request anchor cannot support this rendering kind")
            continue
        index = mention.fact_index
        if index is None or not 0 <= index < len(selection.facts):
            raise ValueError("mention fact index outside selected facts")
        fact = selection.facts[index]
        kind = mention.rendering_kind
        if kind == "literal":
            if surface != fact.value:
                raise ValueError("literal surface differs from selected fact")
        elif kind == "excerpt":
            if (
                fact.field not in ("message_text", "title", "name", "location")
                or surface not in fact.value
            ):
                raise ValueError("excerpt is not an exact selected text substring")
        elif kind == "faithful_quote":
            if fact.field != "message_text" or _quote_inner(surface) not in fact.value:
                raise ValueError("quotation differs from selected source message")
            faithful_ranges.append((start, end))
        elif kind in ("clock", "date", "datetime"):
            if fact.field not in ("start", "end"):
                raise ValueError(
                    "temporal rendering requires selected event start/end fact"
                )
            expected = local_instant(fact.value, zone=zone)
            expected_clock = (expected.hour, expected.minute)
            if kind != "date" and (expected.second or expected.microsecond):
                raise ValueError("minute grammar cannot represent source seconds")
            if kind == "clock":
                valid = parse_clock(surface) == expected_clock
            elif kind == "date":
                valid = parse_date(surface, local_now=local_now) == expected.date()
            else:
                valid = parse_datetime(surface, local_now=local_now) == (
                    expected.date(),
                    expected_clock,
                )
            if not valid:
                raise ValueError("temporal surface contradicts selected fact")
            if fact.field == "start" and kind in ("clock", "datetime"):
                start_expressed = True
        else:
            raise ValueError("proposal rendering cannot use a source fact")

    # Faithful quotations are a deliberate provenance exemption, not a generic
    # literal/excerpt exemption. Malformed temporal words inside exact quotes are
    # preserved as source text; all other draft text still gets the strict scan.
    scanned = list(draft)
    for start, end in faithful_ranges:
        scanned[start:end] = " " * (end - start)
    for token in scan_temporals("".join(scanned), local_now=local_now):
        if not any(
            a <= token.start
            and token.end <= b
            and mention.rendering_kind
            in ("clock", "date", "datetime", "proposal_clock")
            for a, b, mention in positions
        ):
            raise ValueError("unbound temporal occurrence in whole draft")

    current_clauses = current_start_clauses(bundle.request.text, local_now=local_now)
    if current_clauses:
        event = next(
            (
                row
                for row in bundle.context.calendar
                if row.event_id == selection.selected_event_id
            ),
            None,
        )
        if event is None or event.status != "scheduled":
            raise ValueError(
                "current scheduled start requires selected scheduled event"
            )
        if not start_expressed:
            raise ValueError(
                "current scheduled start requires expressed selected start clock"
            )
        expected = local_instant(event.start, zone=zone)
        for clause in current_clauses:
            for token in scan_temporals(clause, local_now=local_now):
                if token.kind == "clock" and token.value != (
                    expected.hour,
                    expected.minute,
                ):
                    raise ValueError(
                        "explicit request clock conflicts with selected current start"
                    )
