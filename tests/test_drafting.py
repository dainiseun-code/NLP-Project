"""Independent fictional Stage 3 examples; no models or research inputs."""

import hashlib
import struct
from datetime import datetime

import pytest

from messaging_nlp.application_types import Mention, OutputEnvelope
from messaging_nlp.context import PublicBundle, PublicContext
from messaging_nlp.contracts import CandidateEvidence, Decision, Request, Selection
from messaging_nlp.drafting import (
    RENDERING_POLICY,
    precompute_anchors,
    validate_envelope,
)
from messaging_nlp.temporal import TimezoneBinding
from messaging_nlp.temporal_language import clauses, mask_quotations, quotation_spans


@pytest.fixture
def binding(tmp_path):
    # Minimal fictional fixed-offset TZif; never use a research/host timezone file.
    data = (
        b"TZif\x00"
        + b"\x00" * 15
        + struct.pack(">6l", 0, 0, 0, 0, 1, 4)
        + struct.pack(">lbb", 0, 0, 0)
        + b"UTC\x00"
    )
    path = tmp_path / "fictional.tzif"
    path.write_bytes(data)
    return TimezoneBinding("UTC", path, hashlib.sha256(data).hexdigest())


def world(text="Ask Ren about supper"):
    request = Request(
        message_id="fictional-request",
        text=text,
        owner_id="owner",
        request_time="2026-10-27T10:00:00+00:00",
        time_zone="UTC",
        context_snapshot_id="snapshot",
    )
    context = PublicContext.model_validate(
        {
            "contacts": [
                {
                    "record_id": "contact-source",
                    "contact_id": "ren",
                    "version": 1,
                    "name": "Ren",
                    "aliases": [],
                    "relationship": "friend",
                    "available_at": "2026-10-26T00:00:00Z",
                    "provenance": "fictional",
                }
            ],
            "calendar": [
                {
                    "record_id": "event-source",
                    "event_id": "supper",
                    "version": 1,
                    "title": "Supper",
                    "participant_contact_ids": ["ren"],
                    "start": "2026-10-29T15:00:00Z",
                    "end": None,
                    "location": None,
                    "status": "scheduled",
                    "available_at": "2026-10-26T00:00:00Z",
                    "provenance": "fictional",
                }
            ],
            "history": [
                {
                    "record_id": "message-source",
                    "sender_contact_id": "ren",
                    "recipient_contact_id": "owner",
                    "text": "I read Moon Book at four pm.",
                    "timestamp": "2026-10-26T00:00:00Z",
                    "available_at": "2026-10-26T00:00:00Z",
                    "provenance": "fictional",
                }
            ],
        }
    )
    bundle = PublicBundle(request=request, context=context)
    candidates = CandidateEvidence(
        contact_ids=["ren"],
        event_ids=["supper"],
        source_record_ids=["contact-source", "event-source", "message-source"],
    )
    return bundle, candidates


def envelope(draft, mentions=(), *, facts=None, event="supper"):
    if facts is None:
        facts = [
            {
                "field": "start",
                "value": "2026-10-29T15:00:00Z",
                "source_record_ids": ["event-source"],
            }
        ]
    selection = Selection.model_validate(
        {
            "recipient_contact_ids": ["ren"],
            "selected_event_id": event,
            "facts": facts,
            "source_record_ids": ["contact-source", "event-source", "message-source"],
        }
    )
    return OutputEnvelope(
        decision=Decision(
            action="draft",
            selection=selection,
            draft=draft,
            clarification=None,
            reason=None,
        ),
        mentions=[Mention.model_validate(m) for m in mentions],
    )


def fact(surface, kind="clock", index=0):
    return {"surface": surface, "rendering_kind": kind, "fact_index": index}


@pytest.mark.parametrize("prefix", ["quarter after ", "half to ", "ten past "])
def test_attached_period_does_not_hide_recognized_unsupported_fraction(binding, prefix):
    # Binding the inner otherwise-correct clock cannot hide the unsupported prefix.
    with pytest.raises(ValueError, match="unsupported|outside.*grammar"):
        check(binding, envelope(f"Supper starts {prefix}3pm.", [fact("3pm")]))


@pytest.mark.parametrize("surface", ["003pm", "003:00", "quarter to 003pm"])
def test_overlong_numeric_clock_components_fail_closed(binding, surface):
    with pytest.raises(ValueError):
        check(binding, envelope(f"Supper starts at {surface}.", [fact(surface)]))


def check(binding, output, request="Ask Ren about supper", *, anchors=None):
    bundle, candidates = world(request)
    if anchors is None:
        anchors = precompute_anchors(bundle.request, binding=binding)
    validate_envelope(
        output, bundle=bundle, candidates=candidates, anchors=anchors, binding=binding
    )


@pytest.mark.parametrize(
    "surface",
    [
        "3 pm",
        "15:00",
        "three pm",
        "three in the afternoon",
        "three o'clock pm",
        "three fifteen pm",
    ],
)
def test_clock_renderings(binding, surface):
    # Fifteen past three is supported grammar but the wrong selected instant.
    output = envelope(f"Supper starts at {surface}.", [fact(surface)])
    if surface == "three fifteen pm":
        with pytest.raises(ValueError):
            check(binding, output)
    else:
        check(binding, output)


@pytest.mark.parametrize(
    "surface",
    [
        "4 pm",
        "four in the afternoon",
        "three pm, not four pm",
        "25:00",
        "3:99 pm",
        "thirteen pm",
        "half past three",
        "at four",
        "4",
        "three pm trailing prose",
        "three at night",
    ],
)
def test_wrong_or_malformed_full_clock_fails(binding, surface):
    with pytest.raises(ValueError):
        check(binding, envelope(f"Supper starts at {surface}.", [fact(surface)]))


@pytest.mark.parametrize(
    "extra",
    [
        "Also at 4 pm.",
        "Also four in the afternoon.",
        "Also quarter past four pm.",
        "Also at four.",
        "Also at 16.",
        "Also 2026-10-30.",
        "Also tomorrow.",
        "Also 25:75.",
        "Also next week.",
        "Also four o'clock.",
        'The phrase "four pm" is odd.',
    ],
)
def test_whole_draft_not_only_bound_surface(binding, extra):
    with pytest.raises(ValueError):
        check(
            binding, envelope("Supper starts at three pm. " + extra, [fact("three pm")])
        )


@pytest.mark.parametrize(
    "draft,mentions",
    [
        ("At three pm, then three pm.", [fact("three pm")]),
        ("At three pm.", [fact("3 pm")]),
        ("At three pm.", [fact("three pm"), fact("pm", "excerpt")]),
        ("At three pm.", [fact("three pm", index=4)]),
    ],
)
def test_surface_reference_representation_errors(binding, draft, mentions):
    with pytest.raises(ValueError):
        check(binding, envelope(draft, mentions))


@pytest.mark.parametrize("numeric", ["3 pm", "3pm"])
def test_different_unique_surfaces_for_same_supported_clock_are_accepted(
    binding, numeric
):
    output = envelope(
        f"Supper starts at three pm ({numeric}).",
        [fact("three pm"), fact(numeric)],
    )
    check(binding, output, "Confirm supper starts at three pm")


@pytest.mark.parametrize("surface", ["three pm", "3 pm", "3pm"])
def test_identical_supported_clock_surfaces_remain_representation_failure(
    binding, surface
):
    output = envelope(f"At {surface}, then {surface}.", [fact(surface)])
    before = output.model_dump()
    with pytest.raises(ValueError, match="exactly once.*representation"):
        check(binding, output)
    assert output.model_dump() == before


@pytest.mark.parametrize(
    "surface,kind",
    [
        ("2026-10-29", "date"),
        ("October 29, 2026", "date"),
        ("29 October 2026", "date"),
        ("Thursday", "date"),
        ("this Thursday", "date"),
        ("2026-10-29 at three pm", "datetime"),
        ("three pm on October 29, 2026", "datetime"),
    ],
)
def test_date_and_compound_datetime(binding, surface, kind):
    check(binding, envelope(surface, [fact(surface, kind)]))


@pytest.mark.parametrize(
    "surface",
    [
        "2026-10-30",
        "October 32, 2026",
        "February 29, 2026",
        "Wednesday",
        "tomorrow",
        "next Thursday",
        "2026-10-29 at four pm",
    ],
)
def test_wrong_dates_and_compounds(binding, surface):
    with pytest.raises(ValueError):
        check(
            binding,
            envelope(
                surface, [fact(surface, "datetime" if " at " in surface else "date")]
            ),
        )


def test_unknown_time_and_location_questions_need_no_answers(binding):
    for draft in ("Where is supper?", "What time does supper start?"):
        check(binding, envelope(draft, facts=[]), "Ask Ren where and when supper is")
    invented = [
        {
            "field": "location",
            "value": "Moon Hall",
            "source_record_ids": ["event-source"],
        }
    ]
    with pytest.raises(ValueError):
        check(binding, envelope("At Moon Hall", facts=invented))


@pytest.mark.parametrize(
    "request_text",
    [
        "Confirm supper starts at three pm",
        "Tell Ren supper starts at three pm",
        "Inform Ren of the scheduled start of supper",
        "Confirm the start time of supper",
    ],
)
def test_current_start_required_after_selection(binding, request_text):
    check(
        binding,
        envelope("Supper starts at three pm.", [fact("three pm")]),
        request_text,
    )
    for output in (
        envelope("See you!", facts=[]),
        envelope("See you!"),
        envelope("See you!", event=None, facts=[]),
    ):
        with pytest.raises(ValueError):
            check(binding, output, request_text)


def test_explicit_current_conflict_does_not_repair(binding):
    output = envelope("Supper starts at three pm.", [fact("three pm")])
    with pytest.raises(ValueError):
        check(binding, output, "Confirm supper starts at four pm")
    assert output.decision.action == "draft"
    clarification = OutputEnvelope(
        decision=Decision(
            action="clarify",
            selection=None,
            draft=None,
            clarification="Should I use the calendar start?",
            reason="The requested clock conflicts.",
        ),
        mentions=[],
    )
    check(binding, clarification, "Confirm supper starts at four pm")


def test_quote_aware_request_scope(binding):
    request = "Ask Ren about the phrase 'confirm supper starts at four pm'"
    check(binding, envelope("What does that phrase mean?", facts=[]), request)


def test_source_faithful_quote_is_provenance_not_current_fact(binding):
    facts = [
        {
            "field": "message_text",
            "value": "I read Moon Book at four pm.",
            "source_record_ids": ["message-source"],
        }
    ]
    quote = '"Moon Book at four pm"'
    check(
        binding,
        envelope("You wrote " + quote, [fact(quote, "faithful_quote")], facts=facts),
    )
    for surface, kind in [
        ('"Moon Book at three pm"', "faithful_quote"),
        ("four pm", "excerpt"),
        ("four pm", "clock"),
    ]:
        with pytest.raises(ValueError):
            check(binding, envelope(surface, [fact(surface, kind)], facts=facts))
    with pytest.raises(ValueError):
        check(
            binding,
            envelope(
                "You wrote " + quote + ". Supper at five pm.",
                [fact(quote, "faithful_quote")],
                facts=facts,
            ),
        )


def test_exact_book_excerpt_and_literal_not_paraphrase_certificate(binding):
    facts = [
        {
            "field": "message_text",
            "value": "I read Moon Book at four pm.",
            "source_record_ids": ["message-source"],
        }
    ]
    check(
        binding,
        envelope("How is Moon Book?", [fact("Moon Book", "excerpt")], facts=facts),
    )
    with pytest.raises(ValueError):
        check(
            binding,
            envelope(
                "You loved Moon Book",
                [fact("You loved Moon Book", "excerpt")],
                facts=facts,
            ),
        )
    assert "unmeasured" in str(RENDERING_POLICY).lower()


@pytest.mark.parametrize(
    "proposal",
    [
        "Could we move supper to four pm?",
        "Would four pm work instead for supper?",
        "Ask whether supper can move to four pm",
    ],
)
def test_proposal_anchor_local_scope_and_common_policy(binding, proposal):
    bundle, _ = world(proposal)
    anchors = precompute_anchors(bundle.request, binding=binding)
    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.kind == "proposal_time" and anchor.text == "four pm"
    mention = {
        "surface": "four pm",
        "rendering_kind": "proposal_clock",
        "anchor_id": anchor.anchor_id,
    }
    for _condition in ("baseline", "B", "A+B"):
        check(binding, envelope(proposal, [mention]), proposal, anchors=anchors)
        with pytest.raises(ValueError):
            check(
                binding,
                envelope("Supper now starts at four pm", [mention]),
                proposal,
                anchors=anchors,
            )
    check(
        binding,
        envelope("Supper starts at three pm. " + proposal, [fact("three pm"), mention]),
        proposal,
        anchors=anchors,
    )


@pytest.mark.parametrize(
    "clock_text,expected_fragments,anchor_count",
    [
        ("4 pm", ["Could we move supper to 4 pm", "Then ask Ren"], 1),
        ("4 p.m.", ["Could we move supper to 4 p.m. Then ask Ren"], 0),
    ],
)
def test_pm_period_may_keep_surrounding_text_in_one_clause(
    binding, clock_text, expected_fragments, anchor_count
):
    separator = " " if clock_text.endswith(".") else ". "
    request = f"Could we move supper to {clock_text}{separator}Then ask Ren."
    bundle, _ = world(request)
    local_now = datetime.fromisoformat(bundle.request.request_time)
    assert [
        fragment for _, _, fragment in clauses(request, local_now=local_now)
    ] == expected_fragments
    assert len(precompute_anchors(bundle.request, binding=binding)) == anchor_count


@pytest.mark.parametrize(
    "request_text",
    [
        "Could we move supper to four pm on Friday?",
        "Could we move supper to four pm on 2026-10-30?",
        "Would four pm on Friday work instead for supper?",
        "Could we move supper to four pm or five pm?",
    ],
)
def test_initial_proposal_recognition_requires_one_clock_and_no_dates(
    binding, request_text
):
    bundle, _ = world(request_text)
    assert precompute_anchors(bundle.request, binding=binding) == []


def test_request_quote_anchor_and_tamper_rejected(binding):
    request = 'Ask Ren about "supper at four pm"'
    bundle, _ = world(request)
    anchors = precompute_anchors(bundle.request, binding=binding)
    assert len(anchors) == 1 and anchors[0].kind == "quotation"
    quote = '"supper at four pm"'
    output = envelope(
        "What does " + quote + " mean?",
        [
            {
                "surface": quote,
                "rendering_kind": "faithful_quote",
                "anchor_id": anchors[0].anchor_id,
            }
        ],
    )
    check(binding, output, request, anchors=anchors)
    bad = anchors[0].model_copy(update={"text": "invented"})
    with pytest.raises(ValueError):
        check(binding, output, request, anchors=[bad])
    with pytest.raises(ValueError):
        check(binding, output, request + " changed", anchors=anchors[0:0])


@pytest.mark.parametrize(
    "text,quote",
    [
        ("He said 'don't be late'", "'don't be late'"),
        ("Ask Ren about 'it's ready'", "'it's ready'"),
        (
            "Ask Ren about 'Ren's supper isn't ready at four pm'",
            "'Ren's supper isn't ready at four pm'",
        ),
        ("Ask Ren about 'don't arrive at 4 p.m.'", "'don't arrive at 4 p.m.'"),
    ],
)
def test_straight_single_quote_keeps_internal_apostrophes(text, quote):
    start = text.index(quote)
    assert quotation_spans(text) == [(start, start + len(quote))]
    assert mask_quotations(text) == text[:start] + " " * len(quote)


def test_unquoted_possessive_and_contractions_are_not_quotes(binding):
    text = "Ask about Ren's supper; don't say it's ready"
    assert quotation_spans(text) == []
    assert mask_quotations(text) == text
    bundle, _ = world(text)
    assert precompute_anchors(bundle.request, binding=binding) == []


@pytest.mark.parametrize(
    "inner", ["don't be late", "it's ready", "Ren's supper isn't at four pm"]
)
def test_complete_request_quote_anchor_validates_without_recreating_anchors(
    binding, monkeypatch, inner
):
    quote = f"'{inner}'"
    request = "Ask Ren about " + quote
    bundle, candidates = world(request)
    anchors = precompute_anchors(bundle.request, binding=binding)
    assert len(anchors) == 1
    anchor = anchors[0]
    start = request.index(quote)
    assert (anchor.start, anchor.end, anchor.text, anchor.value) == (
        start,
        len(request),
        quote,
        inner,
    )
    assert (anchor.scope_start, anchor.scope_end) == (start, len(request))
    output = envelope(
        "What does " + quote + " mean?",
        [
            {
                "surface": quote,
                "rendering_kind": "faithful_quote",
                "anchor_id": anchor.anchor_id,
            }
        ],
        facts=[],
        event=None,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("validation must not recreate request anchors")

    monkeypatch.setattr("messaging_nlp.drafting.precompute_anchors", forbidden)
    before = output.model_dump()
    validate_envelope(
        output, bundle=bundle, candidates=candidates, anchors=anchors, binding=binding
    )
    assert output.model_dump() == before
    # A full contraction quote exempts only its own bytes, not later clocks.
    outside = output.model_copy(
        update={
            "decision": output.decision.model_copy(
                update={"draft": output.decision.draft + " Also five pm."}
            )
        }
    )
    with pytest.raises(ValueError, match="unbound temporal"):
        validate_envelope(
            outside,
            bundle=bundle,
            candidates=candidates,
            anchors=anchors,
            binding=binding,
        )


@pytest.mark.parametrize(
    "quote,fragment",
    [("'don't be late'", "'don'"), ("'it's ready'", "'it'")],
)
def test_forged_contraction_fragment_anchor_is_rejected(binding, quote, fragment):
    request = "Ask Ren about " + quote
    bundle, _ = world(request)
    anchors = precompute_anchors(bundle.request, binding=binding)
    assert len(anchors) == 1 and anchors[0].text == quote
    anchor = anchors[0]
    end = anchor.start + len(fragment)
    forged = anchor.model_copy(
        update={"end": end, "scope_end": end, "text": fragment, "value": fragment[1:-1]}
    )
    # The forged slice really occurs in the original request; it is NOT its quote.
    assert request[forged.start : forged.end] == fragment
    output = envelope(
        "What does " + fragment + " mean?",
        [
            {
                "surface": fragment,
                "rendering_kind": "faithful_quote",
                "anchor_id": anchor.anchor_id,
            }
        ],
        facts=[],
    )
    before = output.model_dump()
    with pytest.raises(ValueError, match="quotation anchor is not exact"):
        check(binding, output, request, anchors=[forged])
    with pytest.raises(ValueError, match="request quotation is not exact"):
        check(binding, output, request, anchors=anchors)
    assert output.model_dump() == before


@pytest.mark.parametrize(
    "inner", ["don't be late", "it's ready", "Ren's supper isn't at four pm"]
)
def test_complete_source_quote_with_apostrophes_and_altered_quote(binding, inner):
    source = "Ren wrote: " + inner + "."
    bundle, candidates = world()
    bundle.context.history[0] = bundle.context.history[0].model_copy(
        update={"text": source}
    )
    facts = [
        {
            "field": "message_text",
            "value": source,
            "source_record_ids": ["message-source"],
        }
    ]
    for quoted_inner, accepted in [(inner, True), (inner + " invented", False)]:
        quote = f"'{quoted_inner}'"
        output = envelope(
            "You wrote " + quote, [fact(quote, "faithful_quote")], facts=facts
        )
        before = output.model_dump()
        if accepted:
            validate_envelope(
                output,
                bundle=bundle,
                candidates=candidates,
                anchors=[],
                binding=binding,
            )
        else:
            with pytest.raises(ValueError, match="quotation differs"):
                validate_envelope(
                    output,
                    bundle=bundle,
                    candidates=candidates,
                    anchors=[],
                    binding=binding,
                )
        assert output.model_dump() == before


@pytest.mark.parametrize(
    "quote",
    [
        "'don't be late",
        "'it's ready”",
        "‘it's ready'",
        "”it's ready“",
        "’it's ready‘",
        "\"it's ready'",
        "'it's\nready'",
    ],
)
def test_malformed_or_mirrored_quotes_cannot_bind_request_or_source(binding, quote):
    request = "Ask Ren about " + quote
    assert quotation_spans(quote) != [(0, len(quote))]
    bundle, candidates = world(request)
    assert precompute_anchors(bundle.request, binding=binding) == []
    source = "it's ready; don't be late"
    bundle.context.history[0] = bundle.context.history[0].model_copy(
        update={"text": source}
    )
    facts = [
        {
            "field": "message_text",
            "value": source,
            "source_record_ids": ["message-source"],
        }
    ]
    output = envelope(quote, [fact(quote, "faithful_quote")], facts=facts)
    with pytest.raises(ValueError, match="complete paired quotation"):
        validate_envelope(
            output, bundle=bundle, candidates=candidates, anchors=[], binding=binding
        )


def test_multiple_complete_quotes_keep_offsets_around_unquoted_possessive():
    text = "Ren's note: 'don't be late'; then 'it's ready'."
    quotes = ["'don't be late'", "'it's ready'"]
    assert quotation_spans(text) == [
        (text.index(q), text.index(q) + len(q)) for q in quotes
    ]


def test_binding_mandatory_and_authenticated(binding):
    output = envelope("Hello", facts=[])
    with pytest.raises((ValueError, TypeError)):
        check(None, output)
    binding.tzif_path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        check(binding, output)


def test_selected_literals_and_candidates_checked(binding):
    output = envelope(
        "Hello",
        facts=[
            {
                "field": "start",
                "value": "2026-10-29T16:00:00Z",
                "source_record_ids": ["event-source"],
            }
        ],
    )
    with pytest.raises(ValueError):
        check(binding, output)
    bundle, candidates = world()
    candidates = candidates.model_copy(update={"source_record_ids": ["contact-source"]})
    with pytest.raises(ValueError):
        validate_envelope(
            envelope("Hello"),
            bundle=bundle,
            candidates=candidates,
            anchors=[],
            binding=binding,
        )


@pytest.mark.parametrize(
    "surface,instant",
    [
        ("half past two pm", "2026-10-29T14:30:00Z"),
        ("quarter to three pm", "2026-10-29T14:45:00Z"),
        ("quarter past three pm", "2026-10-29T15:15:00Z"),
        ("three thirty-five pm", "2026-10-29T15:35:00Z"),
        ("three thirty five in the afternoon", "2026-10-29T15:35:00Z"),
        ("three o’clock in the afternoon", "2026-10-29T15:00:00Z"),
        ("3 p.m.", "2026-10-29T15:00:00Z"),
        ("12 am", "2026-10-29T00:00:00Z"),
        ("noon", "2026-10-29T12:00:00Z"),
        ("midnight", "2026-10-29T00:00:00Z"),
    ],
)
def test_minute_fraction_and_day_period_positive(binding, surface, instant):
    bundle, candidates = world()
    bundle.context.calendar[0] = bundle.context.calendar[0].model_copy(
        update={"start": instant}
    )
    output = envelope(
        surface,
        [fact(surface)],
        facts=[
            {
                "field": "start",
                "value": instant,
                "source_record_ids": ["event-source"],
            }
        ],
    )
    validate_envelope(
        output, bundle=bundle, candidates=candidates, anchors=[], binding=binding
    )


@pytest.mark.parametrize(
    "extra",
    [
        "three sixty pm",
        "half to four pm",
        "October 29st",
        "Oct 29",
        "2026-10-29T15:00:00Z",
        "3:00:15 pm",
        "3:0 pm",
        "02/30/2026",
        "day after tomorrow",
        "tonight",
    ],
)
def test_recognized_unsupported_or_malformed_forms_cannot_hide_in_excerpt(
    binding, extra
):
    bundle, candidates = world()
    bundle.context.history[0] = bundle.context.history[0].model_copy(
        update={"text": extra}
    )
    output = envelope(
        extra,
        [fact(extra, "literal")],
        facts=[
            {
                "field": "message_text",
                "value": extra,
                "source_record_ids": ["message-source"],
            }
        ],
    )
    with pytest.raises(ValueError):
        validate_envelope(
            output, bundle=bundle, candidates=candidates, anchors=[], binding=binding
        )


@pytest.mark.parametrize(
    "quote",
    [
        '"Moon Book at four pm"',
        "'Moon Book at four pm'",
        "“Moon Book at four pm”",
        "‘Moon Book at four pm’",
    ],
)
def test_quotation_delimiters_and_provenance(binding, quote):
    source_fact = {
        "field": "message_text",
        "value": "I read Moon Book at four pm.",
        "source_record_ids": ["message-source"],
    }
    check(
        binding, envelope(quote, [fact(quote, "faithful_quote")], facts=[source_fact])
    )


def test_current_start_cannot_be_satisfied_by_faithful_quote(binding):
    facts = [
        {
            "field": "message_text",
            "value": "I read Moon Book at four pm.",
            "source_record_ids": ["message-source"],
        }
    ]
    quote = '"four pm"'
    with pytest.raises(ValueError):
        check(
            binding,
            envelope(quote, [fact(quote, "faithful_quote")], facts=facts),
            "Confirm the scheduled start of supper",
        )


def test_demanding_multifact_current_plus_proposal(binding):
    request_text = "Confirm supper starts at three pm. Could we move supper to four pm?"
    bundle, candidates = world(request_text)
    event = bundle.context.calendar[0]
    bundle.context.calendar[0] = event.model_copy(
        update={
            "end": "2026-10-29T17:00:00Z",
            "location": "Moon Hall",
        }
    )
    anchors = precompute_anchors(bundle.request, binding=binding)
    facts = [
        {"field": "name", "value": "Ren", "source_record_ids": ["contact-source"]},
        {"field": "title", "value": "Supper", "source_record_ids": ["event-source"]},
        {"field": "start", "value": event.start, "source_record_ids": ["event-source"]},
        {
            "field": "end",
            "value": "2026-10-29T17:00:00Z",
            "source_record_ids": ["event-source"],
        },
        {
            "field": "location",
            "value": "Moon Hall",
            "source_record_ids": ["event-source"],
        },
        {
            "field": "message_text",
            "value": "I read Moon Book at four pm.",
            "source_record_ids": ["message-source"],
        },
    ]
    draft = (
        "Ren, Supper starts October 29, 2026 at three pm and ends at five pm "
        "in Moon Hall. Could we move supper to 4 pm? How is Moon Book?"
    )
    output = envelope(
        draft,
        [
            fact("Ren", "literal", 0),
            fact("Supper", "literal", 1),
            fact("October 29, 2026 at three pm", "datetime", 2),
            fact("five pm", "clock", 3),
            fact("Moon Hall", "literal", 4),
            fact("Moon Book", "excerpt", 5),
            {
                "surface": "4 pm",
                "rendering_kind": "proposal_clock",
                "anchor_id": anchors[0].anchor_id,
            },
        ],
        facts=facts,
    )
    validate_envelope(
        output, bundle=bundle, candidates=candidates, anchors=anchors, binding=binding
    )
    # This is a scripted format fixture, not an assertion about 256-token Qwen fit.


@pytest.mark.parametrize(
    "draft",
    [
        "Could we move supper to four pm? Supper now starts at four pm.",
        "Could we move supper to four pm, and supper now starts then",
        "Would four pm work instead for supper? Supper starts at five pm.",
    ],
)
def test_proposal_does_not_certify_other_clauses(binding, draft):
    request_text = "Could we move supper to four pm?"
    bundle, _ = world(request_text)
    anchor = precompute_anchors(bundle.request, binding=binding)[0]
    output = envelope(
        draft,
        [
            {
                "surface": "four pm",
                "rendering_kind": "proposal_clock",
                "anchor_id": anchor.anchor_id,
            }
        ],
    )
    with pytest.raises(ValueError):
        check(binding, output, request_text)


def test_output_cannot_create_or_retype_request_anchor(binding):
    output = envelope(
        "Could we move supper to four pm?",
        [
            {
                "surface": "four pm",
                "rendering_kind": "proposal_clock",
                "anchor_id": "invented",
            }
        ],
    )
    with pytest.raises(ValueError):
        check(binding, output)
    proposal = "Could we move supper to four pm?"
    bundle, _ = world(proposal)
    anchor = precompute_anchors(bundle.request, binding=binding)[0]
    for kind in ("clock", "literal", "faithful_quote"):
        with pytest.raises(ValueError):
            check(
                binding,
                envelope(
                    "four pm",
                    [
                        {
                            "surface": "four pm",
                            "rendering_kind": kind,
                            "anchor_id": anchor.anchor_id,
                        }
                    ],
                ),
                proposal,
            )


def test_temporal_literals_and_excerpts_never_exempt_scan(binding):
    for kind in ("literal", "excerpt"):
        source = "I read Moon Book at four pm."
        facts = [
            {
                "field": "message_text",
                "value": source,
                "source_record_ids": ["message-source"],
            }
        ]
        with pytest.raises(ValueError):
            check(binding, envelope(source, [fact(source, kind)], facts=facts))


def test_no_event_required_when_requested_draft_does_not_need_one(binding):
    check(
        binding,
        envelope("How is Moon Book?", facts=[], event=None),
        "Ask Ren about Moon Book",
    )


def test_nested_datetime_bindings_rejected(binding):
    output = envelope(
        "2026-10-29 at three pm",
        [
            fact("2026-10-29 at three pm", "datetime"),
            fact("three pm"),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        check(binding, output)


def test_timezone_conversion_uses_authenticated_bytes_not_host(binding, tmp_path):
    data = (
        b"TZif\x00"
        + b"\x00" * 15
        + struct.pack(">6l", 0, 0, 0, 0, 1, 4)
        + struct.pack(">lbb", 7200, 0, 0)
        + b"FXT\x00"
    )
    path = tmp_path / "fictional-offset.tzif"
    path.write_bytes(data)
    trusted = TimezoneBinding("Etc/GMT-2", path, hashlib.sha256(data).hexdigest())
    bundle, candidates = world("Could we move supper to midnight?")
    bundle = bundle.model_copy(
        update={
            "request": bundle.request.model_copy(
                update={
                    "time_zone": "Etc/GMT-2",
                    "request_time": "2026-10-28T23:30:00Z",
                }
            )
        }
    )
    anchors = precompute_anchors(bundle.request, binding=trusted)
    output = envelope("Today at five pm", [fact("Today at five pm", "datetime")])
    validate_envelope(
        output, bundle=bundle, candidates=candidates, anchors=anchors, binding=trusted
    )
    with pytest.raises(ValueError):
        validate_envelope(
            envelope("three pm", [fact("three pm")]),
            bundle=bundle,
            candidates=candidates,
            anchors=anchors,
            binding=trusted,
        )
    with pytest.raises(ValueError):
        validate_envelope(
            output,
            bundle=bundle,
            candidates=candidates,
            anchors=anchors,
            binding=binding,
        )


@pytest.mark.parametrize(
    "text",
    [
        "Ask Ren about the word starts",
        "Tell Ren about the word starts",
        "Ask Ren when supper starts",
        "Ask Ren where supper is",
    ],
)
def test_nonconfirmation_clauses_do_not_require_start(binding, text):
    check(binding, envelope("Could you explain?", facts=[]), text)


@pytest.mark.parametrize(
    "day,surface",
    [
        ("2026-10-27", "today"),
        ("2026-10-28", "tomorrow"),
        ("2026-10-26", "yesterday"),
        ("2026-10-27", "Tuesday"),
        ("2026-11-03", "next Tuesday"),
        ("2026-10-29", "October 29th"),
    ],
)
def test_relative_calendar_and_ordinal_positive(binding, day, surface):
    bundle, candidates = world()
    instant = day + "T15:00:30Z"
    bundle.context.calendar[0] = bundle.context.calendar[0].model_copy(
        update={"start": instant}
    )
    output = envelope(
        surface,
        [fact(surface, "date")],
        facts=[
            {
                "field": "start",
                "value": instant,
                "source_record_ids": ["event-source"],
            }
        ],
    )
    validate_envelope(
        output, bundle=bundle, candidates=candidates, anchors=[], binding=binding
    )


def test_request_quoted_conflict_does_not_override_unquoted_current(binding):
    request_text = "Confirm supper starts at three pm. Discuss 'four pm'."
    check(
        binding, envelope("Supper starts at three pm", [fact("three pm")]), request_text
    )


def test_start_end_fact_not_interchangeable(binding):
    bundle, candidates = world("Confirm supper starts at three pm")
    bundle.context.calendar[0] = bundle.context.calendar[0].model_copy(
        update={
            "end": "2026-10-29T17:00:00Z",
        }
    )
    output = envelope(
        "Supper ends at five pm",
        [fact("five pm")],
        facts=[
            {
                "field": "end",
                "value": "2026-10-29T17:00:00Z",
                "source_record_ids": ["event-source"],
            }
        ],
    )
    with pytest.raises(ValueError, match="start"):
        validate_envelope(
            output, bundle=bundle, candidates=candidates, anchors=[], binding=binding
        )


def test_cancelled_event_cannot_confirm_current_start(binding):
    bundle, candidates = world("Confirm supper starts at three pm")
    bundle.context.calendar[0] = bundle.context.calendar[0].model_copy(
        update={"status": "cancelled"}
    )
    with pytest.raises(ValueError, match="scheduled"):
        validate_envelope(
            envelope("Supper starts at three pm", [fact("three pm")]),
            bundle=bundle,
            candidates=candidates,
            anchors=[],
            binding=binding,
        )


def test_missing_timezone_artifact_fails_closed(binding):
    binding.tzif_path.unlink()
    with pytest.raises(ValueError, match="artifact"):
        check(binding, envelope("Hello", facts=[]))


def test_bad_fact_source_fails_valueerror_without_repair(binding):
    output = envelope(
        "Hello",
        facts=[
            {
                "field": "name",
                "value": "Ren",
                "source_record_ids": ["event-source"],
            }
        ],
    )
    with pytest.raises(ValueError):
        check(binding, output)


@pytest.mark.parametrize(
    "draft,surface,kind",
    [
        ("quarter after three pm", "three pm", "clock"),
        ("ten to three pm", "three pm", "clock"),
        ("half three pm", "three pm", "clock"),
        ("last Thursday", "Thursday", "date"),
        ("three pm UTC", "three pm", "clock"),
    ],
)
def test_unsupported_modifier_cannot_hide_behind_valid_suffix(
    binding, draft, surface, kind
):
    with pytest.raises(ValueError):
        check(binding, envelope(draft, [fact(surface, kind)]))


@pytest.mark.parametrize("clock_text", ["25:00", "four", "three sixty pm"])
def test_malformed_explicit_current_request_requires_not_asserting(binding, clock_text):
    with pytest.raises(ValueError):
        check(
            binding,
            envelope("Supper starts at three pm", [fact("three pm")]),
            "Confirm supper starts at " + clock_text,
        )


def test_cancelled_event_remains_eligible_for_cancellation_question(binding):
    bundle, candidates = world("Ask Ren whether supper is cancelled")
    bundle.context.calendar[0] = bundle.context.calendar[0].model_copy(
        update={"status": "cancelled"}
    )
    output = envelope("Is supper cancelled?", facts=[])
    validate_envelope(
        output, bundle=bundle, candidates=candidates, anchors=[], binding=binding
    )


@pytest.mark.parametrize(
    "request_text",
    [
        "Could we move supper to four pm?",
        "Would four pm work instead for supper?",
    ],
)
def test_local_pronoun_proposal_uses_precomputed_request_anchor(binding, request_text):
    bundle, _ = world(request_text)
    anchor = precompute_anchors(bundle.request, binding=binding)[0]
    check(
        binding,
        envelope(
            "Could we move it to four pm?",
            [
                {
                    "surface": "four pm",
                    "rendering_kind": "proposal_clock",
                    "anchor_id": anchor.anchor_id,
                }
            ],
        ),
        request_text,
    )


def test_integration_compound_date_form(binding):
    bundle, candidates = world()
    instant = "2026-10-27T15:00:00Z"
    bundle.context.calendar[0] = bundle.context.calendar[0].model_copy(
        update={"start": instant}
    )
    surface = "October 27, 2026 at 3 pm"
    output = envelope(
        surface,
        [fact(surface, "datetime")],
        facts=[
            {
                "field": "start",
                "value": instant,
                "source_record_ids": ["event-source"],
            }
        ],
    )
    validate_envelope(
        output, bundle=bundle, candidates=candidates, anchors=[], binding=binding
    )
