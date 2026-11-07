"""Independent fictional retrieval expectations (behavioral specification 1–8)."""

import hashlib
from pathlib import Path
from zoneinfo import TZPATH

import pytest

from messaging_nlp.context import (
    CalendarEvent,
    Contact,
    HistoryMessage,
    PublicBundle,
    PublicContext,
)
from messaging_nlp.contracts import Entity, Extraction, Request
from messaging_nlp.retrieval import search
from messaging_nlp.temporal import TimezoneBinding


def contact(cid, name, aliases=()):
    return Contact(
        record_id=f"contact-{cid}",
        contact_id=cid,
        version=1,
        name=name,
        aliases=list(aliases),
        relationship="friend",
        available_at="2026-01-01T00:00Z",
        provenance="fictional",
    )


def event(eid, title, start="2026-10-30T15:00Z", participants=(), **extra):
    return CalendarEvent(
        record_id=f"event-{eid}",
        event_id=eid,
        version=1,
        title=title,
        participant_contact_ids=list(participants),
        start=start,
        status="scheduled",
        available_at="2026-01-01T00:00Z",
        provenance="fictional",
        **extra,
    )


def message(rid, sender, recipient, text):
    return HistoryMessage(
        record_id=rid,
        sender_contact_id=sender,
        recipient_contact_id=recipient,
        text=text,
        timestamp="2026-10-01T10:00Z",
        available_at="2026-10-01T10:01Z",
        provenance="fictional",
    )


def bundle(
    text, contacts=(), calendar=(), history=(), *, now="2026-10-27T12:00Z", zone="UTC"
):
    return PublicBundle(
        request=Request(
            message_id="request",
            text=text,
            owner_id="owner",
            request_time=now,
            time_zone=zone,
            context_snapshot_id="snapshot",
        ),
        context=PublicContext(
            contacts=list(contacts), calendar=list(calendar), history=list(history)
        ),
    )


def binding(zone="UTC"):
    # Host bytes are a fictional software fixture, not the study's retained binding.
    path = next(Path(root) / zone for root in TZPATH if (Path(root) / zone).is_file())
    return TimezoneBinding(zone, path, hashlib.sha256(path.read_bytes()).hexdigest())


def extraction(text, *pairs):
    entities = [
        Entity(
            type=kind,
            text=surface,
            start=text.index(surface),
            end=text.index(surface) + len(surface),
        )
        for kind, surface in pairs
    ]
    return Extraction(entities=sorted(entities, key=lambda e: e.start)).validate_text(
        text
    )


def run(world, hints=None, **kwargs):
    return search(world, hints, binding=binding(world.request.time_zone), **kwargs)


def test_duplicate_names_and_authored_invitation_both_survive():
    world = bundle(
        "Ask Ren about supper",
        [contact("r1", "Ren"), contact("r2", "Ren")],
        history=[message("invitation", "r2", "owner", "Join me for supper")],
    )
    result = run(world)
    assert set(result.candidates.contact_ids) == {"r1", "r2"}
    assert result.evidence.history[0].sender_contact_id == "r2"
    assert "invitation" in result.candidates.source_record_ids


def test_outgoing_owner_third_party_and_null_endpoints_unchanged():
    world = bundle(
        "Ask Lina about Jules and supper",
        [contact("l", "Lina"), contact("j", "Jules")],
        history=[
            message("out", "owner", "l", "Jules will attend supper"),
            message("unknown", None, None, "supper details"),
        ],
    )
    result = run(world, extraction(world.request.text, ("PERSON", "Lina")))
    assert result.evidence.history == world.context.history
    assert result.evidence.history[0].sender_contact_id == "owner"
    assert result.evidence.history[0].recipient_contact_id == "l"
    assert result.evidence.history[1].sender_contact_id is None
    assert result.evidence.history[1].recipient_contact_id is None


def test_partial_title_and_title_only_candidates_survive():
    world = bundle(
        "Ask about lantern supper",
        calendar=[
            event("weak", "Lantern maintenance"),
            event("strong", "Annual lantern supper by the lake"),
        ],
    )
    result = run(world)
    assert result.candidates.event_ids[0] == "strong"
    assert set(result.candidates.event_ids) == {"weak", "strong"}


def test_typed_group_changes_actual_order_with_exact_noisy_spans():
    # Independent expectation: grouped EVENT outranks a bag-of-words decoy;
    # baseline gives the decoy more request-token overlap. Names aren't roles.
    text = "uh Lina ask lantern supper after repair meeting pls"
    world = bundle(
        text,
        [contact("l", "Lina")],
        calendar=[
            event("decoy", "Repair meeting ask after"),
            event("target", "Annual lantern supper"),
        ],
        history=[message("exchange", "l", "owner", "The invitation is attached")],
    )
    hints = extraction(text, ("PERSON", "Lina"), ("EVENT", "lantern supper"))
    plain, assisted = run(world), run(world, hints)
    assert plain.candidates.event_ids[0] == "decoy"
    assert assisted.candidates.event_ids[0] == "target"
    assert "exchange" in assisted.candidates.source_record_ids
    assert any(
        q.channel == "event" and q.text == "lantern supper" for q in assisted.queries
    )
    assert any(q.channel == "original" and q.text == text for q in assisted.queries)
    assert run(world, max_records=3).candidates.event_ids == ["decoy"]
    assert run(world, hints, max_records=3).candidates.event_ids == ["target"]


def test_associated_exchange_not_just_contact_text_mention():
    text = "Tell Lina about supper"
    world = bundle(
        text,
        [contact("l", "Lina"), contact("j", "Jules")],
        history=[
            message("in", "l", "owner", "Invitation attached"),
            message("out", "owner", "l", "Thanks for that"),
            message("third", "j", "owner", "Lina has details"),
        ],
    )
    result = run(world, extraction(text, ("PERSON", "Lina"), ("EVENT", "supper")))
    assert {"in", "out"} <= set(result.candidates.source_record_ids)
    assert all(
        m.sender_contact_id
        == world.context.history[
            [r.record_id for r in world.context.history].index(m.record_id)
        ].sender_contact_id
        for m in result.evidence.history
    )
    associated = [q for q in result.queries if q.channel == "associated"]
    assert associated and associated[0].contact_ids == ["l"]


def test_cancelled_latest_record_retained_and_no_external_lookup():
    old = event("s", "Supper")
    cancelled = old.model_copy(
        update={"record_id": "cancelled-v2", "version": 2, "status": "cancelled"}
    )
    result = run(bundle("Supper", calendar=[cancelled]))
    assert result.evidence.calendar == [cancelled]
    assert result.candidates.source_record_ids == ["cancelled-v2"]
    # A future/unlisted update cannot be found by this API; eligible v1 stays v1.
    assert run(bundle("Supper", calendar=[old])).evidence.calendar == [old]


def test_later_week_strict_now_exclusive_sunday_and_dst():
    world = bundle(
        "Supper later this week",
        now="2026-03-08T05:30Z",
        zone="America/New_York",
        calendar=[
            event("now", "Supper", "2026-03-08T05:30Z"),
            event("inside", "Supper", "2026-03-15T03:59Z"),
            event("end", "Supper", "2026-03-15T04:00Z"),
            event("elapsed", "Supper", "2026-03-15T04:30Z"),
        ],
    )
    assert run(world).candidates.event_ids == ["inside"]


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("today", "today"),
        ("tomorrow", "tomorrow"),
        ("Friday", "friday"),
        ("2026-10-30", "friday"),
    ],
)
def test_supported_dates_local_not_utc(phrase, expected):
    world = bundle(
        f"Supper {phrase}",
        now="2026-10-28T01:00Z",
        zone="America/New_York",
        calendar=[
            event("today", "Supper", "2026-10-28T01:30Z"),
            event("tomorrow", "Supper", "2026-10-28T16:00Z"),
            event("friday", "Supper", "2026-10-30T16:00Z"),
        ],
    )
    assert run(world).candidates.event_ids == [expected]


def test_contact_endpoint_and_participant_source_closure():
    world = bundle(
        "supper",
        [contact("l", "Lina"), contact("j", "Jules")],
        [event("s", "Supper", participants=["j"])],
        [message("m", "owner", "l", "supper")],
    )
    result = run(world)
    assert set(result.candidates.contact_ids) == {"l", "j"}
    assert {"contact-l", "contact-j"} <= set(result.candidates.source_record_ids)
    assert set(result.candidates.source_record_ids) == {
        r.record_id
        for r in [
            *result.evidence.contacts,
            *result.evidence.calendar,
            *result.evidence.history,
        ]
    }
    world.validate_candidates(result.candidates)


def test_bounded_pack_does_not_silently_choose_one_duplicate():
    world = bundle(
        "Ask Ren about supper",
        [contact("r1", "Ren"), contact("r2", "Ren")],
        history=[message("m", "r2", "owner", "supper")],
    )
    result = run(world, max_records=1)
    assert result.candidates.contact_ids == []
    assert result.evidence.history == []
    assert result.coverage.partial
    assert result.coverage.packed_records <= 1
    assert any("ambig" in note.lower() for note in result.coverage.notes)


def test_bounded_pack_never_drops_endpoint_source_to_fit():
    world = bundle(
        "supper", [contact("l", "Lina")], history=[message("m", "l", "owner", "supper")]
    )
    result = run(world, max_records=1)
    assert not result.evidence.history
    assert result.coverage.partial


def test_empty_extraction_same_as_baseline_and_deterministic_no_mutation():
    world = bundle("Supper", calendar=[event("s", "Supper")])
    before = world.model_dump_json()
    assert run(world) == run(world, Extraction(entities=[]))
    assert world.model_dump_json() == before


def test_invalid_exact_span_and_timezone_binding_fail_closed():
    world = bundle("Supper")
    bad = Extraction(entities=[Entity(type="EVENT", text="upper", start=0, end=5)])
    with pytest.raises(ValueError, match="slice"):
        run(world, bad)
    with pytest.raises(ValueError, match="zone"):
        search(world, binding=binding("America/New_York"))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_record_limit(limit):
    with pytest.raises((ValueError, TypeError)):
        run(bundle("Supper"), max_records=limit)


def test_unsupported_typed_date_records_limited_coverage():
    text = "Supper sometime around harvest"
    world = bundle(text, calendar=[event("s", "Supper")])
    result = run(world, extraction(text, ("DATE", "sometime around harvest")))
    assert result.candidates.event_ids == ["s"]
    assert result.coverage.partial
    assert any("unsupported DATE" in note for note in result.coverage.notes)


def test_alias_search_uses_whole_words_and_not_substring_identity():
    world = bundle(
        "Tell Bee about supper",
        [
            contact("b", "Beatrice", ["Bee"]),
            contact("x", "Beeline"),
        ],
        history=[message("associated", "b", "owner", "Here are the details")],
    )
    result = run(world)
    assert result.candidates.contact_ids == ["b"]
    assert result.candidates.source_record_ids == ["contact-b", "associated"]


def test_topic_group_partial_title_without_named_participants():
    text = "ask Ren about silver river reading and shipping project"
    world = bundle(
        text,
        [contact("r", "Ren")],
        calendar=[
            event("decoy", "Shipping project and ask"),
            event("target", "The silver river reading circle"),
        ],
    )
    hints = extraction(text, ("PERSON", "Ren"), ("TOPIC", "silver river reading"))
    assert run(world, hints).candidates.event_ids[0] == "target"


def test_later_week_remains_strict_with_multiple_date_phrases():
    world = bundle(
        "Supper today or later this week",
        calendar=[
            event("past", "Supper", "2026-10-27T11:00Z"),
            event("future", "Supper", "2026-10-27T13:00Z"),
        ],
    )
    assert run(world).candidates.event_ids == ["future"]


def test_packed_ambiguity_remains_visible_with_authored_join():
    world = bundle(
        "Ask Ren about supper",
        [contact("r1", "Ren"), contact("r2", "Ren")],
        history=[message("invite", "r2", "owner", "Supper invitation")],
    )
    result = run(world, max_records=3)
    assert set(result.candidates.contact_ids) == {"r1", "r2"}
    assert [m.record_id for m in result.evidence.history] == ["invite"]
    assert not result.coverage.partial
    assert result.coverage.matched_records == result.coverage.packed_records == 3


def test_next_weekday_policy_matches_rendering_calendar_arithmetic():
    # Tuesday request: Friday/this Friday mean Oct30; next Friday means Nov6.
    world = bundle(
        "Supper next Friday",
        calendar=[
            event("this", "Supper", "2026-10-30T15:00Z"),
            event("next", "Supper", "2026-11-06T15:00Z"),
        ],
    )
    assert run(world).candidates.event_ids == ["next"]


@pytest.mark.parametrize("sender,recipient", [("l", "owner"), ("owner", "l")])
def test_person_authored_invitation_discovers_title_and_preserves_source_map(
    sender, recipient
):
    text = "Ask Lina to confirm the invitation"
    world = bundle(
        text,
        [contact("l", "Lina"), contact("j", "Jules")],
        calendar=[
            event("supper", "Annual Copper Lantern supper"),
            event("unrelated", "Indigo Orchard picnic"),
        ],
        history=[
            message("invitation", sender, recipient, "Join the Copper Lantern supper"),
            message("third-party", "j", "owner", "Lina mentions Indigo Orchard picnic"),
        ],
    )
    result = run(world, extraction(text, ("PERSON", "Lina")))
    assert result.candidates.event_ids == ["supper"]
    assert {"event-supper", "invitation", "contact-l"} <= set(
        result.candidates.source_record_ids
    )
    assert world.context.history[0] in result.evidence.history
    assert any(
        q.text == "title bridge event-supper <- invitation" and q.contact_ids == ["l"]
        for q in result.queries
    )


@pytest.mark.parametrize(
    "sender,recipient", [("l", None), (None, "l"), (None, None), ("j", "owner")]
)
def test_null_or_third_party_text_does_not_bridge_person_to_title(sender, recipient):
    text = "Ask Lina to confirm the invitation"
    world = bundle(
        text,
        [contact("l", "Lina"), contact("j", "Jules")],
        [event("s", "Annual Copper Lantern supper")],
        [
            message(
                "not-associated",
                sender,
                recipient,
                "Lina mentions Copper Lantern supper",
            )
        ],
    )
    result = run(world, extraction(text, ("PERSON", "Lina")))
    assert result.candidates.event_ids == []
    assert not any(q.text.startswith("title bridge") for q in result.queries)


def test_single_shared_title_token_never_bridges_calendar():
    text = "Ask Lina to confirm the invitation"
    world = bundle(
        text,
        [contact("l", "Lina")],
        [event("s", "Annual Copper Lantern supper")],
        [message("m", "l", "owner", "The Lantern is broken")],
    )
    assert run(world, extraction(text, ("PERSON", "Lina"))).candidates.event_ids == []


def test_grouped_topic_narrows_bridge_but_retains_original_channel():
    text = "Ask Lina about Copper Lantern and repair"
    world = bundle(
        text,
        [contact("l", "Lina")],
        calendar=[
            event("target", "Annual Copper Lantern supper"),
            event("other", "Indigo Orchard picnic"),
            event("original", "Repair"),
        ],
        history=[
            message(
                "m",
                "owner",
                "l",
                "Copper Lantern supper and Indigo Orchard picnic invitations",
            )
        ],
    )
    person = extraction(text, ("PERSON", "Lina"))
    narrowed = extraction(text, ("PERSON", "Lina"), ("TOPIC", "Copper Lantern"))
    assert "other" in run(world, person).candidates.event_ids
    result = run(world, narrowed)
    assert set(result.candidates.event_ids) == {"target", "original"}
    assert any(q.text == "title bridge event-target <- m" for q in result.queries)


def test_bridge_budget_closure_includes_all_supporting_messages_and_ambiguous_seeds():
    text = "Ask Lina to confirm the invitation"
    world = bundle(
        text,
        [contact("l1", "Lina"), contact("l2", "Lina")],
        [event("s", "Annual Copper Lantern supper")],
        [
            message("in", "l1", "owner", "Copper Lantern supper"),
            message("out", "owner", "l2", "Copper Lantern supper details"),
        ],
    )
    hints = extraction(text, ("PERSON", "Lina"))
    full = run(world, hints, max_records=5)
    assert full.candidates.event_ids == ["s"]
    assert set(full.candidates.source_record_ids) == {
        "contact-l1",
        "contact-l2",
        "in",
        "out",
        "event-s",
    }
    limited = run(world, hints, max_records=4)
    assert limited.candidates.event_ids == []
    assert set(limited.candidates.contact_ids) == {"l1", "l2"}
    assert limited.coverage.partial
    assert limited.coverage.matched_records == 5


def test_date_narrows_bridged_events_not_supporting_authored_exchange():
    text = "Ask Lina to confirm the invitation tomorrow"
    world = bundle(
        text,
        [contact("l", "Lina")],
        calendar=[
            event("today", "Copper Lantern supper", "2026-10-27T15:00Z"),
            event("tomorrow", "Copper Lantern supper", "2026-10-28T15:00Z"),
        ],
        history=[message("m", "l", "owner", "Copper Lantern supper invitation")],
    )
    result = run(world, extraction(text, ("PERSON", "Lina"), ("DATE", "tomorrow")))
    assert result.candidates.event_ids == ["tomorrow"]
    assert "m" in result.candidates.source_record_ids
