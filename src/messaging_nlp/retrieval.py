"""Bounded lexical/grouped search over an already eligible public bundle.

PERSON seeds identify contacts and authored exchanges, never recipients. Typed
EVENT/TOPIC token groups add a ranking bonus when all meaningful tokens occur
in a title/body (partial titles are allowed). Original-text overlap remains a
separate channel. This is deliberately not semantic retrieval or an intent model.
Queries and coverage are private diagnostics, not prompt material.
"""

from __future__ import annotations

import re
from datetime import timedelta

from messaging_nlp.application_types import Coverage, SearchQuery, SearchResult
from messaging_nlp.context import (
    CalendarEvent,
    Contact,
    HistoryMessage,
    PublicBundle,
    PublicContext,
)
from messaging_nlp.contracts import CandidateEvidence, Extraction, clock
from messaging_nlp.temporal import TimezoneBinding, local_week, request_zone

_STOP = set("a an the about to me and of for on in at is it this with please".split())
_WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
_DATE = re.compile(
    r"\blater this week\b|\bthis week\b|\btoday\b|\btomorrow\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b(?:(?:next|this)\s+)?(?:" + "|".join(_WEEKDAYS) + r")\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


def _tokens(text: str) -> set[str]:
    return set(_words(text)) - _STOP


def _phrase(phrase: str, text: str) -> bool:
    # Unicode word sequences, no fuzzy matching or inferred spelling repair.
    needle = _words(phrase)
    haystack = _words(text)
    return bool(needle) and any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def search(
    bundle: PublicBundle,
    extraction: Extraction | None = None,
    *,
    binding: TimezoneBinding,
    max_records: int = 40,
) -> SearchResult:
    """Search only supplied records; caller owns dataset/snapshot eligibility.

    Dates support today/tomorrow, ISO dates, [this/next] weekdays and the local
    Sunday week. Multiple date phrases form a union, intersected with the strict
    (now, next local Sunday) window whenever "later this week" occurs. Thus another
    date cannot re-admit now, earlier times or the next Sunday boundary.
    Unsupported typed DATEs leave lexical candidates intact and report partial
    deterministic coverage.
    Dates narrow calendar only, never erase associated exchange evidence.
    Bare/this weekday means its next occurrence including today; next weekday
    means that occurrence plus seven days, matching the rendering grammar.

    A seed-to-owner authored exchange bridges to a calendar title only with at
    least two distinct meaningful title tokens in its body. If EVENT/TOPIC groups
    exist, at least one entire group must occur in both body and title. Every
    supporting exchange and its endpoint/ambiguity joins is packed atomically
    with the bridged calendar record, even if it also has original-text overlap.
    Null endpoints and text-only person mentions do not establish association.

    Each packed record includes all known endpoint/participant contact sources.
    Ambiguous matched-name groups are atomic: if they cannot fit, no member is
    silently retained as the unique answer. max_records counts joined sources.
    """
    if type(max_records) is not int or max_records < 1:
        raise ValueError("max_records must be a positive integer")
    request, context = bundle.request, bundle.context
    zone = request_zone(request, binding=binding)
    now = clock(request.request_time)
    local_day = now.astimezone(zone).date()
    if extraction is not None:
        extraction.validate_text(request.text)
    entities = extraction.entities if extraction is not None else []
    queries = [SearchQuery(channel="original", text=request.text)]
    notes = [
        "Bounded lexical overlap and exact grouped tokens; not exhaustive semantic coverage.",
        "PERSON/contact candidates are not recipient selections.",
    ]
    partial = False
    contacts = {c.contact_id: c for c in context.contacts}
    rows = [*context.contacts, *context.calendar, *context.history]
    records = {r.record_id: r for r in rows}

    # Name/alias ambiguity groups are preserved even when one contact has much
    # stronger authored evidence. A mention in message text is not authorship.
    seeds: set[str] = set()
    ambiguity: list[set[str]] = []
    names: dict[tuple[str, ...], set[str]] = {}
    for c in context.contacts:
        for name in [c.name, *c.aliases]:
            if _phrase(name, request.text):
                names.setdefault(tuple(_words(name)), set()).add(c.contact_id)
    for ids in names.values():
        seeds.update(ids)
        if len(ids) > 1:
            ambiguity.append(ids)
    typed_seeds: set[str] = set()
    groups: list[set[str]] = []
    for entity in entities:
        if entity.type == "PERSON":
            ids = {
                c.contact_id
                for c in context.contacts
                if any(_words(entity.text) == _words(n) for n in [c.name, *c.aliases])
            }
            typed_seeds.update(ids)
            seeds.update(ids)
            if len(ids) > 1:
                ambiguity.append(ids)
            queries.append(
                SearchQuery(channel="person", text=entity.text, contact_ids=sorted(ids))
            )
        elif entity.type in ("EVENT", "TOPIC"):
            tokens = _tokens(entity.text)
            if tokens:
                groups.append(tokens)
            queries.append(SearchQuery(channel=entity.type.lower(), text=entity.text))
    if seeds:
        queries.append(
            SearchQuery(
                channel="associated",
                text="authored endpoint exchanges",
                contact_ids=sorted(seeds),
            )
        )

    date_phrases = [m.group() for m in _DATE.finditer(request.text)]
    date_phrases.extend(e.text for e in entities if e.type == "DATE")
    date_days = set()
    week = False
    later = False
    for phrase in dict.fromkeys(date_phrases):
        value = " ".join(phrase.casefold().split())
        queries.append(SearchQuery(channel="date", text=phrase))
        if value in ("later this week", "this week"):
            week = True
            later = later or value == "later this week"
        elif value in ("today", "tomorrow"):
            date_days.add(local_day + timedelta(days=value == "tomorrow"))
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            try:
                date_days.add(local_day.fromisoformat(value))
            except ValueError:
                notes.append(f"unsupported DATE: {phrase}")
                partial = True
        elif (
            value.split()[-1:]
            and value.split()[-1] in _WEEKDAYS
            and (len(value.split()) == 1 or value.split()[0] in ("this", "next"))
            and len(value.split()) <= 2
        ):
            day = _WEEKDAYS.index(value.split()[-1])
            delta = (day - local_day.weekday()) % 7
            if value.startswith("next "):
                delta += 7
            date_days.add(local_day + timedelta(days=delta))
        else:
            notes.append(f"unsupported DATE: {phrase}")
            partial = True
    week_start, week_end = (
        local_week(request, binding=binding) if week else (None, None)
    )

    def date_match(row: CalendarEvent) -> bool:
        instant = clock(row.start)
        if later and instant <= now:
            return False
        if later and instant >= week_end:
            return False
        in_week = bool(
            week and week_start <= instant < week_end and (not later or now < instant)
        )
        return (
            not (week or date_days)
            or in_week
            or (instant.astimezone(zone).date() in date_days)
        )

    # Establish authorship before looking at calendar titles. A null endpoint
    # must not be interpreted as the account owner.
    associated = {
        m.record_id: m
        for m in context.history
        if request.owner_id in (m.sender_contact_id, m.recipient_contact_id)
        and {m.sender_contact_id, m.recipient_contact_id} & seeds
    }
    bridges: dict[str, set[str]] = {}
    for event in context.calendar:
        if not date_match(event):
            continue
        title_tokens = _tokens(event.title)
        support = {
            rid
            for rid, message in associated.items()
            if len(title_tokens & _tokens(message.text)) >= 2
            and (
                not groups
                or any(g <= title_tokens & _tokens(message.text) for g in groups)
            )
        }
        if support:
            bridges[event.record_id] = support
            queries.append(
                SearchQuery(
                    channel="associated",
                    text=f"title bridge {event.record_id} <- {', '.join(sorted(support))}",
                    contact_ids=sorted(
                        {
                            cid
                            for rid in support
                            for cid in (
                                associated[rid].sender_contact_id,
                                associated[rid].recipient_contact_id,
                            )
                            if cid in seeds
                        }
                    ),
                )
            )

    original_tokens = _tokens(request.text)
    scores: dict[str, int] = {}
    for row in rows:
        if isinstance(row, Contact):
            score = 100 if row.contact_id in seeds else 0
        else:
            text = row.title if isinstance(row, CalendarEvent) else row.text
            tokens = _tokens(text)
            score = len(tokens & original_tokens)
            score += 20 * sum(group <= tokens for group in groups)
            if isinstance(row, CalendarEvent):
                if not date_match(row):
                    continue
                if set(row.participant_contact_ids) & seeds:
                    score += 5
                if week or date_days:
                    score += 1
                if row.record_id in bridges:
                    score += 12
            else:
                endpoints = {row.sender_contact_id, row.recipient_contact_id}
                if row.record_id in associated:
                    score += 5
                    if endpoints & typed_seeds:
                        score += 10
        if score:
            scores[row.record_id] = score

    def closure(rid: str) -> set[str]:
        joined = {rid} | bridges.get(rid, set())
        ids = set()
        for source_id in joined:
            row = records[source_id]
            if isinstance(row, Contact):
                ids.add(row.contact_id)
            elif isinstance(row, CalendarEvent):
                ids.update(row.participant_contact_ids)
            else:
                ids.update((row.sender_contact_id, row.recipient_contact_id))
        # Overlapping aliases can connect multiple ambiguous groups.
        while True:
            enlarged = ids | set().union(*(g for g in ambiguity if g & ids))
            if enlarged == ids:
                break
            ids = enlarged
        return joined | {contacts[cid].record_id for cid in ids if cid in contacts}

    ranked = sorted(scores, key=lambda rid: (-scores[rid], rid))
    matched = set().union(*(closure(rid) for rid in ranked))
    packed: set[str] = set()
    packed_order: list[str] = []
    omitted_ambiguity = False
    for rid in ranked:
        joined = closure(rid)
        if len(packed | joined) > max_records:
            partial = True
            omitted_ambiguity |= any(
                len({contacts[c].record_id for c in g} & joined) > 1 for g in ambiguity
            )
            continue
        for new in [rid, *sorted(joined - {rid})]:
            if new not in packed:
                packed.add(new)
                packed_order.append(new)
    if matched - packed:
        notes.append(
            "Record budget omitted matched evidence with its required source joins."
        )
    if omitted_ambiguity:
        notes.append(
            "Ambiguous name group omitted atomically: budget cannot represent all matches."
        )
    selected = [records[rid] for rid in packed_order]
    evidence = PublicContext(
        contacts=[r for r in selected if isinstance(r, Contact)],
        calendar=[r for r in selected if isinstance(r, CalendarEvent)],
        history=[r for r in selected if isinstance(r, HistoryMessage)],
    )
    candidates = CandidateEvidence(
        contact_ids=[r.contact_id for r in evidence.contacts],
        event_ids=[r.event_id for r in evidence.calendar],
        source_record_ids=packed_order,
    )
    bundle.validate_candidates(candidates)
    return SearchResult(
        candidates=candidates,
        evidence=evidence,
        queries=queries,
        coverage=Coverage(
            eligible_records=len(rows),
            matched_records=len(matched),
            packed_records=len(packed),
            partial=partial,
            notes=notes,
        ),
    )
