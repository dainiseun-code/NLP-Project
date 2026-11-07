"""Public-only JSONL loading and eligibility. No retrieval or prompt rendering."""

from pathlib import Path

from pydantic import Field, model_validator

from messaging_nlp.contracts import (
    CandidateEvidence,
    Decision,
    NonBlank,
    Request,
    StrictModel,
    clock,
    unique,
)
from messaging_nlp.io import read_jsonl


class Contact(StrictModel):
    record_id: NonBlank
    contact_id: NonBlank
    version: int = Field(ge=1)
    name: NonBlank
    aliases: list[NonBlank]
    relationship: NonBlank
    available_at: str
    provenance: NonBlank

    @model_validator(mode="after")
    def valid_time(self):
        clock(self.available_at)
        unique(self.aliases, "aliases")
        return self


class CalendarEvent(StrictModel):
    record_id: NonBlank
    event_id: NonBlank
    version: int = Field(ge=1)
    title: NonBlank
    participant_contact_ids: list[NonBlank]
    start: str
    end: str | None = None
    location: NonBlank | None = None
    status: str = Field(pattern=r"^(scheduled|cancelled)$")
    available_at: str
    provenance: NonBlank

    @model_validator(mode="after")
    def valid_time(self):
        clock(self.available_at)
        start = clock(self.start)
        if self.end is not None and clock(self.end) <= start:
            raise ValueError("event end must follow start")
        unique(self.participant_contact_ids, "participants")
        return self


class HistoryMessage(StrictModel):
    record_id: NonBlank
    sender_contact_id: NonBlank | None
    recipient_contact_id: NonBlank | None
    text: NonBlank
    timestamp: str
    available_at: str
    provenance: NonBlank

    @model_validator(mode="after")
    def valid_time(self):
        clock(self.timestamp)
        clock(self.available_at)
        return self


class Snapshot(StrictModel):
    context_snapshot_id: NonBlank
    timestamp: str
    record_ids: list[NonBlank]

    @model_validator(mode="after")
    def valid_time(self):
        clock(self.timestamp)
        unique(self.record_ids, "snapshot record IDs")
        return self


class PublicContext(StrictModel):
    contacts: list[Contact]
    calendar: list[CalendarEvent]
    history: list[HistoryMessage]


class PublicBundle(StrictModel):
    """Internal provenance packet. Source IDs must be aliased before prompting."""

    request: Request
    context: PublicContext

    def validate_candidates(self, candidates: CandidateEvidence) -> None:
        contacts = {row.contact_id for row in self.context.contacts}
        events = {row.event_id for row in self.context.calendar}
        records = {
            row.record_id
            for rows in (
                self.context.contacts,
                self.context.calendar,
                self.context.history,
            )
            for row in rows
        }
        if not set(candidates.contact_ids) <= contacts:
            raise ValueError("candidate contact not eligible")
        if not set(candidates.event_ids) <= events:
            raise ValueError("candidate event not eligible")
        if not set(candidates.source_record_ids) <= records:
            raise ValueError("candidate source not eligible")

    def validate_decision(
        self, decision: Decision, candidates: CandidateEvidence
    ) -> None:
        """Check references and literal facts, not arbitrary natural-language claims."""
        self.validate_candidates(candidates)
        selection = decision.selection
        if selection is None:
            return
        if not set(selection.recipient_contact_ids) <= set(candidates.contact_ids):
            raise ValueError("selected recipient outside candidates")
        if (
            selection.selected_event_id is not None
            and selection.selected_event_id not in candidates.event_ids
        ):
            raise ValueError("selected event outside candidates")
        if not set(selection.source_record_ids) <= set(candidates.source_record_ids):
            raise ValueError("selected source outside candidates")
        records = {
            row.record_id: row
            for rows in (
                self.context.contacts,
                self.context.calendar,
                self.context.history,
            )
            for row in rows
        }
        for cid in selection.recipient_contact_ids:
            if not any(
                isinstance(records[rid], Contact) and records[rid].contact_id == cid
                for rid in selection.source_record_ids
            ):
                raise ValueError("recipient requires its contact source")
        event = next(
            (
                row
                for row in self.context.calendar
                if row.event_id == selection.selected_event_id
            ),
            None,
        )
        if event is not None and event.record_id not in selection.source_record_ids:
            raise ValueError("selected event requires its source")
        for fact in selection.facts:
            for rid in fact.source_record_ids:
                source = records[rid]
                if fact.field in ("title", "start", "end", "location"):
                    if not isinstance(source, CalendarEvent) or source != event:
                        raise ValueError("event fact must reference selected event")
                    expected = getattr(source, fact.field)
                elif fact.field == "name":
                    if not isinstance(source, Contact):
                        raise ValueError("name fact must reference a contact")
                    expected = source.name
                else:
                    if not isinstance(source, HistoryMessage):
                        raise ValueError("message fact must reference history")
                    expected = source.text
                if expected is None:
                    raise ValueError("selected fact is absent from source")
                equal = (
                    clock(expected) == clock(fact.value)
                    if fact.field in ("start", "end")
                    else expected == fact.value
                )
                if not equal:
                    raise ValueError("selected fact contradicts source")


class PublicDataset(StrictModel):
    """Account identity comes from the caller's explicit policy, never null endpoints."""

    owner_id: NonBlank
    requests: list[Request]
    contacts: list[Contact]
    calendar: list[CalendarEvent]
    history: list[HistoryMessage]
    snapshots: list[Snapshot]

    @model_validator(mode="after")
    def integrity(self):
        unique([r.message_id for r in self.requests], "request IDs")
        unique([s.context_snapshot_id for s in self.snapshots], "snapshot IDs")
        all_records = [*self.contacts, *self.calendar, *self.history]
        unique([r.record_id for r in all_records], "record IDs")
        contact_ids = {r.contact_id for r in self.contacts}
        if self.owner_id in contact_ids:
            raise ValueError("owner must not alias a contact")
        identities = contact_ids | {self.owner_id}
        for row in self.history:
            if any(
                endpoint is not None and endpoint not in identities
                for endpoint in (row.sender_contact_id, row.recipient_contact_id)
            ):
                raise ValueError("unknown message endpoint/owner")
            # Both known endpoints describe an account exchange, not a private
            # conversation between unrelated third parties. Null stays unknown.
            if (
                row.sender_contact_id is not None
                and row.recipient_contact_id is not None
                and self.owner_id
                not in (row.sender_contact_id, row.recipient_contact_id)
            ):
                raise ValueError("message exchange does not involve owner")
        for row in self.calendar:
            if not set(row.participant_contact_ids) <= identities:
                raise ValueError("unknown event participant/owner")
        for rows, field in ((self.contacts, "contact_id"), (self.calendar, "event_id")):
            unique([(getattr(r, field), r.version) for r in rows], "identity versions")
            grouped = {}
            for row in rows:
                grouped.setdefault(getattr(row, field), []).append(row)
            for versions in grouped.values():
                ordered = sorted(versions, key=lambda row: row.version)
                if any(
                    clock(a.available_at) > clock(b.available_at)
                    for a, b in zip(ordered, ordered[1:])
                ):
                    raise ValueError("version availability contradicts version order")
        record_ids = {r.record_id for r in all_records}
        snapshots = {s.context_snapshot_id: s for s in self.snapshots}
        for snapshot in self.snapshots:
            if not set(snapshot.record_ids) <= record_ids:
                raise ValueError("broken snapshot record reference")
        for request in self.requests:
            if request.owner_id != self.owner_id:
                raise ValueError("request owner differs from account owner")
            if request.context_snapshot_id not in snapshots:
                raise ValueError("unknown request snapshot")
        return self

    def bundle(self, message_id: str) -> PublicBundle:
        request = next(
            (row for row in self.requests if row.message_id == message_id), None
        )
        if request is None:
            raise ValueError("unknown request")
        snapshot = next(
            s
            for s in self.snapshots
            if s.context_snapshot_id == request.context_snapshot_id
        )
        # A union export can include snapshots later than an individual request.
        # Neither the snapshot clock nor the request clock may grant access to
        # information after the earlier instant. Future listed records still
        # fail closed below; a later snapshot alone is not invalid evidence.
        cutoff = min(clock(snapshot.timestamp), clock(request.request_time))
        records = {
            r.record_id: r for r in [*self.contacts, *self.calendar, *self.history]
        }
        chosen = [records[rid] for rid in snapshot.record_ids]
        latest = {}
        for rows, field in ((self.contacts, "contact_id"), (self.calendar, "event_id")):
            for row in rows:
                if clock(row.available_at) <= cutoff:
                    key = (field, getattr(row, field))
                    old = latest.get(key)
                    if old is None or row.version > old.version:
                        latest[key] = row
        for row in chosen:
            if clock(row.available_at) > cutoff:
                raise ValueError("future information availability in snapshot")
            if isinstance(row, HistoryMessage) and clock(row.timestamp) > cutoff:
                raise ValueError("future message timestamp in snapshot")
            field = (
                "contact_id"
                if isinstance(row, Contact)
                else "event_id"
                if isinstance(row, CalendarEvent)
                else None
            )
            if (
                field
                and latest[(field, getattr(row, field))].record_id != row.record_id
            ):
                raise ValueError("stale contact/calendar version in snapshot")
        selected_contacts = {
            row.contact_id for row in chosen if isinstance(row, Contact)
        }
        allowed = selected_contacts | {self.owner_id}
        for row in chosen:
            refs = (
                row.participant_contact_ids
                if isinstance(row, CalendarEvent)
                else [row.sender_contact_id, row.recipient_contact_id]
                if isinstance(row, HistoryMessage)
                else []
            )
            if any(ref is not None and ref not in allowed for ref in refs):
                raise ValueError("snapshot lacks referenced eligible contact")
        return PublicBundle(
            request=request,
            context=PublicContext(
                contacts=[r for r in chosen if isinstance(r, Contact)],
                calendar=[r for r in chosen if isinstance(r, CalendarEvent)],
                history=[r for r in chosen if isinstance(r, HistoryMessage)],
            ),
        )


def load_public_dataset(public_root: Path, *, owner_id: str) -> PublicDataset:
    """Read only five allowlisted files inside the explicit public directory.

    Gold, roster, provenance and prior assistant state are never opened. Reject
    redirected public files so a symlink cannot cross the public directory.
    """
    root = Path(public_root).resolve(strict=True)
    content = {}
    for name in ("requests", "contacts", "calendar", "history", "snapshots"):
        path = root / f"{name}.jsonl"
        if path.resolve(strict=True).parent != root:
            raise ValueError("public input path leaves public directory")
        content[name] = read_jsonl(path)
    return PublicDataset(owner_id=owner_id, **content)


def load_request_bundle(
    public_root: Path, message_id: str, *, owner_id: str
) -> PublicBundle:
    return load_public_dataset(public_root, owner_id=owner_id).bundle(message_id)
