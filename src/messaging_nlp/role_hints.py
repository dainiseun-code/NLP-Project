"""Syntax-only addressed/subject hints from the request text.

These hints are evidence, not a recipient selection. They match displayed contact
names/aliases against bounded address cues and report ambiguity; the model still
chooses the action and recipient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ADDRESS_CUE = re.compile(
    r"\b(?:ask|tell|message|contact|remind|inform|notify|send|write|draft|prepare)\b"
    r"|\bfind\s+out\s+from\b|\b(?:check|confirm)\s+with\b"
    r"|\blet\b|\bto\b|\basking\b",
    re.IGNORECASE,
)
_COMPLEMENTIZER = re.compile(r"\b(?:whether|if|about|that)\b", re.IGNORECASE)
_CLAUSE_START = re.compile(r"(?:^|[.!?;:\n])\s*")


@dataclass(frozen=True)
class Mention:
    contact_id: str
    name: str
    surface: str
    kind: str
    start: int
    end: int


def _mentions(text: str, contacts: list[dict]) -> list[Mention]:
    raw: list[Mention] = []
    for contact in contacts:
        surfaces = [(contact["name"], "full_name")]
        surfaces.extend((alias, "alias") for alias in contact["aliases"])
        for surface, kind in surfaces:
            if not surface:
                continue
            # Apostrophes are word-internal here: the alias "T" must not match
            # the "t" in "won't". A real standalone alias is bounded by space or
            # punctuation other than an apostrophe.
            for match in re.finditer(
                rf"(?<![\w'’]){re.escape(surface)}(?![\w'’])", text, re.IGNORECASE
            ):
                raw.append(
                    Mention(
                        contact["contact_id"],
                        contact["name"],
                        match.group(),
                        kind,
                        match.start(),
                        match.end(),
                    )
                )
    # Any full name shadows a shared alias inside it, including that contact's
    # own alias: "Maya Patel" is not the alias "Maya" for ambiguity purposes.
    full = [mention for mention in raw if mention.kind == "full_name"]
    return [
        mention
        for mention in raw
        if not (
            mention.kind == "alias"
            and any(
                other.start <= mention.start and mention.end <= other.end
                for other in full
            )
        )
    ]


def _addressed(text: str, mention: Mention) -> bool:
    prefix = text[: mention.start]
    clause_start = 0
    for boundary in _CLAUSE_START.finditer(prefix):
        clause_start = boundary.end()
    clause = prefix[clause_start:]
    cues = list(_ADDRESS_CUE.finditer(clause))
    if not cues:
        return False
    # A complementizer between the last cue and the mention introduces a
    # subject or content clause, not another addressee.
    return not _COMPLEMENTIZER.search(clause[cues[-1].end() :])


def _dedupe(mentions: list[Mention]) -> list[Mention]:
    """One surface per contact, preferring a full name over an alias."""
    selected: dict[str, Mention] = {}
    for mention in sorted(mentions, key=lambda item: (item.start, item.contact_id)):
        current = selected.get(mention.contact_id)
        if current is None or mention.kind == "full_name":
            selected[mention.contact_id] = mention
    return sorted(selected.values(), key=lambda item: (item.start, item.contact_id))


def single_addressed_contact(text: str, contacts: list[dict]) -> str | None:
    """Return the one syntactically addressed contact, or None if ambiguous."""
    hints = role_hints(text, contacts)
    if hints["unresolved_identity"] or len(hints["addressed"]) != 1:
        return None
    return hints["addressed"][0]["contact_id"]


def role_hints(text: str, contacts: list[dict]) -> dict:
    """Return syntax-only addressed/subject names and ambiguity state."""
    mentions = _mentions(text, contacts)
    addressed: list[Mention] = []
    subject: list[Mention] = []
    for mention in mentions:
        (addressed if _addressed(text, mention) else subject).append(mention)
    addressed = _dedupe(addressed)
    addressed_ids = {item.contact_id for item in addressed}
    subject = [
        item for item in _dedupe(subject) if item.contact_id not in addressed_ids
    ]

    # An addressed alias is unresolved when another displayed contact shares it
    # and no full-name surface distinguishes that contact in the same request.
    unresolved = False
    for mention in addressed:
        surface = mention.surface.casefold()
        if any(
            other["contact_id"] != mention.contact_id
            and (
                (
                    mention.kind == "alias"
                    and surface in {item.casefold() for item in other["aliases"]}
                )
                or (mention.kind == "full_name" and other["name"].casefold() == surface)
            )
            for other in contacts
        ):
            unresolved = True
            break
    return {
        "addressed": [
            {
                "contact_id": item.contact_id,
                "name": item.name,
                "surface": item.surface,
                "match": item.kind,
            }
            for item in addressed
        ],
        "subject_only": [
            {
                "contact_id": item.contact_id,
                "name": item.name,
                "surface": item.surface,
                "match": item.kind,
            }
            for item in subject
        ],
        "unresolved_identity": unresolved,
        "rule": "syntax_only_addressed_names",
        "note": "evidence only; the model still chooses the recipient",
    }
