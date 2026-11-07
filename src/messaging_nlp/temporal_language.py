"""Finite English temporal rendering grammar, not a prose/intent interpreter.

No host timezone normalization is allowed here. Callers supply the authenticated
zone loaded from their trusted TimezoneBinding. Scanning is deliberately broader
than parsing so recognized ambiguous/malformed forms fail rather than disappear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_SMALL = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_NUMBERS = {word: n for n, word in enumerate(_SMALL)}
_NUMBERS.update(
    {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
)
_WORD = "(?:" + "|".join(_NUMBERS) + ")"
_NUMBER = rf"(?:\d+|{_WORD})"
_MINUTE = rf"{_WORD}(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?"
_QUALIFIER = (
    r"(?:a\.m\.?|p\.m\.?|am|pm|in the morning|in the afternoon|in the evening|at night)"
)
_CLOCK_PATTERN = (
    rf"\b(?:(?:half|quarter)\s+(?:past|to)\s+{_NUMBER}(?:\s*{_QUALIFIER})?"
    rf"|{_NUMBER}(?:[ -]{_MINUTE})?(?:\s+o['’]clock)?\s*{_QUALIFIER}"
    rf"|{_NUMBER}\s+o['’]clock(?:\s+{_QUALIFIER})?"
    rf"|\d+:\d+(?::\d+)?(?:\s*{_QUALIFIER})?"
    r"|noon|midnight)(?!\w)"
)
_CLOCK = re.compile(_CLOCK_PATTERN, re.IGNORECASE)
_BARE = re.compile(
    rf"\b(?:at|by|until|from|to)\s+(?P<clock>{_NUMBER})(?![\w:])",
    re.IGNORECASE,
)
_MONTHS = (
    "January February March April May June July August September October November December"
).split()
_MONTH_ABBR = "Jan Feb Mar Apr May Jun Jul Aug Sep Sept Oct Nov Dec".split()
_MONTH = "(?:" + "|".join((*_MONTHS, *_MONTH_ABBR)) + r")\.?"
_WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
_WEEKDAY = "(?:" + "|".join(_WEEKDAYS) + ")"
_DATE_PATTERN = (
    rf"\b(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}"
    rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:\s+\d{{4}})?"
    rf"|(?:(?:this|next)\s+)?{_WEEKDAY}"
    r"|today|tomorrow|yesterday)(?!\w)"
)
_DATE = re.compile(_DATE_PATTERN, re.IGNORECASE)
_UNSUPPORTED = re.compile(
    r"\b(?:tonight|[A-Za-z]+day after tomorrow|day after tomorrow|"
    r"day before yesterday|(?:next|last|this) (?:week|month|year)|"
    r"later this week|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}|"
    rf"(?:half|quarter)\s+(?:(?:after|before)\s+)?{_NUMBER}(?:\s*{_QUALIFIER})?|"
    rf"half\s+to\s+{_NUMBER}(?:\s*{_QUALIFIER})?|"
    rf"{_NUMBER}\s+(?:past|to|after|before)\s+{_NUMBER}(?:\s*{_QUALIFIER})?|"
    rf"last\s+{_WEEKDAY}|"
    rf"(?:today|tomorrow|yesterday|{_WEEKDAY})\s+(?:morning|afternoon|evening|night)|"
    r"UTC|GMT|PST|PDT|EST|EDT|CST|CDT|MST|MDT)\b",
    re.IGNORECASE,
)

RENDERING_POLICY = {
    "version": 1,
    "clock": (
        "24-hour H:MM; 1..12 numeric/written hour with am/pm (dots optional) or "
        "in the morning/afternoon/evening; optional written minute zero..fifty-nine "
        "or o'clock; qualified half past/quarter past/quarter to; noon/midnight. "
        "Bare hours, unqualified half/quarter/o'clock, at night and seconds fail."
    ),
    "date": (
        "ISO YYYY-MM-DD; full or common abbreviated English month D[ordinal][, YYYY] "
        "or D[ordinal] month [YYYY] (e.g. Nov 3, 2026); omitted year is request-local "
        "year. today/tomorrow/yesterday; "
        "weekday/this weekday is the next occurrence including today; next weekday "
        "is that occurrence plus seven days. Local calendar arithmetic, not 24h deltas."
    ),
    "datetime": "Exactly DATE at CLOCK or CLOCK on DATE; no nested mentions.",
    "literal": "Exact entire selected literal; excerpt is an exact contiguous substring.",
    "faithful_quote": (
        "Paired straight/curly quotes, exact contiguous source message excerpt or "
        "exact precomputed request quotation; straight within-word apostrophes are "
        "not delimiters, including inside single quotes. Provenance only, not current schedule."
    ),
    "proposal": (
        "Whole local clauses: Could/Can we move EVENT to CLOCK; Would CLOCK work "
        "instead for EVENT; Ask [NAME] whether EVENT can move to CLOCK; Propose [moving "
        "EVENT to|EVENT at|at] CLOCK. Event is nonempty non-temporal clause text. "
        "Initial recognition requires exactly one clock and no dates in the clause. "
        "Proposal clocks cannot certify current-schedule clauses."
    ),
    "current_start": (
        "Whole quote-masked clauses: Confirm [that] EVENT starts/begins [at DETAIL]; "
        "Tell/Inform NAME [that] EVENT starts/begins [at DETAIL]; Confirm [the] "
        "[current/scheduled] start [time] of/for EVENT; Tell/Inform NAME of/about "
        "[the] [current/scheduled] start [time] of/for EVENT. These require selected scheduled event and an expressed "
        "start clock; explicit request clocks must agree. Unknown-time/location ask "
        "clauses do not require their answers."
    ),
    "scan": (
        "Whole draft, including quotes; every recognized clock/date needs a temporal "
        "binding or faithful quote. Recognized malformed/unsupported forms fail closed: "
        "alternative half/quarter constructions, numeric/written minute past/to/after/"
        "before hour, last weekday, date+day-period, timezone abbreviations, slash/"
        "abbreviated dates, ISO timestamps and named unsupported relative idioms."
    ),
    "limitations": (
        "Finite English patterns only; unknown idioms, arbitrary prose relationships, "
        "attribution, negation and semantic task success remain unmeasured. Exact "
        "references and grammar are mechanical evidence, never semantic success. "
        "Different unique, nonoverlapping surfaces for the same supported clock "
        "(three pm and 3pm) are accepted without semantic deduplication; repeated "
        "identical or ambiguous surfaces remain representation failures."
    ),
    "clauses": (
        "Whole clauses split at . ! ? : ; or newline outside paired quotations; "
        "periods/colons within recognized temporal tokens are not separators. "
        "An am/pm abbreviation's final period may therefore keep surrounding "
        "sentence text in one clause; this is not a full sentence parser. "
        "An address prefix before a colon is a separate clause."
    ),
    "patterns": {
        "clock": _CLOCK_PATTERN,
        "date": _DATE_PATTERN,
        "unsupported": _UNSUPPORTED.pattern,
    },
}


@dataclass(frozen=True)
class TemporalOccurrence:
    start: int
    end: int
    kind: str
    value: tuple[int, int] | date


def _number(text: str) -> int:
    if text.isdigit():
        if len(text) > 2:
            raise ValueError("numeric clock components require at most two digits")
        return int(text)
    words = text.replace("-", " ").split()
    if len(words) == 1 and words[0] in _NUMBERS:
        return _NUMBERS[words[0]]
    if (
        len(words) == 2
        and words[0]
        in (
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        )
        and words[1] in _SMALL[1:10]
    ):
        return _NUMBERS[words[0]] + _NUMBERS[words[1]]
    raise ValueError("unsupported written number")


def parse_clock(surface: str) -> tuple[int, int]:
    if not _CLOCK.fullmatch(surface):
        raise ValueError("clock surface does not fully match grammar")
    text = surface.lower().replace("’", "'")
    if text in ("noon", "midnight"):
        return (12, 0) if text == "noon" else (0, 0)
    qualifier = re.search(rf"\s*({_QUALIFIER})$", text)
    period = None
    if qualifier:
        period = qualifier[1].replace(".", "")
        text = text[: qualifier.start()]
        if period == "at night":
            raise ValueError("ambiguous night clock")
    if re.fullmatch(r"\d+:\d+", text):
        hour_text, minute_text = text.split(":")
        if len(hour_text) > 2 or len(minute_text) != 2:
            raise ValueError("clock requires one/two hour digits and two minute digits")
        hour, minute = int(hour_text), int(minute_text)
    else:
        text = re.sub(r"\s+o'clock$", "", text)
        fraction = re.fullmatch(rf"(half|quarter) (past|to) ({_NUMBER})", text)
        if fraction:
            if fraction[1] == "half" and fraction[2] == "to":
                raise ValueError("half to is outside the qualified fraction grammar")
            hour = _number(fraction[3])
            minute = 30 if fraction[1] == "half" else 15
            if fraction[2] == "to":
                minute = -minute
        else:
            parts = re.fullmatch(rf"({_NUMBER})(?:[ -]({_MINUTE}))?", text)
            if not parts:
                raise ValueError("malformed clock")
            hour = _number(parts[1])
            minute = _number(parts[2]) if parts[2] else 0
        if period is None:
            raise ValueError("ambiguous unqualified clock")
    if period:
        if not 1 <= hour <= 12:
            raise ValueError("qualified clock hour outside 1..12")
        hour %= 12
        if period in ("pm", "in the afternoon", "in the evening"):
            hour += 12
    elif not 0 <= hour <= 23:
        raise ValueError("24-hour clock outside 0..23")
    if not -30 <= minute <= 59:
        raise ValueError("clock minute out of range")
    total = (hour * 60 + minute) % (24 * 60)
    return divmod(total, 60)


def parse_date(surface: str, *, local_now: datetime) -> date:
    if not _DATE.fullmatch(surface):
        raise ValueError("date surface does not fully match grammar")
    text = surface.lower()
    if text in ("today", "tomorrow", "yesterday"):
        return local_now.date() + timedelta(
            days={"today": 0, "tomorrow": 1, "yesterday": -1}[text]
        )
    weekday = re.fullmatch(rf"(?:(this|next) )?({_WEEKDAY})", text, re.IGNORECASE)
    if weekday:
        target = [s.lower() for s in _WEEKDAYS].index(weekday[2].lower())
        delta = (target - local_now.weekday()) % 7
        if weekday[1] == "next":
            delta += 7
        return local_now.date() + timedelta(days=delta)
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        # Parsing is intentionally stricter than the recognizer.
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            raise ValueError("ISO date requires two-digit month/day")
        return date.fromisoformat(text)
    ordinal = re.search(r"(\d+)(st|nd|rd|th)\b", text)
    if ordinal:
        day = int(ordinal[1])
        expected = (
            "th"
            if 10 <= day % 100 <= 20
            else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        )
        if ordinal[2] != expected:
            raise ValueError("malformed date ordinal")
    text = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", text).replace(",", "")
    parts = text.split()
    months = {s.lower(): index for index, s in enumerate(_MONTHS, 1)}
    months.update(
        {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "sept": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
    )

    def month_number(token: str) -> int:
        return months[token.rstrip(".").lower()]

    if parts[0].rstrip(".").lower() in months:
        month, day = month_number(parts[0]), int(parts[1])
    else:
        day, month = int(parts[0]), month_number(parts[1])
    return date(int(parts[2]) if len(parts) == 3 else local_now.year, month, day)


def scan_temporals(
    text: str, *, local_now: datetime, strict: bool = True
) -> list[TemporalOccurrence]:
    """Recognize all supported temporal tokens and fail closed on known bad forms."""
    occurrences = []
    spans: list[tuple[int, int]] = []
    for kind, pattern in (("date", _DATE), ("clock", _CLOCK)):
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in spans):
                continue
            spans.append(match.span())
            try:
                value = (
                    parse_clock(match[0])
                    if kind == "clock"
                    else parse_date(match[0], local_now=local_now)
                )
            except ValueError:
                if strict:
                    raise
                continue
            occurrences.append(
                TemporalOccurrence(match.start(), match.end(), kind, value)
            )
    if strict:
        for match in _BARE.finditer(text):
            start, end = match.span("clock")
            if not any(a <= start and end <= b for a, b in spans):
                raise ValueError("ambiguous bare clock in draft")
        if _UNSUPPORTED.search(text):
            raise ValueError("recognized unsupported temporal form in draft")
    return sorted(occurrences, key=lambda item: item.start)


def parse_datetime(
    surface: str, *, local_now: datetime
) -> tuple[date, tuple[int, int]]:
    for pattern, reverse in (
        (rf"({_DATE_PATTERN}) at ({_CLOCK_PATTERN})", False),
        (rf"({_CLOCK_PATTERN}) on ({_DATE_PATTERN})", True),
    ):
        match = re.fullmatch(pattern, surface, re.IGNORECASE)
        if match:
            day, at = (match[2], match[1]) if reverse else (match[1], match[2])
            return parse_date(day, local_now=local_now), parse_clock(at)
    raise ValueError("datetime surface does not fully match grammar")


def quotation_spans(text: str) -> list[tuple[int, int]]:
    """Paired quotes; straight within-word apostrophes never open or close one."""
    return [
        m.span()
        for m in re.finditer(
            r"""(?<!\w)(?:"[^"\n]+"|'(?:[^'\n]|(?<=\w)'(?=\w))+'(?!\w)|“[^”\n]+”|‘[^’\n]+’)""",
            text,
        )
    ]


def mask_quotations(text: str) -> str:
    chars = list(text)
    for start, end in quotation_spans(text):
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def clauses(text: str, *, local_now: datetime) -> list[tuple[int, int, str]]:
    masked = mask_quotations(text)
    protected = list(masked)
    for occurrence in scan_temporals(masked, local_now=local_now, strict=False):
        for index in range(occurrence.start, occurrence.end):
            if protected[index] in ".:":
                protected[index] = "~"
    boundaries = [0] + [m.end() for m in re.finditer(r"[.!?:;\n]", "".join(protected))]
    boundaries.append(len(text))
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        fragment = masked[start:end].strip(" \t\r\n.!?:;")
        if fragment:
            offset = masked.find(fragment, start, end)
            result.append((offset, offset + len(fragment), fragment))
    return result


def proposal_clocks(
    text: str, *, local_now: datetime
) -> list[tuple[int, int, TemporalOccurrence]]:
    """Return only clocks inside whole, locally recognized proposal clauses."""
    result = []
    patterns = (
        r"(?:could|can) we move (?P<event>.+?) to <CLOCK>",
        r"would <CLOCK> work instead for (?P<event>.+)",
        r"ask(?: \w+)? whether (?P<event>.+?) can move to <CLOCK>",
        r"propose moving (?P<event>.+?) to <CLOCK>",
        r"propose (?P<event>.+?) at <CLOCK>",
        r"propose at <CLOCK>",
    )
    for start, end, fragment in clauses(text, local_now=local_now):
        tokens = scan_temporals(fragment, local_now=local_now, strict=False)
        if len(tokens) != 1 or tokens[0].kind != "clock":
            continue
        token = tokens[0]
        replaced = fragment[: token.start] + "<CLOCK>" + fragment[token.end :]
        if not any(re.fullmatch(p, replaced, re.IGNORECASE) for p in patterns):
            continue
        result.append(
            (
                start,
                end,
                TemporalOccurrence(
                    start + token.start, start + token.end, "clock", token.value
                ),
            )
        )
    return result


def current_start_clauses(text: str, *, local_now: datetime) -> list[str]:
    subject = r"(?!(?:about|of|when|where|whether)\b)[\w][\w' -]*?"
    patterns = (
        rf"confirm(?: that)? {subject} (?:starts|begins)(?: at .+)?",
        rf"(?:tell|inform) \w+(?: that)? {subject} (?:starts|begins)(?: at .+)?",
        rf"confirm (?:the )?(?:(?:current|scheduled) )?start(?: time)? (?:of|for) {subject}",
        rf"(?:tell|inform) \w+ (?:of|about) (?:the )?(?:(?:current|scheduled) )?start(?: time)? (?:of|for) {subject}",
    )
    return [
        fragment
        for _, _, fragment in clauses(text, local_now=local_now)
        if any(re.fullmatch(pattern, fragment, re.IGNORECASE) for pattern in patterns)
    ]


def local_instant(value: str, *, zone: ZoneInfo) -> datetime:
    from messaging_nlp.contracts import clock

    return clock(value).astimezone(zone)
