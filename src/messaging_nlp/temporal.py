"""Civil Sunday week windows over absolute instants; no inferred intent."""

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from messaging_nlp.contracts import Request, clock


@dataclass(frozen=True)
class TimezoneBinding:
    """One trusted IANA-name-to-artifact assertion supplied by the caller.

    Populate all three fields from a trusted retained identity (resolving its path
    against the artifact root), not from the request or a guessed TZif filename.
    TZif contains no IANA identity: this class checks completeness, not the truth
    of the caller's assertion. The caller must authenticate that identity receipt
    and reconcile any separate policy path/hash before constructing this binding.
    """

    time_zone: str
    tzif_path: Path
    tzif_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.time_zone, str)
            or not self.time_zone
            or self.time_zone != self.time_zone.strip()
        ):
            raise ValueError("timezone binding requires an exact nonempty IANA name")
        if not isinstance(self.tzif_path, Path):
            raise TypeError("timezone binding requires a Path")
        if not isinstance(self.tzif_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.tzif_sha256
        ):
            raise ValueError("timezone binding requires a SHA-256 hex digest")


def request_zone(
    request: Request,
    *,
    binding: TimezoneBinding | None = None,
) -> ZoneInfo:
    """Use retained TZif bytes when supplied, rejecting stale/missing identity.

    Frozen study consumers must supply one trusted name/path/hash binding.
    Host ZoneInfo is used only when no binding is supplied (unfrozen checks).
    A supplied binding never falls back to host rules, even on load failure.
    """
    if binding is None:
        return ZoneInfo(request.time_zone)
    if not isinstance(binding, TimezoneBinding):
        raise TypeError("supply one TimezoneBinding, not a mapping or collection")
    if request.time_zone != binding.time_zone:
        raise ValueError("request zone does not match trusted timezone binding")
    content = binding.tzif_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != binding.tzif_sha256:
        raise ValueError("timezone artifact hash mismatch")
    # Hash and parse the same bytes; reopening the path would allow a file race.
    return ZoneInfo.from_file(BytesIO(content), key=binding.time_zone)


def local_week(
    request: Request,
    *,
    binding: TimezoneBinding | None = None,
) -> tuple[datetime, datetime]:
    zone = request_zone(request, binding=binding)
    local = clock(request.request_time).astimezone(zone)
    sunday = local.date() - timedelta(days=(local.weekday() + 1) % 7)

    def boundary(day: date) -> datetime:
        # fold=0 chooses the first occurrence of repeated midnight. Roundtrip
        # moves nonexistent midnight forward through a civil clock gap.
        value = datetime.combine(day, time.min, tzinfo=zone)
        return value.astimezone(timezone.utc).astimezone(zone)

    return boundary(sunday), boundary(sunday + timedelta(days=7))


def later_this_week(
    event_start: str,
    request: Request,
    *,
    binding: TimezoneBinding | None = None,
) -> bool:
    _, end = local_week(request, binding=binding)
    return (
        clock(request.request_time) < clock(event_start) < end.astimezone(timezone.utc)
    )
