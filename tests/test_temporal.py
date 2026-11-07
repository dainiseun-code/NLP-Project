"""Policy-derived software checks; fictional requests, not study observations."""

import hashlib
import json
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import TZPATH

import pytest

from messaging_nlp.contracts import Request
from messaging_nlp.temporal import (
    TimezoneBinding,
    later_this_week,
    local_week,
    request_zone,
)


def request(at="2026-06-10T12:00:00+00:00", zone="Etc/UTC"):
    return Request(
        message_id="fictional-request",
        text="Draft a note about the gathering later this week.",
        owner_id="fictional-owner",
        request_time=at,
        time_zone=zone,
        context_snapshot_id="fictional-snapshot",
    )


@pytest.fixture
def utc_binding(tmp_path):
    # Copy standard TZif bytes, not research data, into an isolated fixture.
    source = next(
        Path(root) / "Etc/UTC" for root in TZPATH if (Path(root) / "Etc/UTC").is_file()
    )
    path = tmp_path / "utc.tzif"
    path.write_bytes(source.read_bytes())
    return TimezoneBinding(
        "Etc/UTC", path, hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_offset_strings_are_instants_and_original_is_preserved():
    value = request("2026-06-07T01:00:00+02:00")
    start, end = local_week(value)
    assert value.request_time == "2026-06-07T01:00:00+02:00"
    assert start.isoformat() == "2026-05-31T00:00:00+00:00"
    assert end.isoformat() == "2026-06-07T00:00:00+00:00"


@pytest.mark.parametrize(
    "at,start,end",
    [
        ("2026-06-06T23:59:59+00:00", "2026-05-31", "2026-06-07"),
        ("2026-06-07T00:00:00+00:00", "2026-06-07", "2026-06-14"),
        ("2026-06-14T00:00:00+00:00", "2026-06-14", "2026-06-21"),
    ],
)
def test_sunday_inclusive_rollover(at, start, end):
    boundaries = local_week(request(at))
    assert [value.isoformat() for value in boundaries] == [
        f"{start}T00:00:00+00:00",
        f"{end}T00:00:00+00:00",
    ]


@pytest.mark.parametrize(
    "event,expected",
    [
        ("2026-06-10T11:59:59+00:00", False),
        ("2026-06-10T12:00:00+00:00", False),
        ("2026-06-10T14:00:00+02:00", False),
        ("2026-06-10T12:00:00.000001+00:00", True),
        ("2026-06-13T23:59:59+00:00", True),
        ("2026-06-14T00:00:00+00:00", False),
        ("2026-06-14T01:00:00+00:00", False),
    ],
)
def test_later_is_strict_and_week_end_exclusive(event, expected):
    assert later_this_week(event, request()) is expected


@pytest.mark.parametrize(
    "at,start,end,hours",
    [
        (
            "2025-03-09T12:00:00-04:00",
            "2025-03-09T00:00:00-05:00",
            "2025-03-16T00:00:00-04:00",
            167,
        ),
        (
            "2025-11-02T12:00:00-05:00",
            "2025-11-02T00:00:00-04:00",
            "2025-11-09T00:00:00-05:00",
            169,
        ),
    ],
)
def test_dst_weeks_use_civil_boundaries(at, start, end, hours):
    first, last = local_week(request(at, "America/New_York"))
    assert (first.isoformat(), last.isoformat()) == (start, end)
    assert last.astimezone(timezone.utc) - first.astimezone(timezone.utc) == timedelta(
        hours=hours
    )


def test_differing_source_offset_converts_validly():
    value = request("2025-03-09T04:30:00+00:00", "America/New_York")
    assert local_week(value)[1].isoformat() == "2025-03-09T00:00:00-05:00"
    assert later_this_week("2025-03-09T04:45:00+00:00", value)
    assert not later_this_week("2025-03-09T05:00:00+00:00", value)


def test_repeated_dst_clock_compares_instants_not_wall_times():
    value = request("2025-11-02T01:45:00-04:00", "America/New_York")
    assert later_this_week("2025-11-02T01:15:00-05:00", value)
    assert not later_this_week("2025-11-02T01:45:00-04:00", value)


def test_retained_binding_propagates_through_public_functions(utc_binding):
    assert request_zone(request(), binding=utc_binding).key == "Etc/UTC"
    assert local_week(request(), binding=utc_binding)[1].isoformat() == (
        "2026-06-14T00:00:00+00:00"
    )
    assert later_this_week("2026-06-11T00:00:00Z", request(), binding=utc_binding)


def test_correct_file_hash_cannot_be_relabelled_as_request_zone(utc_binding):
    with pytest.raises(ValueError, match="zone.*binding"):
        request_zone(request(zone="America/New_York"), binding=utc_binding)


def test_zone_alias_requires_exact_identity(utc_binding):
    with pytest.raises(ValueError, match="zone.*binding"):
        request_zone(request(zone="UTC"), binding=utc_binding)


def test_stale_hash_fails_closed(utc_binding):
    utc_binding.tzif_path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="hash"):
        request_zone(request(), binding=utc_binding)


def test_substituted_valid_wrong_zone_file_fails_hash_binding(utc_binding):
    source = next(
        Path(root) / "America/New_York"
        for root in TZPATH
        if (Path(root) / "America/New_York").is_file()
    )
    utc_binding.tzif_path.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="hash"):
        request_zone(request(), binding=utc_binding)


def test_missing_file_fails_closed(utc_binding):
    utc_binding.tzif_path.unlink()
    with pytest.raises(FileNotFoundError):
        request_zone(request(), binding=utc_binding)


def test_bad_tzif_with_matching_hash_does_not_fall_back(tmp_path):
    path = tmp_path / "invalid.tzif"
    path.write_bytes(b"not TZif")
    binding = TimezoneBinding(
        "Etc/UTC", path, hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        request_zone(request(), binding=binding)


@pytest.mark.parametrize("member", ["time_zone", "tzif_path", "tzif_sha256"])
def test_missing_binding_member_rejected(utc_binding, member):
    fields = {
        name: getattr(utc_binding, name)
        for name in ("time_zone", "tzif_path", "tzif_sha256")
        if name != member
    }
    with pytest.raises(TypeError):
        TimezoneBinding(**fields)


@pytest.mark.parametrize(
    "member,value",
    [
        ("time_zone", None),
        ("time_zone", ""),
        ("time_zone", " Etc/UTC"),
        ("tzif_path", None),
        ("tzif_path", "utc.tzif"),
        ("tzif_sha256", None),
        ("tzif_sha256", ""),
        ("tzif_sha256", "stale"),
    ],
)
def test_invalid_binding_member_rejected(utc_binding, member, value):
    fields = {
        name: getattr(utc_binding, name)
        for name in ("time_zone", "tzif_path", "tzif_sha256")
    }
    fields[member] = value
    with pytest.raises((TypeError, ValueError)):
        TimezoneBinding(**fields)


@pytest.mark.parametrize("binding", [{}, [], ["Etc/UTC", "America/New_York"]])
def test_non_singular_or_untyped_bindings_rejected(binding):
    with pytest.raises(TypeError):
        request_zone(request(), binding=binding)


def test_bare_path_and_hash_are_not_an_identity(utc_binding):
    with pytest.raises(TypeError):
        request_zone(
            request(),
            tzif_path=utc_binding.tzif_path,
            tzif_sha256=utc_binding.tzif_sha256,
        )


def test_ambiguous_old_and_new_binding_inputs_rejected(utc_binding):
    with pytest.raises(TypeError):
        request_zone(request(), binding=utc_binding, tzif_path=utc_binding.tzif_path)


def test_retained_binding_does_not_consult_host(monkeypatch, utc_binding):
    value = request()

    class RetainedOnly:
        @staticmethod
        def from_file(source, key):
            from zoneinfo import ZoneInfo

            return ZoneInfo.from_file(source, key=key)

    monkeypatch.setattr("messaging_nlp.temporal.ZoneInfo", RetainedOnly)
    assert request_zone(value, binding=utc_binding).key == "Etc/UTC"


@pytest.fixture
def retained_vancouver_binding():
    # Portable retained bytes only; no study request, output or data mutation.
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (root / "data/stage1-v1/provenance/timezones/identity.json").read_text()
    )
    return TimezoneBinding(
        receipt["time_zone"], root / receipt["tzif_path"], receipt["tzif_sha256"]
    )


def test_retained_vancouver_2026e_november_offset(retained_vancouver_binding):
    binding = retained_vancouver_binding
    value = request("2026-11-02T09:00:00-07:00", "America/Vancouver")
    zone = request_zone(value, binding=binding)
    from datetime import datetime

    assert datetime.fromisoformat(value.request_time).astimezone(zone).isoformat() == (
        "2026-11-02T09:00:00-07:00"
    )
    assert local_week(value, binding=binding)[1].isoformat() == (
        "2026-11-08T00:00:00-07:00"
    )


@pytest.mark.parametrize(
    "event,expected",
    [
        ("2026-11-02T15:59:59Z", False),
        ("2026-11-02T16:00:00Z", False),
        ("2026-11-02T09:00:00-07:00", False),
        ("2026-11-02T16:00:00.000001Z", True),
        ("2026-11-08T06:59:59.999999Z", True),
        ("2026-11-08T07:00:00Z", False),
        ("2026-11-08T00:00:00-07:00", False),
        ("2026-11-08T07:30:00Z", False),
    ],
)
def test_retained_vancouver_later_this_week_strict_boundaries(
    retained_vancouver_binding, event, expected
):
    value = request("2026-11-02T09:00:00-07:00", "America/Vancouver")
    # UTC examples distinguish the retained -07 Sunday end from obsolete -08.
    assert later_this_week(event, value, binding=retained_vancouver_binding) is expected
