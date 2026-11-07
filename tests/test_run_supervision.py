"""Fictional supervision cases; process tests require the lead's exclusive lease."""

import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from messaging_nlp import run_supervision as supervision
from messaging_nlp.run_contracts import ResourceLimits, ResourceSample, WorkerRequest

BOOT = "darwin-kern-boottime:100:0"
NEXT_BOOT = "darwin-kern-boottime:200:0"


def sample(**updates):
    values = dict(
        monotonic_seconds=1,
        pressure_level=1,
        swap_used_bytes=0,
        free_disk_bytes=20 * 1024**3,
        tree_rss_bytes=0,
        tree_phys_footprint_bytes=0,
    )
    return ResourceSample(**(values | updates))


class FictionalSensors:
    def __init__(self):
        self.fail = False
        self.calls = []

    def sample(self, pid):
        self.calls.append(pid)
        if self.fail:
            raise OSError("fictional sensor failure")
        return sample(monotonic_seconds=time.monotonic())


@pytest.fixture(autouse=True)
def no_process_in_pure_cases(request, monkeypatch):
    if "test_process_" in request.node.name:
        return

    def forbidden(*args, **kwargs):
        raise AssertionError("pure test attempted a process or real sensor")

    monkeypatch.setattr(supervision.subprocess, "Popen", forbidden)
    monkeypatch.setattr(supervision, "_command", forbidden)
    monkeypatch.setattr(supervision, "_footprint", forbidden)
    monkeypatch.setattr(supervision.os, "killpg", forbidden)
    monkeypatch.setattr(supervision.os, "kill", forbidden)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "{ sec = 1700000000, usec = 123 } Tue Oct 27",
            "darwin-kern-boottime:1700000000:123",
        ),
        ("{ sec = 1700000000, usec = 0 }", "darwin-kern-boottime:1700000000:0"),
    ],
)
def test_boot_normalization(monkeypatch, raw, expected):
    monkeypatch.setattr(supervision, "_command", lambda args: raw)
    assert supervision.boot_identity() == expected


@pytest.mark.parametrize(
    "raw", ["", "garbage", "{ sec = 0, usec = 0 }", "{ sec = 1, usec = 1000000 }"]
)
def test_boot_rejects_unknown(monkeypatch, raw):
    monkeypatch.setattr(supervision, "_command", lambda args: raw)
    with pytest.raises(supervision.SupervisionError, match="boot"):
        supervision.boot_identity()


@pytest.mark.parametrize(
    "updates,warning_seconds,reason",
    [
        ({"tree_rss_bytes": 3758096385}, 0, "tree_rss_limit"),
        ({"tree_phys_footprint_bytes": 3758096385}, 0, "tree_phys_footprint_limit"),
        ({"swap_used_bytes": 268435457}, 0, "system_swap_growth_limit"),
        ({"free_disk_bytes": 16106127359}, 0, "free_disk_limit"),
        ({"pressure_level": 4}, 0, "critical_system_memory_pressure"),
        ({"pressure_level": 2}, 9.999, None),
        ({"pressure_level": 2}, 10, "sustained_system_memory_warning"),
        (
            {"tree_rss_bytes": 3000000000, "tree_phys_footprint_bytes": 3000000000},
            0,
            None,
        ),
        (
            {
                "tree_rss_bytes": 3758096384,
                "tree_phys_footprint_bytes": 3758096384,
                "swap_used_bytes": 268435456,
                "free_disk_bytes": 16106127360,
            },
            0,
            None,
        ),
    ],
)
def test_resource_limits_independent(updates, warning_seconds, reason):
    assert (
        supervision.check_sample(
            sample(**updates), sample(), ResourceLimits(), warning_seconds
        )
        == reason
    )


def test_swap_is_growth_not_absolute():
    assert (
        supervision.check_sample(
            sample(swap_used_bytes=10**9),
            sample(swap_used_bytes=10**9),
            ResourceLimits(),
        )
        is None
    )


def test_mutex_exclusive_stale_and_unresolved(tmp_path):
    path = tmp_path / "portable-mutex"
    first = supervision.ModelMutex(path, "first", BOOT)
    first.acquire()
    with pytest.raises(supervision.SupervisionError):
        supervision.ModelMutex(path, "second", NEXT_BOOT).acquire()
    first.release(cleanup_verified=False)
    assert path.exists()
    assert json.loads((path / "owner.json").read_text())["cleanup_verified"] is False
    with pytest.raises(supervision.SupervisionError):
        supervision.ModelMutex(path, "first", BOOT).acquire()


def test_mutex_release_only_owner(tmp_path):
    path = tmp_path / "mutex"
    owner = supervision.ModelMutex(path, "owner", BOOT)
    owner.acquire()
    impostor = supervision.ModelMutex(path, "owner", BOOT)
    with pytest.raises(supervision.SupervisionError):
        impostor.release()
    owner.release()
    assert not path.exists()


def test_mutex_replacement_preserved(tmp_path):
    path = tmp_path / "mutex"
    owner = supervision.ModelMutex(path, "owner", BOOT)
    owner.acquire()
    (path / "owner.json").unlink()
    (path / "owner.json").write_text('{"owner":"replacement"}')
    with pytest.raises(supervision.SupervisionError):
        owner.release()
    assert path.exists()


def test_baseline_failure_durable_before_launch(tmp_path):
    sensors = FictionalSensors()
    sensors.fail = True
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", sensors, ResourceLimits()
    )
    with pytest.raises(supervision.SupervisionError) as caught:
        child.start()
    assert caught.value.code == "sensor_error"
    receipt = json.loads((child.output / "supervision.json").read_text())
    assert receipt["failure"]["code"] == "sensor_error"
    assert child.close().verified
    for name in (
        "stdout.log",
        "stderr.log",
        "protocol.jsonl",
        "raw-fragments.jsonl",
        "resources.jsonl",
        "requests.jsonl",
        "warnings.jsonl",
    ):
        assert (child.output / name).exists()


def test_baseline_low_disk_prevents_launch(tmp_path):
    class LowDisk:
        def sample(self, pid):
            return sample(free_disk_bytes=0)

    child = supervision.ChildSession(
        ["never"], tmp_path / "out", LowDisk(), ResourceLimits()
    )
    with pytest.raises(supervision.SupervisionError) as caught:
        child.start()
    assert caught.value.code == "resource_stop"
    assert child.close().verified


def test_baseline_stuck_sensor_is_bounded_and_retained(tmp_path):
    import threading

    release, done = threading.Event(), threading.Event()

    class Stuck:
        def sample(self, pid):
            try:
                release.wait(5)
                return sample()
            finally:
                done.set()

    child = supervision.ChildSession(
        ["never-launch"], tmp_path / "stuck", Stuck(), ResourceLimits()
    )
    started = time.monotonic()
    try:
        with pytest.raises(supervision.SupervisionError) as caught:
            child.start()
        assert caught.value.code == "sensor_error"
        assert time.monotonic() - started < 3
        assert child.failure["code"] == "sensor_error"
    finally:
        release.set()
        assert done.wait(1)
        assert child.close().verified


def test_final_pending_sensor_failure_separate_from_verified_cleanup(tmp_path):
    import threading

    child = supervision.ChildSession([], tmp_path, FictionalSensors(), ResourceLimits())
    event = threading.Event()
    event.set()
    child._sample_pending = dict(
        done=event, result=None, error=RuntimeError("late sensor"), deadline=0
    )
    assert child.close().verified  # No child ever existed: cleanup is known.
    assert child.failure["code"] == "sensor_error"  # Not a successful supervised run.


def test_final_pending_resource_breach_is_not_successful_supervision(tmp_path):
    event = threading.Event()
    event.set()
    child = supervision.ChildSession([], tmp_path, FictionalSensors(), ResourceLimits())
    child.baseline = sample()
    child._sample_pending = dict(
        done=event, result=(sample(free_disk_bytes=0), []), error=None, deadline=0
    )
    assert child.close().verified
    assert child.failure["code"] == "resource_stop"


def test_launch_failure_retained(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("fictional launch failure")

    monkeypatch.setattr(supervision.subprocess, "Popen", fail)
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FictionalSensors(), ResourceLimits()
    )
    with pytest.raises(supervision.SupervisionError) as caught:
        child.start()
    assert caught.value.code == "launch_error"
    assert json.loads((child.output / "supervision.json").read_text())["failure"]
    assert child.close().verified


def test_fresh_logs_and_no_overwrite(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "stdout.log").write_bytes(b"keep")
    child = supervision.ChildSession(
        ["never"], output, FictionalSensors(), ResourceLimits()
    )
    with pytest.raises((FileExistsError, supervision.SupervisionError)):
        child.start()
    assert (output / "stdout.log").read_bytes() == b"keep"


def test_sensor_strict_parsing_and_separate_footprint(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision.sys, "platform", "darwin")

    def command(args):
        if args[0] == "/bin/ps":
            return (
                "100 1 100 10 Tue Oct 27 12:00:00 2026\n"
                "101 100 100 20 Tue Oct 27 12:00:01 2026\n"
                f"{supervision.os.getpid()} 1 1 40 Tue Oct 27 12:00:00 2026"
            )
        if args[-1] == "vm.swapusage":
            return "total = 1.00G used = 2.50M free = 0.00M (encrypted)"
        return "1"

    monkeypatch.setattr(supervision, "_command", command)
    monkeypatch.setattr(supervision, "_footprint", lambda pid: 7)
    monkeypatch.setattr(
        supervision.shutil, "disk_usage", lambda root: type("D", (), {"free": 10**12})()
    )
    result = supervision.MacSensors(tmp_path).sample(100)
    assert result.tree_rss_bytes == 30 * 1024
    assert result.tree_phys_footprint_bytes == 14
    assert result.swap_used_bytes == 2621440
    monkeypatch.setattr(supervision, "_command", lambda args: "invalid")
    with pytest.raises(supervision.SupervisionError):
        supervision.MacSensors(tmp_path).sample(100)


def test_sensor_footprint_failure_not_rss_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision.sys, "platform", "darwin")
    monkeypatch.setattr(
        supervision, "_command", lambda args: "100 1 100 10 Tue Oct 27 12:00:00 2026"
    )

    def fail(pid):
        raise PermissionError("libproc denied")

    monkeypatch.setattr(supervision, "_footprint", fail)
    with pytest.raises(supervision.SupervisionError):
        supervision.MacSensors(tmp_path).sample(100)


def test_protocol_and_fragments_fsync_before_delivery(tmp_path, monkeypatch):
    sensors = FictionalSensors()
    sensors.fail = True
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", sensors, ResourceLimits()
    )
    with pytest.raises(supervision.SupervisionError):
        child.start()
    child._active = WorkerRequest(kind="generate", row_id="r")
    syncs = []
    original = supervision.os.fsync

    def record(fd):
        syncs.append(fd)
        return original(fd)

    monkeypatch.setattr(supervision.os, "fsync", record)
    data = (
        b'{"protocol":"nlp3-qwen-serial-v1","kind":"generation_started","row_id":"r","payload":{}}\n'
        b'{"protocol":"nlp3-qwen-serial-v1","kind":"fragment","row_id":"r","payload":{"text":"partial"}}\n'
    )
    child._append("protocol.jsonl", data)
    child._parse_protocol(data)
    assert len(syncs) == 2
    assert "partial" in (child.output / "raw-fragments.jsonl").read_text()
    assert (child.output / "protocol.jsonl").read_bytes() == data
    assert child.close().verified


@pytest.mark.parametrize(
    "wire",
    [
        b'{"kind":"fragment","row_id":"other","payload":{"text":"bad"}}\n',
        b'{"kind":"fragment","row_id":"r","payload":{"text":"before start"}}\n',
        b"not-json\n",
        b'{"kind":"terminal","row_id":"r","payload":{"finish_reason":"stop"}}\n',
    ],
)
def test_protocol_rejects_unbound_or_malformed_events(tmp_path, wire, monkeypatch):
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FictionalSensors(), ResourceLimits()
    )
    child._active = WorkerRequest(kind="generate", row_id="r")
    monkeypatch.setattr(child, "_append", lambda name, data: None)
    wire = wire.replace(b'{"kind":', b'{"protocol":"nlp3-qwen-serial-v1","kind":')
    with pytest.raises(supervision.SupervisionError):
        child._parse_protocol(wire)


def test_protocol_requires_explicit_version(tmp_path):
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FictionalSensors(), ResourceLimits()
    )
    child._active = WorkerRequest(kind="load")
    with pytest.raises(supervision.SupervisionError, match="explicit protocol"):
        child._parse_protocol(b'{"kind":"ready","payload":{}}\n')


def test_wait_deadline_independent_of_stuck_sensor(tmp_path, monkeypatch):
    release = threading.Event()

    class StuckSensors:
        def sample(self, pid):
            release.wait(2)
            return sample()

    child = supervision.ChildSession(
        ["never"], tmp_path / "out", StuckSensors(), ResourceLimits()
    )
    child.process = type("Process", (), {"pid": 123, "poll": lambda self: None})()
    monkeypatch.setattr(supervision, "_process_rows", lambda: [])
    monkeypatch.setattr(child, "_pump", lambda timeout: time.sleep(max(0, timeout)))

    def fail(code, message):
        raise supervision.SupervisionError(code, message)

    monkeypatch.setattr(child, "_fail", fail)
    started = time.monotonic()
    try:
        with pytest.raises(supervision.SupervisionError) as caught:
            child.wait_event(started + 0.04)
        assert caught.value.code == "timeout"
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        assert child._sample_pending["done"].wait(1), (
            "mock-only sampler must finish before patches restore"
        )


def test_cleanup_failure_is_unverified_and_mutex_stays(tmp_path, monkeypatch):
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FictionalSensors(), ResourceLimits()
    )
    child.process = type(
        "Process",
        (),
        {"pid": 123, "stdin": None, "stdout": None, "stderr": None, "returncode": None},
    )()

    def fail():
        raise OSError("cannot verify process identities")

    monkeypatch.setattr(supervision, "_process_rows", fail)
    receipt = child.close()
    assert not receipt.verified
    assert "unresolved" in receipt.detail
    mutex = supervision.ModelMutex(tmp_path / "mutex", "owner", BOOT).acquire()
    mutex.release(cleanup_verified=receipt.verified)
    with pytest.raises(supervision.SupervisionError):
        supervision.ModelMutex(tmp_path / "mutex", "later", NEXT_BOOT).acquire()


def test_permission_error_never_proves_absence_of_live_owned_group(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    child = supervision.ChildSession(
        [], tmp_path, FictionalSensors(), SimpleNamespace(cleanup_seconds=0.02)
    )
    child.process = SimpleNamespace(
        pid=123,
        stdin=None,
        stdout=None,
        stderr=None,
        returncode=None,
        poll=lambda: None,
    )
    monkeypatch.setattr(
        supervision,
        "_process_rows",
        lambda: [supervision._ProcessRow(123, 1, 123, 1, "same-birth")],
    )

    def denied(*args):
        raise PermissionError("fictional signal denial")

    monkeypatch.setattr(supervision.os, "killpg", denied)
    monkeypatch.setattr(
        child, "_pump", lambda timeout, **kwargs: time.sleep(min(timeout, 0.002))
    )
    assert child.close().verified is False
    assert child._signal_errors


# Strict owned-absence evidence used by the real parent-loss test. Production
# already retains EPERM and verifies absence independently; these deterministic
# controls launch no process and pin the same test-side contract: a signal error
# is evidence to retain, never proof of absence. Absence requires strict /bin/ps
# observations of the owned pgid and the originally observed (pid, birth)
# descendant identities. Unresolved ps output, live owned rows and a group that
# disappears then reappears all fail closed.

BIRTH = "Tue Oct 27 12:00:00 2026"


class FakeMonotonicClock:
    """Deterministic clock for bounded loops; sleeping advances monotonic time."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


OWNED_ABSENCE_SECONDS = 8.0


class OwnedAbsenceError(AssertionError):
    """Strict test-side failure to prove an owned group is absent."""

    def __init__(self, message, *, signal_errors=()):
        super().__init__(message)
        self.signal_errors = tuple(signal_errors)


@dataclass(frozen=True)
class OwnedAbsence:
    """Evidence behind a declared absence; signal errors are retained, not erased."""

    pgid: int
    absence: str
    attempts: int
    observed: tuple
    last_owned: tuple
    signal_errors: tuple
    reap_results: tuple


def _capture_owned_identities(pgid, *, rows=None):
    """Snapshot the owned leader and group members by (pid, birth) identity."""
    try:
        table = (rows or supervision._process_rows)()
    except Exception as error:
        raise OwnedAbsenceError(f"process table unresolved: {error}") from error
    leader = next((row for row in table if row.pid == pgid), None)
    if leader is None or leader.group != pgid:
        raise OwnedAbsenceError(
            f"owned group leader {pgid} was not observed; refusing to signal"
        )
    owned = {row.pid for row in table if row.group == pgid}
    while True:
        expanded = owned | {row.pid for row in table if row.parent in owned}
        if expanded == owned:
            break
        owned = expanded
    return {row.pid: row.birth for row in table if row.pid in owned}


def _check_owned_absence(
    *,
    pgid,
    observed,
    timeout=OWNED_ABSENCE_SECONDS,
    interval=0.05,
    required_clean_reads=2,
    rows=None,
    signal_group=None,
    reap=None,
    clock=None,
    sleep=None,
    signal_errors=(),
):
    """Bounded, /bin/ps-backed proof that an owned group is gone.

    Absence is never inferred from EPERM: each declaration needs
    ``required_clean_reads`` consecutive strict ``_process_rows`` observations in
    which neither the owned pgid nor any originally observed (pid, birth)
    descendant appears. ``PermissionError`` from the group signal is retained as
    evidence and verification continues; ``ProcessLookupError`` is the
    definitive-gone race case. A live owned row, a group that disappears then
    reappears, an owned pgid with no matching observed identity (possible pid
    reuse) or an unresolved process table raises ``OwnedAbsenceError`` (never a
    pass). SIGKILL is sent only to ``pgid`` after a member identity captured at
    observation time still matches; escaped descendants are verified by identity
    and never signalled blindly.
    """
    rows_fn = rows or supervision._process_rows
    signal_fn = signal_group or supervision.os.killpg
    clock_fn = clock or time.monotonic
    sleep_fn = sleep or time.sleep
    if pgid not in observed:
        raise OwnedAbsenceError(
            f"no observed leader identity for pgid {pgid}; refusing to verify"
        )
    observed = dict(observed)
    if type(required_clean_reads) is not int or required_clean_reads < 1:
        raise ValueError("required_clean_reads must be a positive integer")
    errors = list(signal_errors)
    reaps = []
    attempts = 0
    clean_reads = 0
    group_absent_seen = False
    last_owned = ()
    deadline = clock_fn() + timeout
    while True:
        attempts += 1
        if reap is not None:
            try:
                reaps.append(reap())
            except Exception as error:
                raise OwnedAbsenceError(
                    f"reap unresolved after {attempts} attempt(s): {error}",
                    signal_errors=errors,
                ) from error
        try:
            table = rows_fn()
        except Exception as error:
            raise OwnedAbsenceError(
                f"process table unresolved after {attempts} observation(s): {error}",
                signal_errors=errors,
            ) from error
        group_members = [row for row in table if row.group == pgid]
        if group_members and not any(
            observed.get(row.pid) == row.birth for row in group_members
        ):
            raise OwnedAbsenceError(
                "owned pgid present without an observed identity; refusing unrelated signals",
                signal_errors=errors,
            )
        identities = {row.pid for row in table if observed.get(row.pid) == row.birth}
        identities.update(row.pid for row in group_members)
        while True:
            expanded = identities | {
                row.pid for row in table if row.parent in identities
            }
            if expanded == identities:
                break
            identities = expanded
        observed.update((row.pid, row.birth) for row in table if row.pid in identities)
        owned = [row for row in table if row.pid in identities]
        last_owned = tuple((row.pid, row.birth) for row in owned)
        group_rows = [row for row in owned if row.group == pgid]
        if not group_rows:
            group_absent_seen = True
        if not owned:
            clean_reads += 1
            if clean_reads >= required_clean_reads:
                return OwnedAbsence(
                    pgid=pgid,
                    absence="process_table",
                    attempts=attempts,
                    observed=tuple(sorted(observed.items())),
                    last_owned=(),
                    signal_errors=tuple(errors),
                    reap_results=tuple(reaps),
                )
        else:
            clean_reads = 0
            if group_rows:
                if group_absent_seen:
                    raise OwnedAbsenceError(
                        "owned group disappeared then reappeared; refusing to "
                        f"signal unrelated processes: {last_owned}",
                        signal_errors=errors,
                    )
                if not any(observed.get(row.pid) == row.birth for row in group_rows):
                    raise OwnedAbsenceError(
                        "owned pgid present without an observed identity; "
                        f"refusing to signal unrelated processes: {last_owned}",
                        signal_errors=errors,
                    )
                try:
                    signal_fn(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The group vanished between observation and signal.
                except PermissionError as error:
                    # Retain, never pass: absence still needs clean ps reads.
                    errors.append(
                        {
                            "pgid": pgid,
                            "signal": int(signal.SIGKILL),
                            "error": str(error),
                        }
                    )
        if clock_fn() >= deadline:
            raise OwnedAbsenceError(
                "owned group/observed descendants still present after "
                f"{timeout}s: owned={last_owned}; retained signal errors={errors}",
                signal_errors=errors,
            )
        sleep_fn(min(interval, max(0.0, deadline - clock_fn())))


def _verify_owned_group_gone(
    pgid, observed, *, timeout=OWNED_ABSENCE_SECONDS, **kwargs
):
    try:
        return _check_owned_absence(
            pgid=pgid, observed=observed, timeout=timeout, **kwargs
        )
    except OwnedAbsenceError as error:
        pytest.fail(str(error), pytrace=False)


def test_capture_owned_identities_includes_escaped_descendant_tree():
    table = [
        supervision._ProcessRow(4321, 1, 4321, 1, BIRTH),
        supervision._ProcessRow(4322, 4321, 999, 1, BIRTH),
        supervision._ProcessRow(4323, 4322, 998, 1, BIRTH),
    ]
    assert _capture_owned_identities(4321, rows=lambda: table) == {
        4321: BIRTH,
        4322: BIRTH,
        4323: BIRTH,
    }


def test_owned_absence_tracks_observed_descendant_after_reparenting():
    leader = supervision._ProcessRow(4321, 1, 4321, 1, BIRTH)
    child = supervision._ProcessRow(4322, 4321, 999, 1, BIRTH)
    escaped = supervision._ProcessRow(4322, 1, 999, 1, BIRTH)
    table = [[leader, child], [escaped]]
    clock = FakeMonotonicClock()
    with pytest.raises(OwnedAbsenceError, match="still present"):
        _check_owned_absence(
            pgid=4321,
            observed={4321: BIRTH},
            timeout=0.2,
            rows=lambda: table.pop(0) if table else [escaped],
            signal_group=lambda pgid, sig: None,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )


def test_owned_absence_eperm_without_group_needs_clean_observations(monkeypatch):
    leader = supervision._ProcessRow(4321, 1, 4321, 1, BIRTH)
    table = [[leader], [], []]

    def rows():
        return table.pop(0) if table else []

    def denied(group, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(supervision, "_process_rows", rows)
    monkeypatch.setattr(supervision.os, "killpg", denied)
    clock = FakeMonotonicClock()
    result = _check_owned_absence(
        pgid=4321,
        observed={4321: BIRTH},
        clock=clock.monotonic,
        sleep=clock.sleep,
        reap=lambda: 0,
    )
    # Success comes only from two consecutive clean ps reads; the EPERM stays
    # retained as evidence and reaps remain recorded.
    assert result.absence == "process_table"
    assert result.attempts == 3
    assert result.last_owned == ()
    assert len(result.signal_errors) == 1
    assert "Operation not permitted" in result.signal_errors[0]["error"]
    assert result.reap_results == (0, 0, 0)


def test_owned_absence_eperm_live_group_fails_closed(monkeypatch):
    leader = supervision._ProcessRow(4321, 1, 4321, 1, BIRTH)
    member = supervision._ProcessRow(4322, 4321, 4321, 1, BIRTH)
    monkeypatch.setattr(supervision, "_process_rows", lambda: [leader, member])

    def denied(group, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(supervision.os, "killpg", denied)
    clock = FakeMonotonicClock()
    with pytest.raises(OwnedAbsenceError, match="still present") as caught:
        _check_owned_absence(
            pgid=4321,
            observed={4321: BIRTH},
            timeout=0.2,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
    assert "Operation not permitted" in str(caught.value)
    assert caught.value.signal_errors


def test_owned_absence_eperm_unresolved_table_fails_closed(monkeypatch):
    leader = supervision._ProcessRow(4321, 1, 4321, 1, BIRTH)
    calls = {"count": 0}

    def rows():
        calls["count"] += 1
        if calls["count"] == 1:
            return [leader]
        raise ValueError("truncated ps output")

    def denied(group, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(supervision, "_process_rows", rows)
    monkeypatch.setattr(supervision.os, "killpg", denied)
    clock = FakeMonotonicClock()
    with pytest.raises(OwnedAbsenceError, match="unresolved") as caught:
        _check_owned_absence(
            pgid=4321,
            observed={4321: BIRTH},
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
    assert "truncated ps output" in str(caught.value)


def test_owned_group_survives_leader_exit_and_pid_replacement_not_signalled(tmp_path):
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FictionalSensors(), ResourceLimits()
    )
    child.process = type("Process", (), {"pid": 100})()
    row = supervision._ProcessRow
    child._observe([row(100, 1, 100, 1, "leader"), row(101, 100, 100, 1, "child")])
    assert [r.pid for r in child._observe([row(101, 1, 100, 1, "child")])] == [101]
    assert child._observe([row(101, 1, 999, 1, "replacement")]) == []
    assert child._group_absent


def test_mutex_context_exception_retains_unresolved(tmp_path):
    path = tmp_path / "mutex"
    with pytest.raises(ValueError):
        with supervision.ModelMutex(path, "owner", BOOT):
            raise ValueError("fictional failure")
    assert json.loads((path / "owner.json").read_text())["cleanup_verified"] is False


def test_optional_runtimes_not_imported():
    assert not {"torch", "mlx", "transformers", "tokenizers"}.intersection(sys.modules)


def test_launch_pipe_failure_is_durable(tmp_path, monkeypatch):
    def fail():
        raise OSError("no file descriptors")

    monkeypatch.setattr(supervision.os, "pipe", fail)
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FictionalSensors(), ResourceLimits()
    )
    with pytest.raises(supervision.SupervisionError) as caught:
        child.start()
    assert caught.value.code == "launch_error"
    assert json.loads((child.output / "supervision.json").read_text())["failure"]
    assert child.close().verified


def test_interrupted_launch_is_unresolved_not_claimed_never_launched(
    tmp_path, monkeypatch
):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(supervision.subprocess, "Popen", interrupt)
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FictionalSensors(), ResourceLimits()
    )
    with pytest.raises(supervision.SupervisionError) as caught:
        child.start()
    assert caught.value.code == "interrupted"
    assert not child.close().verified
    assert (
        json.loads((child.output / "supervision.json").read_text())["cleanup"][
            "verified"
        ]
        is False
    )


FIXTURE = Path(__file__).parent / "fixtures" / "fake_qwen_child.py"


def start_fake(tmp_path, mode="success", sensors=None):
    child = supervision.ChildSession(
        [sys.executable, str(FIXTURE), mode],
        tmp_path / "child",
        sensors or FictionalSensors(),
        ResourceLimits(),
    )
    child.start()
    child.send(WorkerRequest(kind="load"))
    assert child.wait_event(time.monotonic() + 5).kind == "ready"
    return child


def events_for(child, row="r", timeout=3):
    child.send(WorkerRequest(kind="generate", row_id=row))
    events = []
    deadline = time.monotonic() + timeout
    while True:
        event = child.wait_event(deadline)
        events.append(event)
        if event.kind in ("terminal", "error"):
            return events


def test_process_persistent_success_durable(tmp_path):
    child = start_fake(tmp_path)
    try:
        pid = child.process.pid
        for row in ("first", "second"):
            assert events_for(child, row)[-1].payload["finish_reason"] == "stop"
            assert child.process.pid == pid
            assert row in (child.output / "raw-fragments.jsonl").read_text()
        assert "incidental stdout" in (child.output / "stdout.log").read_text()
        assert "fictional stderr" in (child.output / "stderr.log").read_text()
        assert "incidental" not in (child.output / "protocol.jsonl").read_text()
    finally:
        assert child.close().verified


@pytest.mark.parametrize("mode", ["partial", "stall", "crash"])
def test_process_partial_crash_prefill_deadline(tmp_path, mode):
    child = start_fake(tmp_path, mode)
    try:
        started = time.monotonic()
        with pytest.raises(supervision.SupervisionError) as caught:
            events_for(child, timeout=0.35)
        assert caught.value.code in ("timeout", "runtime_error")
        assert time.monotonic() - started < 3
        if mode != "stall":
            assert "fictional" in (child.output / "raw-fragments.jsonl").read_text()
        assert json.loads((child.output / "supervision.json").read_text())["failure"]
    finally:
        assert child.close().verified


@pytest.mark.parametrize(
    "mode,finish", [("invalid_json", "stop"), ("length", "length")]
)
def test_process_no_output_repair(tmp_path, mode, finish):
    child = start_fake(tmp_path, mode)
    try:
        events = events_for(child)
        assert events[-1].payload["finish_reason"] == finish
        text = next(e.payload["text"] for e in events if e.kind == "fragment")
        assert text == ("not JSON" if mode == "invalid_json" else '{"fictional":true}')
    finally:
        assert child.close().verified


def test_process_live_sensor_failure(tmp_path):
    sensors = FictionalSensors()
    child = start_fake(tmp_path, "stall", sensors)
    try:
        sensors.fail = True
        with pytest.raises(supervision.SupervisionError) as caught:
            events_for(child)
        assert caught.value.code == "sensor_error"
    finally:
        assert child.close().verified


def test_process_live_resource_stop(tmp_path):
    sensors = FictionalSensors()
    child = start_fake(tmp_path, "stall", sensors)
    try:
        sensors.sample = lambda pid: sample(
            free_disk_bytes=0, monotonic_seconds=time.monotonic()
        )
        with pytest.raises(supervision.SupervisionError) as caught:
            events_for(child)
        assert caught.value.code == "resource_stop"
        assert (child.output / "resources.jsonl").read_text().count(
            "free_disk_bytes"
        ) >= 2
    finally:
        assert child.close().verified


def test_process_interruption_cleans_up(tmp_path, monkeypatch):
    child = start_fake(tmp_path, "stall")
    try:
        child.send(WorkerRequest(kind="generate", row_id="r"))
        original = child._pump

        def interrupt_once(*args, **kwargs):
            monkeypatch.setattr(child, "_pump", original)
            raise KeyboardInterrupt()

        monkeypatch.setattr(child, "_pump", interrupt_once)
        with pytest.raises(supervision.SupervisionError) as caught:
            child.wait_event(time.monotonic() + 3)
        assert caught.value.code == "interrupted"
        assert child.close().verified
    finally:
        child.close()


def test_process_load_error(tmp_path):
    child = supervision.ChildSession(
        [sys.executable, str(FIXTURE), "load_error"],
        tmp_path / "child",
        FictionalSensors(),
        ResourceLimits(),
    )
    try:
        child.start()
        child.send(WorkerRequest(kind="load"))
        assert child.wait_event(time.monotonic() + 5).kind == "error"
    finally:
        assert child.close().verified


def test_process_ready_mode(tmp_path):
    child = start_fake(tmp_path, "ready")
    assert child.close().verified


def test_process_blocked_stdin_uses_parent_dispatch_deadline(tmp_path):
    child = supervision.ChildSession(
        [sys.executable, str(FIXTURE), "blocked_stdin"],
        tmp_path / "child",
        FictionalSensors(),
        ResourceLimits(),
    )
    child.start()
    started = time.monotonic()
    try:
        with pytest.raises(supervision.SupervisionError) as caught:
            child.send(
                WorkerRequest(kind="load", payload={"fictional": "x" * 200000}),
                deadline=started + 0.3,
            )
        assert caught.value.code == "timeout"
        assert time.monotonic() - started < 3
        assert (child.output / "requests.jsonl").stat().st_size > 200000
    finally:
        assert child.close().verified


def test_process_descendant_after_leader_exit_unrelated_survives(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child = start_fake(tmp_path, "descendant")
    try:
        with pytest.raises(supervision.SupervisionError):
            events_for(child)
        receipt = child.close()
        assert receipt.verified
        assert receipt.sigkill_sent or receipt.sigterm_sent
        assert unrelated.poll() is None
    finally:
        child.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_process_parent_loss_lifeline(tmp_path):
    # A separate fictional coordinator is killed, not this pytest process.
    script = """
import json,sys,time
from pathlib import Path
from messaging_nlp.run_contracts import ResourceLimits, ResourceSample, WorkerRequest
from messaging_nlp.run_supervision import ChildSession
class Sensors:
 def sample(self,pid):
  return ResourceSample(monotonic_seconds=time.monotonic(), pressure_level=1,
    swap_used_bytes=0,free_disk_bytes=10**12,tree_rss_bytes=0,tree_phys_footprint_bytes=0)
c=ChildSession([sys.executable,sys.argv[1],"parent_loss"],Path(sys.argv[2]),Sensors(),ResourceLimits())
c.start()
c.send(WorkerRequest(kind="load"))
c.wait_event(time.monotonic()+5)
Path(sys.argv[3]).write_text(str(c.process.pid))
time.sleep(60)
"""
    pidfile = tmp_path / "pid"
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(FIXTURE),
            str(tmp_path / "child"),
            str(pidfile),
        ]
    )
    pgid = None
    observed = {}
    try:
        deadline = time.monotonic() + 8
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists()
        pgid = int(pidfile.read_text())
        # Capture the owned group and every observed descendant identity while
        # the isolated session is still alive; later checks match on birth too.
        observed = _capture_owned_identities(pgid)
        parent.kill()
        parent.wait(timeout=5)
        # Raw killpg is not evidence: macOS can return EPERM while an exited
        # group leader awaits reap. Absence must come from strict, repeated
        # /bin/ps observations of the owned pgid/observed identities.
        evidence = _verify_owned_group_gone(pgid, observed, timeout=8)
        assert evidence.absence == "process_table"
        assert evidence.last_owned == ()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if pgid is not None and observed:
            # Retains EPERM, never passes on it, re-verifies independently.
            _verify_owned_group_gone(pgid, observed, timeout=8)
