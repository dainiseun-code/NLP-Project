"""Fictional Stage6-revision supervision checks (software/fake-only).

Covers continuous sampled warning pressure (not the old 3-sample count), durable
warning start/recovery/stop records, coordinator RSS/footprint, the central
protocol event-queue bound and the ``latest_sample`` progress hook. No model,
runtime or tokenizer is imported and no real sensor/process is exercised.
"""

import json
import os
import threading

import pytest

from messaging_nlp import run_supervision as supervision
from messaging_nlp.limits import MAX_PROTOCOL_QUEUE_EVENTS
from messaging_nlp.run_contracts import ResourceLimits, ResourceSample, WorkerRequest


def sample(**updates):
    values = dict(
        monotonic_seconds=1.0,
        pressure_level=1,
        swap_used_bytes=0,
        free_disk_bytes=20 * 1024**3,
        tree_rss_bytes=0,
        tree_phys_footprint_bytes=0,
    )
    return ResourceSample(**(values | updates))


class FailingSensors:
    def sample(self, pid):
        raise OSError("fictional sensor failure")


class NormalSensors:
    def sample(self, pid):
        return sample()


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


def make_child(tmp_path):
    """A child with baseline set and no launched process (timing-only)."""
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FailingSensors(), ResourceLimits()
    )
    child.baseline = sample()
    return child


def started_child(tmp_path):
    """A child whose durable logs exist (start fails before any launch)."""
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FailingSensors(), ResourceLimits()
    )
    with pytest.raises(supervision.SupervisionError):
        child.start()
    child.baseline = sample()
    return child


def check(child, monotonic_seconds, pressure_level):
    current = sample(monotonic_seconds=monotonic_seconds, pressure_level=pressure_level)
    warning_seconds = child._track_warning(current)
    return (
        current,
        warning_seconds,
        supervision.check_sample(
            current, child.baseline, child.limits, warning_seconds
        ),
    )


def test_transient_warnings_under_ten_seconds_do_not_stop(tmp_path):
    child = make_child(tmp_path)
    # Three warning samples spanning 0.64s, then a normal sample: never 10s.
    for offset in (0.0, 0.32, 0.64):
        _, _, reason = check(child, offset, 2)
        assert reason is None
    _, warning_seconds, reason = check(child, 0.64, 1)
    assert warning_seconds == 0.0
    assert reason is None


def test_warning_boundary_9999_no_stop_and_10_stop(tmp_path):
    child = make_child(tmp_path)
    child._track_warning(sample(pressure_level=2, monotonic_seconds=0.0))
    _, warning_seconds, reason = check(child, 9.999, 2)
    assert warning_seconds == pytest.approx(9.999)
    assert reason is None
    _, warning_seconds, reason = check(child, 10.0, 2)
    assert warning_seconds == pytest.approx(10.0)
    assert reason == "sustained_system_memory_warning"


def test_warning_recovery_resets_accumulated_duration(tmp_path):
    child = make_child(tmp_path)
    assert child._track_warning(sample(pressure_level=2, monotonic_seconds=0.0)) == 0.0
    assert child._track_warning(sample(pressure_level=2, monotonic_seconds=5.0)) == 5.0
    # A normal sample resets the timer.
    assert child._track_warning(sample(pressure_level=1, monotonic_seconds=5.0)) == 0.0
    # A fresh warning restarts from zero, so 1s later it is still not 10s.
    assert child._track_warning(sample(pressure_level=2, monotonic_seconds=9.0)) == 0.0
    _, warning_seconds, reason = check(child, 10.0, 2)
    assert warning_seconds == 1.0
    assert reason is None


def test_critical_and_unknown_pressure_stop_immediately(tmp_path):
    child = make_child(tmp_path)
    assert (
        supervision.check_sample(
            sample(pressure_level=4), child.baseline, child.limits, 0.0
        )
        == "critical_system_memory_pressure"
    )
    # Critical/unknown stop even with a huge accumulated warning duration.
    assert (
        supervision.check_sample(
            sample(pressure_level=3), child.baseline, child.limits, 999.0
        )
        == "unknown_system_memory_pressure"
    )
    assert (
        supervision.check_sample(
            sample(pressure_level=0), child.baseline, child.limits, 999.0
        )
        == "unknown_system_memory_pressure"
    )


def test_monotonic_backwards_during_warning_fails_closed(tmp_path):
    child = make_child(tmp_path)
    child._track_warning(sample(pressure_level=2, monotonic_seconds=10.0))
    with pytest.raises(supervision.SupervisionError, match="backwards"):
        child._track_warning(sample(pressure_level=2, monotonic_seconds=9.0))


def test_clock_cannot_regress_between_warning_samples_above_onset(tmp_path):
    child = make_child(tmp_path)
    child._track_warning(sample(pressure_level=2, monotonic_seconds=10.0))
    child._track_warning(sample(pressure_level=2, monotonic_seconds=12.0))
    with pytest.raises(supervision.SupervisionError, match="backwards"):
        child._track_warning(sample(pressure_level=2, monotonic_seconds=11.0))


def test_missing_coordinator_sensor_is_not_fabricated_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision.sys, "platform", "darwin")
    monkeypatch.setattr(supervision, "_process_rows", lambda: [])
    with pytest.raises(supervision.SupervisionError, match="coordinator absent"):
        supervision.MacSensors(tmp_path).sample(0)


def test_durable_warning_start_recovery_stop_records(tmp_path):
    child = started_child(tmp_path)
    try:
        assert (
            child._track_warning(sample(pressure_level=2, monotonic_seconds=5.0)) == 0.0
        )
        assert (
            child._track_warning(sample(pressure_level=2, monotonic_seconds=7.0)) == 2.0
        )
        assert (
            child._track_warning(sample(pressure_level=1, monotonic_seconds=7.5)) == 0.0
        )
        assert (
            child._track_warning(sample(pressure_level=2, monotonic_seconds=10.0))
            == 0.0
        )
        warning_seconds = child._track_warning(
            sample(pressure_level=2, monotonic_seconds=20.0)
        )
        assert warning_seconds == 10.0
        current = sample(pressure_level=2, monotonic_seconds=20.0)
        assert (
            supervision.check_sample(
                current, child.baseline, child.limits, warning_seconds
            )
            == "sustained_system_memory_warning"
        )
        child._record_warning_stop(current, warning_seconds)
    finally:
        assert child.close().verified
    records = [
        json.loads(line)
        for line in (child.output / "warnings.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [r["event"] for r in records] == [
        "warning_start",
        "warning_recovery",
        "warning_start",
        "warning_stop",
    ]
    assert records[1]["warning_seconds"] == 2.5
    assert records[1]["warning_start_seconds"] == 5.0
    assert records[-1]["warning_seconds"] == 10.0
    assert records[-1]["warning_start_seconds"] == 10.0


def test_final_pending_sustained_warning_not_skipped(tmp_path):
    event = threading.Event()
    event.set()
    child = supervision.ChildSession([], tmp_path, FailingSensors(), ResourceLimits())
    child.baseline = sample()
    child._warning_since = 0.0
    child._sample_pending = dict(
        done=event,
        result=(sample(pressure_level=2, monotonic_seconds=10.0), []),
        error=None,
        deadline=0,
    )
    assert child.close().verified
    assert child.failure["code"] == "resource_stop"
    assert child.failure["message"] == "sustained_system_memory_warning"


def test_final_pending_normal_sample_is_not_flagged(tmp_path):
    event = threading.Event()
    event.set()
    child = supervision.ChildSession([], tmp_path, FailingSensors(), ResourceLimits())
    child.baseline = sample()
    final = sample(monotonic_seconds=2.0)
    child._sample_pending = dict(done=event, result=(final, []), error=None, deadline=0)
    assert child.close().verified
    assert child.failure is None
    assert child.latest_sample is final


def test_coordinator_rss_footprint_separate_from_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision.sys, "platform", "darwin")
    coordinator_pid = os.getpid()

    def command(args):
        if args[0] == "/bin/ps":
            return (
                f"{coordinator_pid} 1 1 40 Tue Oct 27 12:00:00 2026\n"
                "100 1 100 10 Tue Oct 27 12:00:00 2026\n"
                "101 100 100 20 Tue Oct 27 12:00:01 2026"
            )
        if args[-1] == "vm.swapusage":
            return "total = 1.00G used = 2.50M free = 0.00M (encrypted)"
        return "1"

    monkeypatch.setattr(supervision, "_command", command)
    footprints = {coordinator_pid: 5, 100: 7, 101: 7}
    monkeypatch.setattr(supervision, "_footprint", lambda pid: footprints[pid])
    monkeypatch.setattr(
        supervision.shutil,
        "disk_usage",
        lambda root: type("D", (), {"free": 10**12})(),
    )
    result = supervision.MacSensors(tmp_path).sample(100)
    assert result.tree_rss_bytes == 30 * 1024  # 100 + 101 only, coordinator excluded
    assert result.tree_phys_footprint_bytes == 14
    assert result.coordinator_rss_bytes == 40 * 1024
    assert result.coordinator_phys_footprint_bytes == 5


def test_baseline_includes_coordinator_but_not_worker_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision.sys, "platform", "darwin")
    coordinator_pid = os.getpid()

    def command(args):
        if args[0] == "/bin/ps":
            return (
                f"{coordinator_pid} 1 1 40 Tue Oct 27 12:00:00 2026\n"
                "100 1 100 10 Tue Oct 27 12:00:00 2026\n"
                "101 100 100 20 Tue Oct 27 12:00:01 2026"
            )
        if args[-1] == "vm.swapusage":
            return "total = 1.00G used = 2.50M free = 0.00M (encrypted)"
        return "1"

    monkeypatch.setattr(supervision, "_command", command)
    footprints = {coordinator_pid: 5, 100: 7, 101: 7}
    monkeypatch.setattr(supervision, "_footprint", lambda pid: footprints[pid])
    monkeypatch.setattr(
        supervision.shutil,
        "disk_usage",
        lambda root: type("D", (), {"free": 10**12})(),
    )
    result = supervision.MacSensors(tmp_path).sample(0)  # system-only baseline
    assert result.tree_rss_bytes == 0
    assert result.tree_phys_footprint_bytes == 0
    assert result.coordinator_rss_bytes == 40 * 1024
    assert result.coordinator_phys_footprint_bytes == 5


def test_latest_sample_is_baseline_before_any_current_sample(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("fictional launch failure")

    monkeypatch.setattr(supervision.subprocess, "Popen", fail)
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", NormalSensors(), ResourceLimits()
    )
    with pytest.raises(supervision.SupervisionError):
        child.start()
    assert child.latest_sample is child.baseline
    assert child.latest_sample.pressure_level == 1


def _protocol_line(kind, payload=None):
    value = {
        "protocol": "nlp3-qwen-serial-v1",
        "kind": kind,
        "row_id": "r",
        "payload": payload if payload is not None else {},
    }
    return (json.dumps(value) + "\n").encode()


def _queue_child(tmp_path, monkeypatch):
    child = supervision.ChildSession(
        ["never"], tmp_path / "out", FailingSensors(), ResourceLimits()
    )
    child._active = WorkerRequest(kind="generate", row_id="r")
    monkeypatch.setattr(child, "_append", lambda name, data: None)
    return child


def test_protocol_queue_bound_holds_realistic_fragment_burst(tmp_path, monkeypatch):
    child = _queue_child(tmp_path, monkeypatch)
    start = _protocol_line("generation_started")
    fragment = _protocol_line("fragment", {"text": "x"})
    terminal = _protocol_line(
        "terminal",
        {
            "finish_reason": "stop",
            "generation_tokens": 1024,
            "input_tokens": 10,
            "elapsed_seconds": 0.1,
        },
    )
    # 1024 fragments + generation_started + terminal = 1026 events fits.
    child._parse_protocol(start + fragment * 1024 + terminal)
    assert len(child._events) == 1026


def test_protocol_queue_bound_rejects_overflow(tmp_path, monkeypatch):
    child = _queue_child(tmp_path, monkeypatch)
    start = _protocol_line("generation_started")
    fragment = _protocol_line("fragment", {"text": "x"})
    # MAX_PROTOCOL_QUEUE_EVENTS == 1028: start + 1027 fragments exactly fills it.
    child._parse_protocol(start + fragment * 1027)
    assert len(child._events) == MAX_PROTOCOL_QUEUE_EVENTS
    with pytest.raises(supervision.SupervisionError, match="event queue"):
        child._parse_protocol(fragment)  # the 1029th event overflows
