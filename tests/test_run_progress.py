"""Requirement-first progress reporter checks: fictional state and real pipes.

No test imports a model framework/tokenizer, reads study data or launches a
process. Expected behavior is from docs/stage6-revision-contract.md. The
reporter is a software worker: heartbeats and durability are verified here, not
model quality.
"""

import builtins
import json
import os
import subprocess
import threading
import time

import pytest

from messaging_nlp.io import read_jsonl
from messaging_nlp.run_progress import (
    ProgressPersistenceError,
    RunProgress,
    format_progress_line,
)


@pytest.fixture(autouse=True)
def deny_processes_and_optional_models(monkeypatch):
    """Fail even if error handling swallows a forbidden attempt."""
    attempts = []
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "mlx",
            "mlx_lm",
            "transformers",
            "tokenizers",
            "safetensors",
        }:
            attempts.append(name)
            raise AssertionError(f"optional model import forbidden: {name}")
        return original_import(name, *args, **kwargs)

    def no_process(*args, **kwargs):
        attempts.append("process launch")
        raise AssertionError("progress reporter tests cannot launch processes")

    monkeypatch.setattr(builtins, "__import__", import_guard)
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, no_process)
    yield
    assert not attempts, f"forbidden attempts: {attempts}"


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def assert_returns_promptly(fn, timeout=2.0):
    result = {}

    def run():
        try:
            fn()
        except BaseException as error:  # pragma: no cover - only on failure
            result["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "console transport blocked on a full/closed sink"
    if "error" in result:
        raise result["error"]


def drain(pipe_fd):
    os.set_blocking(pipe_fd, True)
    data = b""
    while True:
        chunk = os.read(pipe_fd, 65536)
        if not chunk:
            break
        data += chunk
    return data.decode("utf-8")


def test_heartbeat_continues_during_blocked_load(tmp_path):
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=0.01, console_fd=write_fd) as progress:
        progress.update(phase="loading", run_id="r", planned_rows=369)
        progress.attach(out)
        time.sleep(0.15)  # blocked fake model load/prefill; reporter keeps beating
    os.close(write_fd)
    console = drain(read_fd)
    os.close(read_fd)
    lines = [line for line in console.splitlines() if line]
    heartbeats = [line for line in lines if "supervisor=alive" in line]
    assert len(heartbeats) >= 2
    assert all("phase=loading" in line for line in heartbeats)
    assert all("model=advancing" not in line for line in heartbeats)
    records = read_jsonl(out / "progress.jsonl")
    assert any(r.get("record_kind") == "heartbeat" for r in records)


def test_no_token_heartbeat_is_honestly_pending(tmp_path):
    clock = FakeClock()
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=10, clock=clock, console_fd=write_fd) as p:
        p.attach(out)
        p.update(
            phase="generating", row_ordinal=0, planned_rows=369, condition="baseline"
        )
        clock.advance(5)
        p.pulse()
    os.close(write_fd)
    console = drain(read_fd)
    os.close(read_fd)
    assert "supervisor=alive" in console
    assert "model=advancing" not in console
    assert "tokens=" not in console
    assert "elapsed=5.0s" in console


def test_evidenced_token_age_and_count_are_shown(tmp_path):
    clock = FakeClock()
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=10, clock=clock, console_fd=write_fd) as p:
        p.attach(out)
        clock.advance(10)
        p.update(
            phase="generating",
            row_ordinal=2,
            planned_rows=369,
            generation_tokens=12,
            token_limit=1024,
            last_token_monotonic=clock.now - 0.4,
        )
        p.pulse()
    os.close(write_fd)
    console = drain(read_fd)
    os.close(read_fd)
    assert "tokens=12/1024" in console
    assert "model=advancing" in console
    assert "last_token_age=0.4s" in console


def test_console_is_stderr_never_stdout(capfd, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=0.01, clock=FakeClock()) as progress:
        progress.attach(out)
        progress.update(phase="loading")
        progress.pulse()
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "supervisor=alive" in captured.err


def test_full_stderr_does_not_block_and_durable_log_still_written(tmp_path):
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    chunk = b"x" * 8192
    while True:
        try:
            written = os.write(write_fd, chunk)
            if written == 0:
                break
        except BlockingIOError:
            break
    out = tmp_path / "out"
    out.mkdir()
    clock = FakeClock()

    def emit_heartbeats():
        with RunProgress(interval_seconds=0.01, clock=clock, console_fd=write_fd) as p:
            p.attach(out)
            p.update(phase="generating")
            p.pulse()
            p.pulse()
            p.check()

    assert_returns_promptly(emit_heartbeats)
    os.close(write_fd)
    os.close(read_fd)
    records = read_jsonl(out / "progress.jsonl")
    assert sum(1 for r in records if r["record_kind"] == "heartbeat") >= 2


def test_closed_stderr_disables_console_with_diagnostic(tmp_path):
    read_fd, write_fd = os.pipe()
    os.close(read_fd)  # disconnected pipe: no reader remains
    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=10, clock=FakeClock(), console_fd=write_fd) as p:
        p.attach(out)
        p.update(phase="loading")
        p.pulse()
        assert p.console_diagnostic is not None
        p.check()  # console failure must not become a persistence failure
    os.close(write_fd)
    records = read_jsonl(out / "progress.jsonl")
    assert any(r.get("console_disabled") for r in records)


def test_attach_collision_preserves_existing_files(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    jsonl = out / "progress.jsonl"
    jsonl.write_bytes(b"previous evidence\n")
    with RunProgress(interval_seconds=10, clock=FakeClock(), console_fd=None) as p:
        with pytest.raises(FileExistsError):
            p.attach(out)
    assert jsonl.read_bytes() == b"previous evidence\n"


def test_durable_records_are_wellformed_append_only_and_final(tmp_path):
    clock = FakeClock()
    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=10, clock=clock, console_fd=None) as p:
        p.update(phase="validating", run_id="r", planned_rows=369)
        p.attach(out)
        p.update(phase="loading")
        p.update(phase="generating", row_ordinal=0, condition="baseline")
        clock.advance(1)
        p.update(generation_tokens=12, token_limit=1024, last_token_monotonic=clock.now)
        p.pulse()
    records = read_jsonl(out / "progress.jsonl")
    kinds = [r["record_kind"] for r in records]
    assert kinds[0] == "attach"
    assert kinds[-1] == "close"
    assert "update" in kinds and "heartbeat" in kinds
    assert all(r["supervisor"] == "alive" for r in records)
    assert not any(
        r["record_kind"] == "update" and r.get("phase") == "validating" for r in records
    ), "pre-attach updates must not be persisted as their own records"
    snapshot = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert snapshot["generation_tokens"] == 12
    assert snapshot["token_limit"] == 1024
    assert snapshot["record_kind"] == "close"


def test_shutdown_joins_reporter_and_stops_heartbeats(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    read_fd, write_fd = os.pipe()
    progress = RunProgress(
        interval_seconds=0.01, clock=FakeClock(), console_fd=write_fd
    )
    with progress:
        progress.attach(out)
        time.sleep(0.05)
    assert not progress.reporter.is_alive()
    size_after_close = (out / "progress.jsonl").stat().st_size
    time.sleep(0.05)
    assert (out / "progress.jsonl").stat().st_size == size_after_close
    with pytest.raises(RuntimeError):
        progress.update(phase="cleanup")
    with pytest.raises(RuntimeError):
        progress.pulse()
    os.close(write_fd)
    os.close(read_fd)


def test_persistence_failure_is_surfaced_by_check(monkeypatch, tmp_path):
    import messaging_nlp.run_progress as run_progress

    def boom(path, data):
        raise OSError("fictional disk failure")

    out = tmp_path / "out"
    out.mkdir()
    read_fd, write_fd = os.pipe()
    with RunProgress(interval_seconds=10, clock=FakeClock(), console_fd=write_fd) as p:
        p.attach(out)
        monkeypatch.setattr(run_progress, "append_bytes", boom)
        p.update(phase="loading")  # must not raise here
        with pytest.raises(ProgressPersistenceError):
            p.check()
    os.close(write_fd)
    os.close(read_fd)


def test_attach_failure_is_observable(monkeypatch, tmp_path):
    import messaging_nlp.run_progress as run_progress

    def boom(path, data):
        raise OSError("fictional attach failure")

    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=10, clock=FakeClock(), console_fd=None) as p:
        monkeypatch.setattr(run_progress, "write_new", boom)
        with pytest.raises(OSError):
            p.attach(out)
        with pytest.raises(ProgressPersistenceError):
            p.check()


def test_update_rejects_unknown_fields_and_invalid_values(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=10, clock=FakeClock(), console_fd=None) as p:
        p.attach(out)
        with pytest.raises(ValueError):
            p.update(not_a_field=1)
        with pytest.raises(ValueError):
            p.update(planned_rows=-1)
        with pytest.raises(ValueError):
            p.update(last_token_monotonic=float("nan"))
        with pytest.raises(ValueError):
            p.update(row_ordinal=1.5)
        # valid resource samples pass through and reach the durable record
        p.update(resource_sample={"free_disk_bytes": 123})
        assert p._snapshot("update")["resource_sample"] == {"free_disk_bytes": 123}


def test_no_durable_writes_before_attach(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with RunProgress(interval_seconds=10, clock=FakeClock(), console_fd=None) as p:
        p.update(phase="validating")
        p.pulse()
        assert not (out / "progress.jsonl").exists()
        assert not (out / "progress.json").exists()
        p.attach(out)
        assert (out / "progress.jsonl").exists()
        assert (out / "progress.json").exists()


def test_format_progress_line_is_pure_and_prose_free():
    snapshot = {
        "record_kind": "heartbeat",
        "phase": "loading",
        "elapsed_seconds": 5.0,
        "supervisor": "alive",
    }
    line = format_progress_line(snapshot)
    assert line == "progress phase=loading elapsed=5.0s supervisor=alive"
    assert format_progress_line(dict(snapshot)) == line


def test_blocking_pipe_is_disabled_without_changing_inherited_flags(tmp_path):
    read_fd, write_fd = os.pipe()
    try:
        assert os.get_blocking(write_fd)
        with RunProgress(console_fd=write_fd) as progress:
            progress.attach(tmp_path)
            progress.update(phase="loading")
            assert os.get_blocking(write_fd)
            assert "blocking stderr pipe" in progress.console_diagnostic
        assert os.get_blocking(write_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_real_tty_gets_independent_nonblocking_progress_and_stale_token_is_not_advance(
    tmp_path,
):
    import pty

    master, slave = pty.openpty()
    try:
        assert os.get_blocking(slave)
        clock = FakeClock()
        with RunProgress(clock=clock, console_fd=slave) as progress:
            assert os.get_blocking(slave)
            assert not os.get_blocking(progress._owned_console_fd)
            progress.attach(tmp_path)
            progress.update(
                phase="generating",
                row_ordinal=0,
                generation_tokens=1,
                last_token_monotonic=clock.now,
            )
            clock.advance(20)
            progress.pulse()
            assert progress._snapshot("test")["model_progress"] == "no_new_token"
        assert os.get_blocking(slave)
        os.set_blocking(master, False)
        assert b"row=1" in os.read(master, 8192)
    finally:
        os.close(master)
        os.close(slave)


def test_token_updates_are_coalesced_not_persisted_per_token(tmp_path):
    clock = FakeClock()
    with RunProgress(clock=clock, console_fd=None) as progress:
        progress.attach(tmp_path)
        progress.update(phase="generating")
        before = len(read_jsonl(tmp_path / "progress.jsonl"))
        for count in range(1, 1025):
            progress.update(generation_tokens=count, last_token_monotonic=clock.now)
        assert len(read_jsonl(tmp_path / "progress.jsonl")) == before
        clock.advance(10)
        progress.pulse()
        assert read_jsonl(tmp_path / "progress.jsonl")[-1]["generation_tokens"] == 1024


def test_blocking_socket_is_disabled_without_touching_its_flags(tmp_path):
    import socket

    left, right = socket.socketpair()
    try:
        assert os.get_blocking(left.fileno())
        with RunProgress(console_fd=left.fileno()) as progress:
            progress.attach(tmp_path)
            progress.update(phase="loading")
            assert "socket/device" in progress.console_diagnostic
        assert os.get_blocking(left.fileno())
    finally:
        left.close()
        right.close()
