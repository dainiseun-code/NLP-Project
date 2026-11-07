"""Durable, model-free supervision of one isolated serial worker.

The caller owns authorization, budget and the portable ModelMutex. This is a
sampled stop policy, not an OS memory sandbox. RSS and physical footprint are
independent observations, never additive. No optional model runtime is imported.
"""

from __future__ import annotations

import ctypes
import errno
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from messaging_nlp.io import canonical_json, loads_json
from messaging_nlp.limits import (
    CALL_TIMEOUT_SECONDS,
    MAX_PROTOCOL_QUEUE_EVENTS,
    MEMORY_WARNING_SECONDS,
)
from messaging_nlp.run_contracts import (
    PROTOCOL_VERSION,
    CleanupReceipt,
    ResourceLimits,
    ResourceSample,
    WorkerEvent,
    WorkerRequest,
)


class SupervisionError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _command(args):
    return subprocess.check_output(
        args, text=True, timeout=1, env=dict(os.environ, LC_ALL="C")
    ).strip()


def boot_identity() -> str:
    """Normalize macOS kern.boottime, refusing absent/ambiguous identities."""
    try:
        raw = _command(["/usr/sbin/sysctl", "-n", "kern.boottime"])
        match = re.fullmatch(
            r"\{\s*sec\s*=\s*([0-9]+),\s*usec\s*=\s*([0-9]+)\s*\}(?:\s+[^{}]*)?",
            raw,
        )
        if match is None:
            raise ValueError("unrecognized kern.boottime")
        seconds, micros = map(int, match.groups())
        if seconds <= 0 or not 0 <= micros < 1_000_000:
            raise ValueError("invalid kern.boottime")
        return f"darwin-kern-boottime:{seconds}:{micros}"
    except Exception as error:
        raise SupervisionError("sensor_error", f"boot identity: {error}") from error


class _RusageInfoV0(ctypes.Structure):
    # Public <sys/resource.h> RUSAGE_INFO_V0; retained independently of archive.
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
        )
    ]


@lru_cache(maxsize=1)
def _rusage_function():
    if sys.platform != "darwin":
        raise OSError(errno.ENOSYS, "physical footprint requires macOS")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


def _footprint(pid):
    info = _RusageInfoV0()
    ctypes.set_errno(0)
    if _rusage_function()(pid, 0, ctypes.byref(info)) != 0:
        code = ctypes.get_errno() or errno.EIO
        if code == errno.ESRCH:  # An observed process exited between samples.
            return 0
        raise OSError(code, f"proc_pid_rusage({pid}): {os.strerror(code)}")
    return int(info.ri_phys_footprint)


@dataclass(frozen=True)
class _ProcessRow:
    pid: int
    parent: int
    group: int
    rss_kib: int
    birth: str


def _process_rows():
    output = _command(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,lstart="])
    rows = []
    seen = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 9:
            raise ValueError("unrecognized ps process identity/RSS output")
        pid, parent, group, rss = map(int, parts[:4])
        birth = " ".join(parts[4:])
        if (
            pid < 0
            or min(parent, group, rss) < 0
            or pid in seen
            or not re.fullmatch(
                r"[A-Z][a-z]{2} [A-Z][a-z]{2} [0-9]{1,2} "
                r"[0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4}",
                birth,
            )
        ):
            raise ValueError("invalid ps process identity/RSS")
        seen.add(pid)
        rows.append(_ProcessRow(pid, parent, group, rss, birth))
    if not rows:
        raise ValueError("empty process table")
    return rows


def _tree(rows, pid):
    selected = {r.pid for r in rows if r.pid == pid or r.group == pid}
    while True:
        expanded = selected | {r.pid for r in rows if r.parent in selected}
        if selected == expanded:
            return [r for r in rows if r.pid in selected]
        selected = expanded


class MacSensors:
    """Strict system and owned-tree sensors; pid=0 is a system-only baseline."""

    def __init__(self, root):
        self.root = Path(root)

    def sample(self, pid) -> ResourceSample:
        try:
            if sys.platform != "darwin":
                raise OSError("MacSensors requires macOS")
            if type(pid) is not int or pid < 0:
                raise ValueError("sample requires a nonnegative pid")
            table = _process_rows()
            tree_rows = _tree(table, pid) if pid else []
            rss = sum(r.rss_kib * 1024 for r in tree_rows)
            physical = sum(_footprint(r.pid) for r in tree_rows)
            coordinator = next((r for r in table if r.pid == os.getpid()), None)
            if coordinator is None:
                raise ValueError("coordinator absent from process sensor")
            coordinator_rss = coordinator.rss_kib * 1024
            coordinator_footprint = _footprint(os.getpid())
            pressure = int(
                _command(
                    [
                        "/usr/sbin/sysctl",
                        "-n",
                        "kern.memorystatus_vm_pressure_level",
                    ]
                )
            )
            if pressure not in (1, 2, 4):
                raise ValueError("unknown memory pressure level")
            swap = _command(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
            matches = re.findall(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGTP])\b", swap)
            if len(matches) != 1:
                raise ValueError("unrecognized macOS swap use")
            amount, unit = matches[0]
            swap_bytes = int(float(amount) * 1024 ** ("KMGTP".index(unit) + 1))
            return ResourceSample(
                monotonic_seconds=time.monotonic(),
                pressure_level=pressure,
                swap_used_bytes=swap_bytes,
                free_disk_bytes=shutil.disk_usage(self.root).free,
                tree_rss_bytes=rss,
                tree_phys_footprint_bytes=physical,
                coordinator_rss_bytes=coordinator_rss,
                coordinator_phys_footprint_bytes=coordinator_footprint,
            )
        except Exception as error:
            raise SupervisionError(
                "sensor_error", f"resource sensor: {error}"
            ) from error


def check_sample(sample, baseline, limits, warning_seconds=0) -> str | None:
    if sample.pressure_level >= 4:
        return "critical_system_memory_pressure"
    if sample.pressure_level not in (1, 2):
        return "unknown_system_memory_pressure"
    warning_threshold = (
        limits.memory_warning_seconds
        if limits.memory_warning_seconds is not None
        else MEMORY_WARNING_SECONDS
    )
    if sample.pressure_level == 2 and warning_seconds >= warning_threshold:
        return "sustained_system_memory_warning"
    if sample.swap_used_bytes - baseline.swap_used_bytes > limits.max_swap_growth_bytes:
        return "system_swap_growth_limit"
    if sample.free_disk_bytes < limits.min_free_disk_bytes:
        return "free_disk_limit"
    if sample.tree_rss_bytes > limits.max_tree_rss_bytes:
        return "tree_rss_limit"
    if sample.tree_phys_footprint_bytes > limits.max_tree_phys_footprint_bytes:
        return "tree_phys_footprint_limit"
    return None


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _no_symlinks(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise SupervisionError("runtime_error", f"symlink in owned path: {part}")
    return path


def _identity(path):
    stat = path.stat(follow_symlinks=False)
    return stat.st_dev, stat.st_ino


def _json_bytes(value):
    return canonical_json(value) + b"\n"


def _replace_json(path, value):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


class ModelMutex:
    """Exclusive portable directory lock; no stale-lock reclamation of any kind.

    A false cleanup verification leaves an unresolved owner record. A new object,
    even with the same owner string, cannot release an existing lock.
    """

    def __init__(self, path, owner, boot):
        self.path = Path(path)
        if (
            not isinstance(owner, str)
            or not owner.strip()
            or not isinstance(boot, str)
            or not boot.strip()
        ):
            raise ValueError("mutex requires owner and boot identity")
        if not re.fullmatch(
            r"darwin-kern-boottime:[1-9][0-9]*:(?:0|[1-9][0-9]{0,5})", boot
        ):
            raise ValueError("mutex requires a canonical macOS boot identity")
        if ".git" in self.path.parts:
            raise ValueError("model mutex must be portable, outside .git")
        self.owner = owner
        self.boot = boot
        self._token = uuid.uuid4().hex
        self._directory = None
        self._file = None

    def acquire(self):
        if self._directory is not None:
            raise SupervisionError("runtime_error", "mutex already acquired")
        self.path = _no_symlinks(self.path)
        try:
            self.path.mkdir(mode=0o700)
        except FileExistsError as error:
            raise SupervisionError(
                "runtime_error",
                "model mutex exists; stale/unresolved ownership requires explicit reconciliation",
            ) from error
        self._directory = _identity(self.path)
        _fsync_directory(self.path.parent)
        # On storage failure, the directory intentionally remains blocking.
        with (self.path / "owner.json").open("xb") as handle:
            handle.write(
                _json_bytes(
                    dict(
                        owner=self.owner,
                        boot=self.boot,
                        pid=os.getpid(),
                        token=self._token,
                        cleanup_verified=None,
                    )
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        self._file = _identity(self.path / "owner.json")
        _fsync_directory(self.path)
        return self

    def _assert_owner(self):
        try:
            _no_symlinks(self.path)
            record = loads_json((self.path / "owner.json").read_text(encoding="utf-8"))
            if (
                self._directory is None
                or _identity(self.path) != self._directory
                or _identity(self.path / "owner.json") != self._file
                or record.get("token") != self._token
                or record.get("owner") != self.owner
                or record.get("boot") != self.boot
                or record.get("pid") != os.getpid()
            ):
                raise ValueError("mutex ownership changed or was never acquired")
            return record
        except Exception as error:
            raise SupervisionError(
                "runtime_error", f"refusing mutex release: {error}"
            ) from error

    def release(self, cleanup_verified=True):
        record = self._assert_owner()
        if cleanup_verified is not True:
            record["cleanup_verified"] = False
            _replace_json(self.path / "owner.json", record)
            self._file = _identity(self.path / "owner.json")
            self._directory = None  # Even this object cannot later quietly clear it.
            return
        (self.path / "owner.json").unlink()
        _fsync_directory(self.path)
        self.path.rmdir()
        _fsync_directory(self.path.parent)
        self._directory = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback):
        if self._directory is not None:
            self.release(cleanup_verified=exc_type is None)


def watch_parent(lifeline_fd, protocol_fd=None):
    """Start a daemon watcher before loading anything; EOF kills our own group.

    A pipe is portable to macOS (unlike Linux PR_SET_PDEATHSIG). Only the parent
    owns a write end. No protocol write on parent death can block this fail-safe.
    The unused optional protocol fd is accepted for the worker's common API.
    """
    group = os.getpgrp()
    if group != os.getpid() or type(lifeline_fd) is not int or lifeline_fd < 0:
        raise SupervisionError(
            "runtime_error",
            "parent watcher requires an isolated group leader and lifeline",
        )
    os.fstat(lifeline_fd)

    def watch():
        try:
            while True:
                try:
                    if not os.read(lifeline_fd, 1):
                        break
                except InterruptedError:
                    continue
        finally:
            # Never signal the coordinator's or another process's group.
            if os.getpgrp() == group and os.getpid() == group:
                os.killpg(group, signal.SIGKILL)

    thread = threading.Thread(target=watch, name="nlp3-parent-lifeline", daemon=True)
    thread.start()
    return thread


class ChildSession:
    """One persistent child, serial stdin requests and dedicated protocol pipe.

    wait_event takes an absolute time.monotonic deadline. Every received byte and
    resource observation is fsynced before an event is returned. send is bounded
    and services logs/resources while writing, so blocked stdin is not unguarded.
    close verifies the entire owned group even when the leader already exited.
    """

    _LOGS = (
        "stdout.log",
        "stderr.log",
        "protocol.jsonl",
        "raw-fragments.jsonl",
        "resources.jsonl",
        "requests.jsonl",
        "warnings.jsonl",
    )

    def __init__(self, command, output, sensors, limits: ResourceLimits, *, env=None):
        self.command = list(command)
        self.output = Path(output)
        self.sensors = sensors
        self.limits = limits
        self.env = dict(env or {})
        self.process = None
        self.baseline = None
        self._selector = None
        self._logs = {}
        self._events = deque()
        self._buffer = bytearray()
        self._lifeline = None
        self._protocol = None
        self._started = False
        self._closed = False
        self._closing = False
        self._cleanup = None
        self._failure = None
        self._receipt_inode = None
        self._active = None
        self._request_deadline = None
        self._next_sample = 0
        self._warning_since = None
        self._last_sample_monotonic = None
        self._latest_sample = None
        self._known = {}
        self._group_absent = False
        self._ownership_error = None
        self._generation_started = False
        self._protocol_terminal = False
        self._launch_unresolved = False
        self._sample_pending = None
        self._protocol_eof = False
        self._signal_errors = []

    @property
    def returncode(self):
        return self.process.returncode if self.process is not None else None

    @property
    def failure(self):
        return dict(self._failure) if self._failure else None

    @property
    def latest_sample(self):
        """Most recent observed ResourceSample (baseline, then current)."""
        return self._latest_sample

    def _append(self, name, data):
        handle = self._logs[name]
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

    def _snapshot(self):
        if self._receipt_inode is None:
            return
        target = self.output / "supervision.json"
        if _identity(target) != self._receipt_inode:
            raise SupervisionError(
                "runtime_error", "supervision receipt ownership changed"
            )
        value = dict(
            command=self.command,
            status="failed"
            if self._failure
            else (
                "closed" if self._closed else "running" if self.process else "preflight"
            ),
            baseline=self.baseline.model_dump() if self.baseline else None,
            latest_sample=self._latest_sample.model_dump()
            if self._latest_sample
            else None,
            warning_since=self._warning_since,
            pid=self.process.pid if self.process else None,
            returncode=self.returncode,
            failure=self._failure,
            cleanup=self._cleanup.model_dump() if self._cleanup else None,
            active_request=self._active.model_dump() if self._active else None,
            sensor_sample_pending=self._sample_pending is not None,
            signal_errors=self._signal_errors,
            observed_processes=[
                dict(pid=pid, birth=birth) for pid, birth in self._known.items()
            ],
            physical_footprint_sensor="libproc.proc_pid_rusage:RUSAGE_INFO_V0",
            interpretation="sampled RSS and physical footprint are separate, not additive; system swap/pressure include other applications",
        )
        _replace_json(target, value)
        self._receipt_inode = _identity(target)

    def _fail(self, code, message):
        self._failure = self._failure or dict(
            code=code, message=str(message), monotonic_seconds=time.monotonic()
        )
        try:
            self._snapshot()
        finally:
            if self.process is not None and not self._closing:
                self.close()
        raise SupervisionError(code, str(message))

    def start(self):
        if self._started or self._closed:
            raise SupervisionError("runtime_error", "session cannot be started twice")
        self._started = True
        self.output = _no_symlinks(self.output)
        if not self.output.exists():
            self.output.mkdir(mode=0o700)
            _fsync_directory(self.output.parent)
        # Check every owned destination before creating any of them.
        if any(
            (self.output / name).exists() or (self.output / name).is_symlink()
            for name in (*self._LOGS, "supervision.json")
        ):
            raise FileExistsError("refusing to overwrite supervision evidence")
        for name in self._LOGS:
            self._logs[name] = (self.output / name).open("xb")
            self._logs[name].flush()
            os.fsync(self._logs[name].fileno())
        with (self.output / "supervision.json").open("xb") as handle:
            handle.write(b"{}\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._receipt_inode = _identity(self.output / "supervision.json")
        _fsync_directory(self.output)
        self._snapshot()
        try:
            pending = dict(done=threading.Event(), sample=None, error=None)

            def baseline_sample():
                try:
                    pending["sample"] = self.sensors.sample(0)
                except BaseException as error:
                    pending["error"] = error
                finally:
                    pending["done"].set()

            threading.Thread(
                target=baseline_sample, daemon=True, name="nlp3-baseline-sample"
            ).start()
            if not pending["done"].wait(2):
                raise TimeoutError("baseline sensor exceeded two-second deadline")
            if pending["error"] is not None:
                raise pending["error"]
            self.baseline = pending["sample"]
            self._latest_sample = self.baseline
            self._last_sample_monotonic = self.baseline.monotonic_seconds
            self._append("resources.jsonl", _json_bytes(self.baseline.model_dump()))
        except BaseException as error:
            self._fail(
                "sensor_error" if isinstance(error, Exception) else "interrupted", error
            )
        reason = check_sample(self.baseline, self.baseline, self.limits)
        if reason or self.baseline.pressure_level != 1:
            self._fail("resource_stop", reason or "abnormal_baseline_memory_pressure")
        self._snapshot()
        created_fds = []
        try:
            protocol_read, protocol_write = os.pipe()
            created_fds.extend((protocol_read, protocol_write))
            life_read, life_write = os.pipe()
            created_fds.extend((life_read, life_write))
        except BaseException as error:
            for fd in created_fds:
                os.close(fd)
            self._fail(
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "launch_error",
                error,
            )
        self._lifeline = life_write
        self._protocol = protocol_read
        try:
            child_env = dict(os.environ, **self.env)
            child_env.update(
                NLP3_PROTOCOL_FD=str(protocol_write),
                NLP3_LIFELINE_FD=str(life_read),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                PYTHONUNBUFFERED="1",
            )
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                bufsize=0,
                close_fds=True,
                pass_fds=(protocol_write, life_read),
                start_new_session=True,
            )
        except BaseException as error:
            os.close(protocol_read)
            os.close(life_write)
            self._protocol = self._lifeline = None
            self._launch_unresolved = not isinstance(error, Exception)
            self._fail(
                "interrupted" if self._launch_unresolved else "launch_error", error
            )
        finally:
            os.close(protocol_write)
            os.close(life_read)
        try:
            self._selector = selectors.DefaultSelector()
            for stream, name in (
                (self.process.stdout, "stdout.log"),
                (self.process.stderr, "stderr.log"),
                (protocol_read, "protocol.jsonl"),
            ):
                fd = stream if isinstance(stream, int) else stream.fileno()
                os.set_blocking(fd, False)
                self._selector.register(fd, selectors.EVENT_READ, name)
            os.set_blocking(self.process.stdin.fileno(), False)
            self._observe(_process_rows())
            self._snapshot()
        except BaseException as error:
            self._fail(
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "sensor_error",
                error,
            )
        return self

    def _observe(self, rows):
        """Track group and observed escaped descendants with birth identities."""
        group = self.process.pid
        members = [r for r in rows if r.group == group]
        if members and self._group_absent:
            self._ownership_error = (
                "original group disappeared then reappeared; refusing unrelated signals"
            )
            return []
        if not members:
            self._group_absent = True
        leader = next((r for r in rows if r.pid == group), None)
        if leader and group in self._known and self._known[group] != leader.birth:
            self._ownership_error = (
                "leader PID identity changed; refusing unrelated signals"
            )
            return []
        owned = {r.pid for r in rows if self._known.get(r.pid) == r.birth}
        owned.update(r.pid for r in members)
        while True:
            expanded = owned | {r.pid for r in rows if r.parent in owned}
            if expanded == owned:
                break
            owned = expanded
        current = [r for r in rows if r.pid in owned]
        self._known.update((r.pid, r.birth) for r in current)
        return current

    def _sample(self):
        # Sensor subprocesses are bounded individually, but even a stuck native
        # sensor must not block this selector's generation deadline. A daemon
        # thread only collects data; persistence and policy remain on the parent
        # selector thread. At most one sample is outstanding; never retry it.
        now = time.monotonic()
        if self._sample_pending is None:
            if now < self._next_sample:
                return
            pending = dict(
                done=threading.Event(), result=None, error=None, deadline=now + 2
            )
            self._sample_pending = pending

            def collect():
                try:
                    pending["result"] = (
                        self.sensors.sample(self.process.pid),
                        _process_rows(),
                    )
                except BaseException as error:
                    pending["error"] = error
                finally:
                    pending["done"].set()

            threading.Thread(
                target=collect, name="nlp3-resource-sample", daemon=True
            ).start()
            return
        pending = self._sample_pending
        if not pending["done"].is_set():
            if now >= pending["deadline"]:
                self._fail(
                    "sensor_error",
                    "resource sample exceeded two-second sensor deadline",
                )
            return
        self._sample_pending = None
        if pending["error"] is not None:
            self._fail("sensor_error", pending["error"])
        current, rows = pending["result"]
        self._append("resources.jsonl", _json_bytes(current.model_dump()))
        self._observe(rows)
        if self._ownership_error:
            self._fail("runtime_error", self._ownership_error)
        self._latest_sample = current
        try:
            warning_seconds = self._track_warning(current)
        except SupervisionError as error:
            self._fail(error.code, error)
        reason = check_sample(current, self.baseline, self.limits, warning_seconds)
        self._next_sample = time.monotonic() + self.limits.sample_interval_seconds
        if reason:
            if reason == "sustained_system_memory_warning":
                self._record_warning_stop(current, warning_seconds)
            self._fail("resource_stop", reason)

    def _warning_log(self, record):
        """Durably append a warning record once the warnings log exists."""
        if "warnings.jsonl" in self._logs:
            self._append("warnings.jsonl", _json_bytes(record))

    def _track_warning(self, current: ResourceSample) -> float:
        """Track continuous sampled warning pressure with durable records.

        Returns accumulated warning seconds on the sample's own monotonic clock.
        Level 2 accumulates; normal pressure resets the timer and records a
        recovery with its duration. Monotonic time must never move backwards
        while a warning is being tracked (fails closed).
        """
        warning_since = self._warning_since
        if (
            self._last_sample_monotonic is not None
            and current.monotonic_seconds < self._last_sample_monotonic
        ):
            raise SupervisionError(
                "sensor_error", "resource sample monotonic time moved backwards"
            )
        self._last_sample_monotonic = current.monotonic_seconds
        if warning_since is not None and current.monotonic_seconds < warning_since:
            raise SupervisionError(
                "sensor_error",
                "resource sample monotonic time moved backwards during warning",
            )
        if current.pressure_level == 2:
            if warning_since is None:
                self._warning_since = current.monotonic_seconds
                self._warning_log(
                    {
                        "event": "warning_start",
                        "monotonic_seconds": current.monotonic_seconds,
                        "pressure_level": current.pressure_level,
                    }
                )
            return current.monotonic_seconds - self._warning_since
        if current.pressure_level == 1 and warning_since is not None:
            self._warning_log(
                {
                    "event": "warning_recovery",
                    "monotonic_seconds": current.monotonic_seconds,
                    "warning_seconds": current.monotonic_seconds - warning_since,
                    "warning_start_seconds": warning_since,
                    "pressure_level": current.pressure_level,
                }
            )
            self._warning_since = None
        return 0.0

    def _record_warning_stop(self, current: ResourceSample, warning_seconds: float):
        self._warning_log(
            {
                "event": "warning_stop",
                "monotonic_seconds": current.monotonic_seconds,
                "warning_seconds": warning_seconds,
                "warning_start_seconds": self._warning_since,
                "pressure_level": current.pressure_level,
            }
        )

    def _parse_protocol(self, data, *, validate=True):
        self._buffer.extend(data)
        if len(self._buffer) > 1024 * 1024:
            raise SupervisionError(
                "runtime_error", "protocol line buffer exceeded 1 MiB"
            )
        while b"\n" in self._buffer:
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            try:
                value = loads_json(line.decode("utf-8"))
                # Even a state-invalid fragment remains durable raw evidence.
                if (
                    isinstance(value, dict)
                    and value.get("kind") == "fragment"
                    and isinstance(value.get("payload"), dict)
                    and isinstance(value["payload"].get("text"), str)
                ):
                    self._append("raw-fragments.jsonl", line + b"\n")
                if not validate:
                    continue
                if (
                    not isinstance(value, dict)
                    or value.get("protocol") != PROTOCOL_VERSION
                ):
                    raise ValueError("explicit protocol version required")
                event = WorkerEvent.model_validate(value)
                if self._active is None:
                    raise ValueError("unsolicited worker event")
                if self._protocol_terminal:
                    raise ValueError("event after terminal worker event")
                if event.row_id != self._active.row_id:
                    raise ValueError("worker row identity mismatch")
                if self._active.kind == "load" and event.kind not in ("ready", "error"):
                    raise ValueError("unexpected load event")
                if self._active.kind == "generate" and event.kind == "ready":
                    raise ValueError("unexpected generation ready")
                if event.kind == "generation_started":
                    if self._generation_started:
                        raise ValueError("duplicate generation_started")
                    self._generation_started = True
                if event.kind == "fragment":
                    if not self._generation_started or not isinstance(
                        event.payload.get("text"), str
                    ):
                        raise ValueError("fragment requires text")
                if event.kind == "terminal":
                    payload = event.payload
                    if (
                        not self._generation_started
                        or payload.get("finish_reason") not in ("stop", "length")
                        or any(
                            type(payload.get(k)) is not int or payload[k] < 0
                            for k in ("generation_tokens", "input_tokens")
                        )
                        or type(payload.get("elapsed_seconds")) not in (int, float)
                        or not math.isfinite(payload["elapsed_seconds"])
                        or payload["elapsed_seconds"] < 0
                    ):
                        raise ValueError("invalid terminal generation evidence")
                if event.kind == "error" and not isinstance(
                    event.payload.get("message"), str
                ):
                    raise ValueError("error requires message")
                if event.kind in ("ready", "terminal", "error"):
                    self._protocol_terminal = True
                if len(self._events) >= MAX_PROTOCOL_QUEUE_EVENTS:
                    raise ValueError(
                        f"unconsumed event queue exceeded {MAX_PROTOCOL_QUEUE_EVENTS}"
                    )
                self._events.append(event)
            except Exception as error:
                if not validate:
                    continue  # Exact malformed bytes already fsynced to protocol.
                raise SupervisionError(
                    "runtime_error", f"invalid worker protocol: {error}"
                ) from error

    def _pump(self, timeout, *, parsing=True):
        if self._selector is None:
            time.sleep(min(timeout, 0.05))
            return
        for key, mask in self._selector.select(max(0, timeout)):
            if key.data == "stdin":
                continue
            try:
                data = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not data:
                self._selector.unregister(key.fd)
                if key.data == "protocol.jsonl":
                    self._protocol_eof = True
                continue
            self._append(key.data, data)
            if key.data == "protocol.jsonl":
                self._parse_protocol(data, validate=parsing)

    def send(self, request: WorkerRequest, *, deadline=None):
        if (
            self.process is None
            or self._closed
            or self._active is not None
            or self._events
            or self._failure
        ):
            raise SupervisionError(
                "runtime_error", "worker unavailable or request already active"
            )
        if not isinstance(request, WorkerRequest):
            raise TypeError("send requires WorkerRequest")
        self._active = request
        self._generation_started = False
        self._protocol_terminal = False
        self._request_deadline = min(
            time.monotonic() + CALL_TIMEOUT_SECONDS,
            deadline if deadline is not None else float("inf"),
        )
        if not math.isfinite(self._request_deadline):
            raise ValueError("dispatch requires a finite deadline")
        wire = _json_bytes(request.model_dump())
        self._append("requests.jsonl", wire)
        self._snapshot()
        data = memoryview(wire)
        fd = self.process.stdin.fileno()
        self._selector.register(fd, selectors.EVENT_WRITE, "stdin")
        try:
            while data:
                if time.monotonic() >= self._request_deadline:
                    self._fail("timeout", "worker stdin dispatch deadline")
                self._sample()
                try:
                    written = os.write(fd, data)
                    data = data[written:]
                except BlockingIOError:
                    self._pump(min(0.05, self._request_deadline - time.monotonic()))
        except BaseException as error:
            if isinstance(error, SupervisionError):
                raise
            self._fail(
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "runtime_error",
                error,
            )
        finally:
            if self._selector is not None:
                try:
                    self._selector.unregister(fd)
                except KeyError:
                    pass
        if request.kind == "shutdown":
            self._active = None

    def wait_event(self, deadline) -> WorkerEvent:
        if not isinstance(deadline, (float, int)) or not math.isfinite(deadline):
            raise ValueError("deadline must be an absolute finite monotonic time")
        if self.process is None or self._closed or self._failure:
            raise SupervisionError("runtime_error", "worker is not running")
        deadline = min(deadline, self._request_deadline or deadline)
        try:
            while True:
                callback = getattr(self, "on_tick", None)
                if callback is not None:
                    callback()
                now = time.monotonic()
                if now >= deadline:
                    self._fail(
                        "timeout", "worker event deadline (including blocked prefill)"
                    )
                self._sample()
                if time.monotonic() >= deadline:
                    self._fail("timeout", "worker deadline during resource sampling")
                if self._events:
                    event = self._events.popleft()
                    if event.kind in ("ready", "terminal", "error"):
                        self._active = None
                        self._request_deadline = None
                        self._snapshot()
                    return event
                if self._protocol_eof:
                    self._fail(
                        "runtime_error",
                        "worker protocol closed before a terminal event",
                    )
                sample_due = (
                    0.02
                    if self._sample_pending is not None
                    else max(0, self._next_sample - time.monotonic())
                )
                self._pump(min(deadline - time.monotonic(), sample_due, 0.1))
                # Drain pipes before reporting a leader crash; fragments were
                # fsynced even when the process exited between selector wakes.
                if not self._events and self.process.poll() is not None:
                    self._pump(0)
                    if not self._events:
                        self._fail(
                            "runtime_error",
                            f"worker exited {self.process.returncode} before terminal event",
                        )
        except BaseException as error:
            if isinstance(error, SupervisionError):
                if not self._failure:
                    self._fail(error.code, error)
                raise
            self._fail(
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "runtime_error",
                error,
            )

    def _signal_owned(self, owned, sig):
        group = self.process.pid
        sent = False
        if any(row.group == group for row in owned):
            try:
                os.killpg(group, sig)
                sent = True
            except ProcessLookupError:
                pass
            except PermissionError as error:
                # macOS can return EPERM while a just-exited group leader is
                # awaiting reap. This is NOT proof of cleanup: continue bounded
                # absence verification, retaining the error. Live survivors still
                # make close() unverified and keep the model mutex blocking.
                self._signal_errors.append(
                    {"group": group, "signal": int(sig), "error": str(error)}
                )
                self.process.poll()
        for row in owned:
            if row.group != group:
                # These were observed as descendants, then left the group.
                # Recheck identity immediately before a targeted signal.
                current = next((r for r in _process_rows() if r.pid == row.pid), None)
                if current and current.birth == row.birth:
                    try:
                        os.kill(row.pid, sig)
                        sent = True
                    except ProcessLookupError:
                        pass
                    except PermissionError as error:
                        self._signal_errors.append(
                            {"pid": row.pid, "signal": int(sig), "error": str(error)}
                        )
        return sent

    def close(self) -> CleanupReceipt:
        if self._cleanup is not None:
            return self._cleanup
        self._closing = True
        cleanup_deadline = time.monotonic() + self.limits.cleanup_seconds
        verified = self.process is None and not self._launch_unresolved
        term = kill = False
        detail = (
            "no child launched"
            if verified
            else (
                "launch interrupted; child existence unresolved"
                if self._launch_unresolved
                else ""
            )
        )
        try:
            if self.process is not None:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
                self.process.poll()
                deadline = cleanup_deadline
                kill_at = time.monotonic() + self.limits.cleanup_seconds / 2
                owned = self._observe(_process_rows())
                if self._ownership_error:
                    raise RuntimeError(self._ownership_error)
                term = self._signal_owned(owned, signal.SIGTERM)
                while True:
                    self._pump(0, parsing=False)
                    self.process.poll()  # Reap the leader, but never equate that with cleanup.
                    owned = self._observe(_process_rows())
                    if self._ownership_error:
                        raise RuntimeError(self._ownership_error)
                    if not owned:
                        verified = self.process.returncode is not None
                        detail = "owned group and observed descendants absent"
                        break
                    now = time.monotonic()
                    if now >= deadline:
                        detail = "owned processes remain after cleanup deadline"
                        break
                    if now >= kill_at:
                        kill = self._signal_owned(owned, signal.SIGKILL) or kill
                    self._pump(min(0.05, deadline - now), parsing=False)
                self._pump(0, parsing=False)
        except BaseException as error:
            detail = f"cleanup unresolved: {type(error).__name__}: {error}"
            verified = False
        finally:
            # A completed model row does not discard an in-flight observation.
            # Settle it within the existing cleanup allocation, without allowing
            # a stuck sensor to extend that allocation or block the lifeline.
            if self._lifeline is not None:
                os.close(self._lifeline)
                self._lifeline = None
            pending = self._sample_pending
            if pending is not None:
                remaining = max(
                    0, min(cleanup_deadline, pending["deadline"]) - time.monotonic()
                )
                pending["done"].wait(remaining)
                if not pending["done"].is_set() and self._failure is None:
                    self._failure = dict(
                        code="sensor_error",
                        message="final resource sample unresolved within sensor/cleanup deadline",
                        monotonic_seconds=time.monotonic(),
                    )
            if pending is not None and pending["done"].is_set():
                try:
                    if pending["result"] is not None and self._logs:
                        self._append(
                            "resources.jsonl",
                            _json_bytes(pending["result"][0].model_dump()),
                        )
                    if pending["result"] is not None and self.baseline is not None:
                        current = pending["result"][0]
                        self._latest_sample = current
                        warning_seconds = self._track_warning(current)
                        reason = check_sample(
                            current, self.baseline, self.limits, warning_seconds
                        )
                        if reason and self._failure is None:
                            if reason == "sustained_system_memory_warning":
                                self._record_warning_stop(current, warning_seconds)
                            self._failure = dict(
                                code="resource_stop",
                                message=reason,
                                monotonic_seconds=time.monotonic(),
                            )
                    if pending["error"] is not None and self._failure is None:
                        self._failure = dict(
                            code="sensor_error",
                            message=str(pending["error"]),
                            monotonic_seconds=time.monotonic(),
                        )
                except Exception as error:
                    self._failure = self._failure or dict(
                        code="runtime_error",
                        message=f"final resource evidence: {error}",
                        monotonic_seconds=time.monotonic(),
                    )
                self._sample_pending = None
            # Closing the lifeline also activates the child-side group fail-safe.
            # If verification failed, callers MUST retain ModelMutex ownership.
            if self._selector is not None:
                self._selector.close()
                self._selector = None
            if self._protocol is not None:
                os.close(self._protocol)
                self._protocol = None
            if self.process is not None:
                for stream in (self.process.stdout, self.process.stderr):
                    if stream is not None:
                        stream.close()
            self._closed = True
            self._cleanup = CleanupReceipt(
                verified=verified,
                process_group_id=self.process.pid if self.process else None,
                returncode=self.returncode,
                sigterm_sent=term,
                sigkill_sent=kill,
                detail=detail,
            )
            try:
                self._snapshot()
            finally:
                for handle in self._logs.values():
                    handle.close()
                self._logs.clear()
                self._closing = False
        return self._cleanup
