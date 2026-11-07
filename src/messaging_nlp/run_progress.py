"""Self-contained, model-free run progress reporter (stderr + durable records).

This module owns live phase/row/heartbeat reporting for the serial Stage 6 run.
It imports no model runtime, tokenizer or study data, never writes to stdout and
never changes the caller's descriptor flags. The append-only ``progress.jsonl``
and the atomic ``progress.json`` snapshot are the authoritative record; the
stderr console is best-effort and drops under backpressure without blocking.

Primary integration (after :func:`messaging_nlp.runner.create_output`)::

    progress = RunProgress()
    with progress:
        progress.update(phase="validating", run_id=..., planned_rows=...)
        validated = load_and_validate(...)
        output = create_output(validated)
        progress.attach(output)
        ...
        progress.update(phase="generating", row_ordinal=..., condition=...)
        ...
        progress.update(generation_tokens=..., token_limit=...,
                        last_token_monotonic=...)
    progress.check()  # raises if any durable progress write failed

Thread liveness (a heartbeat) is never reported as model advancement. The model
is only shown as advancing when ``last_token_monotonic`` evidence exists, and
token counts/ages are only shown when actually evidenced.
"""

from __future__ import annotations

import math
import os
import select
import stat
import threading
import time
from pathlib import Path

from messaging_nlp.io import canonical_json
from messaging_nlp.limits import PROGRESS_INTERVAL_SECONDS
from messaging_nlp.run_storage import append_bytes, atomic_json, write_new

_KNOWN_FIELDS = frozenset(
    {
        "phase",
        "run_id",
        "planned_rows",
        "row_ordinal",
        "condition",
        "rows_finished",
        "technical_ok",
        "attempted_failures",
        "unattempted",
        "generation_tokens",
        "token_limit",
        "last_token_monotonic",
        "row_started_monotonic",
        "resource_sample",
        "rows_saved",
        "outcome",
        "stop_reason",
    }
)
_COUNTERS = frozenset(
    {
        "planned_rows",
        "rows_finished",
        "technical_ok",
        "attempted_failures",
        "unattempted",
        "generation_tokens",
        "token_limit",
        "rows_saved",
    }
)
_MONOTONIC_FIELDS = frozenset({"last_token_monotonic", "row_started_monotonic"})
_STRING_FIELDS = frozenset({"phase", "run_id", "condition", "outcome", "stop_reason"})


class ProgressPersistenceError(RuntimeError):
    """A durable progress write failed; the authoritative log may be incomplete."""


def format_progress_line(snapshot: dict) -> str:
    """Render one compact, prose-free console line from a snapshot dict.

    Pure and deterministic: it never fabricates a token count, a model claim or
    a progress figure that is absent from ``snapshot``.
    """
    fields = ["progress"]
    if snapshot.get("phase") is not None:
        fields.append(f"phase={snapshot['phase']}")
    if snapshot.get("run_id") is not None:
        fields.append(f"run_id={snapshot['run_id']}")
    if snapshot.get("row_ordinal") is not None:
        planned = snapshot.get("planned_rows")
        row = snapshot["row_ordinal"]
        fields.append(f"row={row + 1}" + (f"/{planned}" if planned is not None else ""))
    if snapshot.get("condition") is not None:
        fields.append(f"condition={snapshot['condition']}")
    counters = []
    for name, label in (
        ("rows_finished", "terminal"),
        ("technical_ok", "valid"),
        ("attempted_failures", "attempt_failed"),
        ("unattempted", "not_attempted"),
    ):
        if snapshot.get(name) is not None:
            counters.append(f"{label}={snapshot[name]}")
    if counters:
        fields.append("rows=" + ",".join(counters))
    if snapshot.get("outcome") is not None:
        fields.append(f"outcome={snapshot['outcome']}")
    if snapshot.get("generation_tokens") is not None:
        limit = snapshot.get("token_limit")
        fields.append(
            "tokens="
            + str(snapshot["generation_tokens"])
            + (f"/{limit}" if limit is not None else "")
        )
    if snapshot.get("last_token_age_seconds") is not None:
        fields.append("model=" + snapshot.get("model_progress", "token_observed"))
        fields.append(f"last_token_age={snapshot['last_token_age_seconds']:.1f}s")
    if snapshot.get("row_elapsed_seconds") is not None:
        fields.append(f"row_elapsed={snapshot['row_elapsed_seconds']:.1f}s")
    fields.append(f"elapsed={snapshot.get('elapsed_seconds', 0.0):.1f}s")
    fields.append("supervisor=alive")
    resource = snapshot.get("resource_sample")
    if resource:
        fields.append(f"pressure={resource.get('pressure_level', 'unknown')}")
    return " ".join(fields)


class RunProgress:
    """Thread-safe reporter for one serial run; stderr console + durable records.

    ``__init__`` only configures. ``__enter__`` starts the reporter thread and
    records the run start time; ``__exit__``/``close`` stop and join it, then
    finalize the durable record. The periodic heartbeat keeps beating through a
    blocked model load/prefill because it runs on its own daemon thread.

    Durability failures are recorded, never silently swallowed, and surface from
    :meth:`check`. A console failure only disables the console (with a snapshot
    diagnostic) and never obstructs inference or durability.
    """

    def __init__(
        self,
        interval_seconds=PROGRESS_INTERVAL_SECONDS,
        clock=time.monotonic,
        console_fd=2,
    ):
        if not isinstance(interval_seconds, (int, float)) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if console_fd is not None and (type(console_fd) is not int or console_fd < 0):
            raise ValueError("console_fd must be a nonnegative int or None")
        self._interval = interval_seconds
        self._clock = clock
        self._console_fd = console_fd
        self._console_enabled = console_fd is not None
        self._console_diagnostic = None
        self._state = {name: None for name in _KNOWN_FIELDS}
        self._state_lock = threading.Lock()
        self._emit_lock = threading.Lock()
        self._persist_lock = threading.Lock()
        self._publish_lock = threading.RLock()
        self._errors_lock = threading.Lock()
        self._stop = threading.Event()
        self._reporter = threading.Thread(
            target=self._reporter_loop, name="nlp3-progress-reporter", daemon=True
        )
        self._output = None
        self._persistence_errors = []
        self._started = self._clock()
        self._entered = False
        self._closed = False
        self._owned_console_fd = None
        self._last_emitted = float("-inf")
        self._last_token_evidence = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        if self._entered:
            raise RuntimeError("reporter already started")
        if self._closed:
            raise RuntimeError("reporter is closed")
        self._entered = True
        self._started = self._clock()
        self._prepare_console()
        self._reporter.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._reporter.is_alive():
            self._reporter.join()
        self._finalize()
        if self._owned_console_fd is not None:
            os.close(self._owned_console_fd)
            self._owned_console_fd = None

    @property
    def reporter(self):
        """The reporter thread (joined and stopped after :meth:`close`)."""
        return self._reporter

    @property
    def console_diagnostic(self):
        """Why the console was disabled, or None while it is still enabled."""
        return self._console_diagnostic

    @property
    def persistence_failures(self):
        with self._errors_lock:
            return list(self._persistence_errors)

    def check(self) -> None:
        """Raise the first durable progress persistence failure, if any."""
        with self._errors_lock:
            errors = list(self._persistence_errors)
        if errors:
            raise ProgressPersistenceError(
                f"{len(errors)} progress persistence failure(s); first "
                f"{type(errors[0]).__name__}: {errors[0]}"
            ) from errors[0]

    # -- state -------------------------------------------------------------

    def update(self, **fields) -> None:
        """Merge telemetry cheaply; emit phase/row changes, coalesce token updates.

        Unknown fields or invalid values raise immediately; persistence failures
        are recorded (see :meth:`check`) and never raised here.
        """
        if self._closed:
            raise RuntimeError("reporter is closed")
        with self._state_lock:
            changed = any(
                self._state.get(key) != value
                for key, value in fields.items()
                if key
                not in {"generation_tokens", "last_token_monotonic", "resource_sample"}
            )
        self._merge(fields)
        if not changed and self._clock() - self._last_emitted < self._interval:
            return
        with self._publish_lock:
            snapshot = self._snapshot("update")
            self._emit(snapshot)
            self._persist(snapshot)

    def pulse(self) -> None:
        """Run one heartbeat synchronously (test/force seam; also runs on a timer)."""
        if self._closed:
            raise RuntimeError("reporter is closed")
        self._heartbeat()

    def attach(self, output: Path) -> None:
        """Create the durable records after a fresh output directory exists.

        Refuses to overwrite existing ``progress.jsonl``/``progress.json`` and
        performs no durable writes before this point.
        """
        if self._closed:
            raise RuntimeError("reporter is closed")
        if self._output is not None:
            raise RuntimeError("reporter already attached")
        output = Path(output)
        jsonl = output / "progress.jsonl"
        snapshot_path = output / "progress.json"
        if jsonl.exists() or snapshot_path.exists():
            raise FileExistsError("progress evidence exists; refusing to overwrite")
        snapshot = self._snapshot("attach")
        try:
            write_new(jsonl, canonical_json(snapshot) + b"\n")
            atomic_json(snapshot_path, snapshot)
        except BaseException as error:
            self._record_persistence_error(error)
            raise
        self._output = output

    # -- validation --------------------------------------------------------

    def _merge(self, fields: dict) -> None:
        unknown = set(fields) - _KNOWN_FIELDS
        if unknown:
            raise ValueError(f"unknown progress field(s): {sorted(unknown)}")
        cleaned = {
            name: self._validate_field(name, value) for name, value in fields.items()
        }
        with self._state_lock:
            self._state.update(cleaned)

    def _validate_field(self, name: str, value):
        if value is None:
            return None
        if name in _COUNTERS or name == "row_ordinal":
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} requires a nonnegative int")
            return value
        if name in _MONOTONIC_FIELDS:
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} requires a nonnegative finite number")
            return value
        if name in _STRING_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"{name} requires a string")
            return value
        if name == "resource_sample":
            return self._validate_resource_sample(value)
        raise ValueError(f"unsupported field: {name}")  # pragma: no cover

    @staticmethod
    def _validate_resource_sample(value):
        if isinstance(value, dict):
            return value
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            value = dump()
            if not isinstance(value, dict):
                raise ValueError("resource_sample model_dump must return a dict")
            return value
        raise ValueError("resource_sample requires a dict or a model with model_dump()")

    # -- snapshot / emit / persist ----------------------------------------

    def _snapshot(self, record_kind: str) -> dict:
        with self._state_lock:
            state = dict(self._state)
        now = self._clock()
        snapshot = {
            "record_kind": record_kind,
            "monotonic_seconds": now,
            "supervisor": "alive",
        }
        for name, value in state.items():
            if value is not None:
                snapshot[name] = value
        started = self._started if self._started is not None else now
        snapshot["elapsed_seconds"] = max(0.0, now - started)
        if state.get("last_token_monotonic") is not None:
            snapshot["last_token_age_seconds"] = max(
                0.0, now - state["last_token_monotonic"]
            )
            evidence = (
                state.get("row_ordinal"),
                state.get("generation_tokens"),
                state["last_token_monotonic"],
            )
            snapshot["model_progress"] = (
                "stopped"
                if state.get("phase") in ("row_saved", "cleanup", "finalizing")
                else "advancing"
                if evidence != self._last_token_evidence
                and snapshot["last_token_age_seconds"] <= self._interval
                else "no_new_token"
            )
            self._last_token_evidence = evidence
        if state.get("row_started_monotonic") is not None:
            snapshot["row_elapsed_seconds"] = max(
                0.0, now - state["row_started_monotonic"]
            )
        if not self._console_enabled:
            snapshot["console_disabled"] = self._console_diagnostic
        return snapshot

    def _emit(self, snapshot: dict) -> None:
        self._last_emitted = self._clock()
        line = (format_progress_line(snapshot) + "\n").encode("utf-8")
        with self._emit_lock:
            self._console_emit(line)

    def _prepare_console(self):
        """Use an independently opened nonblocking TTY; never change inherited flags.

        A select-ready blocking pipe can become full between select and write.
        Refuse that unsafe sink rather than promising a nonexistent write bound.
        Redirected regular files have no consumer-backpressure wait; storage
        failures remain subject to the same filesystem limitations as all logs.
        """
        if not self._console_enabled:
            return
        try:
            fd = self._console_fd
            if os.isatty(fd):
                self._owned_console_fd = os.open(
                    os.ttyname(fd), os.O_WRONLY | os.O_NONBLOCK | os.O_NOCTTY
                )
                self._console_fd = self._owned_console_fd
            else:
                mode = os.fstat(fd).st_mode
                if not stat.S_ISREG(mode) and os.get_blocking(fd):
                    kind = "pipe" if stat.S_ISFIFO(mode) else "socket/device"
                    self._disable_console(
                        f"blocking stderr {kind} unsupported; use durable progress.jsonl"
                    )
        except (ValueError, OSError) as error:
            self._disable_console(f"console unavailable: {error}")

    def _console_emit(self, line: bytes) -> None:
        if not self._console_enabled:
            return
        fd = self._console_fd
        # Bound every write below POSIX's minimum PIPE_BUF. Never wait for a
        # consumer and never change the caller's descriptor flags.
        line = line[:511].rstrip(b"\n") + b"\n"
        try:
            readable, writable, exceptional = select.select([], [fd], [], 0)
        except (OSError, ValueError) as error:
            self._disable_console(f"console unavailable: {error}")
            return
        if writable:
            try:
                os.write(fd, line)
            except BlockingIOError:
                return
            except (BrokenPipeError, OSError) as error:
                self._disable_console(f"console closed: {error}")
            return
        if readable or exceptional:
            # Disconnected (read end closed) or errored sink: a write fails
            # immediately here instead of blocking, so use it to confirm.
            try:
                os.write(fd, line)
            except (BrokenPipeError, OSError) as error:
                self._disable_console(f"console disconnected: {error}")
            except BlockingIOError:
                return
        # Full/backpressured but open: drop this line; the durable log is
        # authoritative and the console must never block inference.

    def _disable_console(self, diagnostic: str) -> None:
        self._console_enabled = False
        self._console_diagnostic = diagnostic

    def _heartbeat(self) -> None:
        with self._publish_lock:
            snapshot = self._snapshot("heartbeat")
            self._emit(snapshot)
            self._persist(snapshot)

    def _persist(self, snapshot: dict) -> None:
        if self._output is None:
            return
        with self._persist_lock:
            try:
                append_bytes(
                    self._output / "progress.jsonl", canonical_json(snapshot) + b"\n"
                )
                atomic_json(self._output / "progress.json", snapshot)
            except BaseException as error:
                self._record_persistence_error(error)

    def _finalize(self) -> None:
        if self._output is None:
            return
        snapshot = self._snapshot("close")
        with self._persist_lock:
            try:
                append_bytes(
                    self._output / "progress.jsonl", canonical_json(snapshot) + b"\n"
                )
                atomic_json(self._output / "progress.json", snapshot)
            except BaseException as error:
                self._record_persistence_error(error)

    def _record_persistence_error(self, error: BaseException) -> None:
        with self._errors_lock:
            self._persistence_errors.append(error)

    def _reporter_loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._heartbeat()
            except BaseException as error:  # never die silently; surface via check
                self._record_persistence_error(error)
