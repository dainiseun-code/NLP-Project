"""Private fictional child; never exposed by the production command."""

import importlib.abc
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class _NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch",
            "mlx",
            "mlx_lm",
            "transformers",
            "tokenizers",
            "sentencepiece",
        }:
            raise AssertionError(f"optional runtime import in fake child: {fullname}")


sys.meta_path.insert(0, _NoModels())
_PROJECT = Path(__file__).resolve().parents[2]


def _guard_assets(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, Path)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(
            path.is_relative_to(_PROJECT / name)
            for name in ("data", "models", "Old-version")
        ):
            raise AssertionError(f"retained asset open in fake child: {path}")


sys.addaudithook(_guard_assets)


def main():
    from messaging_nlp.run_contracts import WorkerEvent, WorkerRequest
    from messaging_nlp.run_supervision import watch_parent

    protocol = int(os.environ["NLP3_PROTOCOL_FD"])
    watch_parent(int(os.environ["NLP3_LIFELINE_FD"]), protocol)
    mode = sys.argv[1]
    if mode not in {
        "success",
        "partial",
        "invalid_json",
        "length",
        "stall",
        "crash",
        "descendant",
        "parent_loss",
        "ready",
        "load_error",
        "application",
        "application_length",
        "blocked_stdin",
        "progress_application",
    }:
        raise ValueError("unknown private fictional mode")

    def emit(kind, row=None, **payload):
        event = WorkerEvent(kind=kind, row_id=row, payload=payload)
        os.write(protocol, (event.model_dump_json() + "\n").encode())

    print("incidental stdout, not a protocol event", flush=True)
    print("fictional stderr", file=sys.stderr, flush=True)
    if mode == "blocked_stdin":
        time.sleep(60)
    for line in sys.stdin:
        request = WorkerRequest.model_validate_json(line)
        if request.kind == "shutdown":
            return
        if request.kind == "load":
            if mode == "progress_application":
                time.sleep(0.06)
            if mode == "load_error":
                emit("error", message="fictional load error")
            else:
                emit("ready", identity={"fictional": True, "pid": os.getpid()})
            continue
        row = request.row_id
        emit("generation_started", row)
        if mode in ("stall", "parent_loss"):
            time.sleep(60)
            continue
        if mode == "descendant":
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "time.sleep(60)",
                ]
            )
            emit("fragment", row, text=json.dumps({"descendant_pid": child.pid}))
            os._exit(7)
        text = "not JSON" if mode == "invalid_json" else '{"fictional":true}'
        if mode in ("application", "application_length", "progress_application"):
            text = json.dumps(
                {
                    "decision": {
                        "action": "unsupported",
                        "selection": None,
                        "draft": None,
                        "clarification": None,
                        "reason": "Fictional no-send limit.",
                    },
                    "mentions": [],
                }
            )
        if mode in ("partial", "crash"):
            emit("fragment", row, text='{"fictional":')
            if mode == "crash":
                os._exit(9)
            time.sleep(60)
            continue
        if mode == "progress_application":
            time.sleep(0.06)
            for count in range(1, 1025):
                if count == 1:
                    emit("fragment", row, text=text, generation_tokens=count)
                else:
                    emit("progress", row, generation_tokens=count)
        else:
            emit("fragment", row, text=text)
        emit(
            "terminal",
            row,
            finish_reason="length"
            if mode in ("length", "application_length")
            else "stop",
            generation_tokens=1024 if mode == "progress_application" else 4,
            input_tokens=request.payload.get("token_receipt", {}).get("token_count", 8),
            elapsed_seconds=0.01,
        )


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    main()
