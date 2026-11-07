"""Private model-free process fixture, not a public preparation override."""

import os
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()


def guard(event, args):
    if event == "import" and args[0].split(".")[0] in {
        "torch",
        "transformers",
        "mlx",
        "mlx_lm",
        "safetensors",
        "tokenizers",
    }:
        raise AssertionError("fake preparation cannot import a model runtime")
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(
            path.is_relative_to(ROOT / name)
            for name in ("models", "Old-version", "data")
        ):
            raise AssertionError("fake preparation cannot read research/assets/archive")


sys.addaudithook(guard)


def main():
    from messaging_nlp.contracts import Extraction, ExtractionResult
    from messaging_nlp.encoder_cache import (
        ExtractionArtifact,
        retained_identity,
        text_sha256,
    )
    from messaging_nlp.io import canonical_json
    from messaging_nlp.preparation import digest
    from messaging_nlp.preparation_worker import _serve, _verify_worker_source
    from messaging_nlp.run_contracts import WorkerEvent
    from messaging_nlp.run_supervision import watch_parent

    fd = int(os.environ["NLP3_PROTOCOL_FD"])
    watch_parent(int(os.environ["NLP3_LIFELINE_FD"]), fd)

    class Runtime:
        def __init__(self, payload):
            self.mode, self.model = payload["mode"], payload["model"]
            self.identity = dict(
                mode=self.mode,
                worker_source_sha256=_verify_worker_source(Path(payload["root"])),
                encoder_sha256=retained_identity().digest(),
            )

        def generate(self, payload, emit):
            emit("generation_started", dict(qwen_generation=False, operation=self.mode))
            if self.mode == "encoder":
                identity = retained_identity()
                result = ExtractionResult(
                    message_id=payload["message_id"],
                    status="ok",
                    extraction=Extraction(entities=[]),
                    failure=None,
                    raw_output="fictional",
                    input_sha256=text_sha256(payload["text"]),
                    extractor_sha256=identity.digest(),
                )
                artifact = ExtractionArtifact(
                    text=payload["text"],
                    identity=identity,
                    result=result,
                    diagnostics={"forward_performed": False, "fictional": True},
                )
                reply = dict(mode=self.mode, artifact=artifact.model_dump())
            else:
                reply = dict(
                    mode=self.mode,
                    messages_sha256=payload["messages_sha256"],
                    model_receipt_sha256=self.model["receipt"]["sha256"],
                    template_sha256=self.model["template_sha256"],
                    token_ids=[1, 2, 3],
                    token_count=3,
                    token_ids_sha256=digest(canonical_json([1, 2, 3])),
                    fits=True,
                )
            emit("fragment", dict(text=canonical_json(reply).decode()))
            emit(
                "terminal",
                dict(
                    finish_reason="stop",
                    generation_tokens=0,
                    input_tokens=3,
                    elapsed_seconds=0.001,
                ),
            )

    def emit(kind, payload, row_id=None):
        raw = (
            canonical_json(
                WorkerEvent(kind=kind, payload=payload, row_id=row_id).model_dump()
            )
            + b"\n"
        )
        while raw:
            count = os.write(fd, raw)
            raw = raw[count:]

    return _serve(sys.stdin.buffer, emit, Runtime)


if __name__ == "__main__":
    raise SystemExit(main())
