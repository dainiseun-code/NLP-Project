"""Fictional retrieval -> scripted response -> validation, with no model or send.

Run: uv run --locked python examples/model_free_demo.py
The draft is deliberately scripted, not a saved Qwen output or a study result.
Only a temporary directory owned by this example receives preparation artifacts.
"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import TZPATH

from messaging_nlp.application import execute_condition
from messaging_nlp.application_types import GenerationOutput, PreparationPacket
from messaging_nlp.context import PublicDataset
from messaging_nlp.encoder_cache import retained_identity
from messaging_nlp.preparation import prepare_inputs
from messaging_nlp.temporal import TimezoneBinding


class ScriptedProvider:
    """Return one explicitly fictional draft; no model or messaging integration."""

    def __init__(self, contact_alias):
        self.contact_alias = contact_alias

    def generate(self, prompt, *, max_input_tokens, max_output_tokens):
        return GenerationOutput(
            raw_output=json.dumps(
                {
                    "action": "draft",
                    "recipient_contact_ids": [self.contact_alias],
                    "selected_event_id": None,
                    "draft": "Hi Rowan, do you have any updates on the seed order?",
                    "clarification": None,
                    "reason": None,
                }
            ),
            failure=None,
        )


def run_demo():
    stamp = "2026-10-01T10:00:00Z"
    dataset = PublicDataset.model_validate(
        {
            "owner_id": "fictional-owner",
            "requests": [
                {
                    "message_id": "fictional-request",
                    "text": "Ask Rowan Vale for an update on the seed order.",
                    "owner_id": "fictional-owner",
                    "request_time": stamp,
                    "time_zone": "UTC",
                    "context_snapshot_id": "fictional-snapshot",
                }
            ],
            "contacts": [
                {
                    "contact_id": "fictional-rowan",
                    "record_id": "fictional-contact",
                    "version": 1,
                    "name": "Rowan Vale",
                    "aliases": ["Rowan"],
                    "relationship": "fictional friend",
                    "available_at": stamp,
                    "provenance": "fictional_software_demo",
                }
            ],
            "calendar": [],
            "history": [],
            "snapshots": [
                {
                    "context_snapshot_id": "fictional-snapshot",
                    "timestamp": stamp,
                    "record_ids": ["fictional-contact"],
                }
            ],
        }
    )
    utc = next((Path(p) / "UTC" for p in TZPATH if (Path(p) / "UTC").is_file()), None)
    if utc is None:
        raise RuntimeError("this example requires a system UTC timezone file")
    with TemporaryDirectory(prefix="messaging-nlp-fictional-") as directory:
        root = Path(directory)
        tzif = utc.read_bytes()
        (root / "UTC.tzif").write_bytes(tzif)
        binding = TimezoneBinding(
            "UTC", root / "UTC.tzif", hashlib.sha256(tzif).hexdigest()
        )
        prepared = prepare_inputs(
            dataset,
            "fictional-request",
            root=root,
            baseline_path="baseline.json",
            assisted_path="assisted.json",
            extraction_path="not-provided.json",
            identity=retained_identity(),  # Identity only; no encoder assets opened.
            binding=binding,
        )
        packet = PreparationPacket.model_validate_json(prepared.baseline.data)
        alias = next(k for k, v in packet.aliases.items() if v.kind == "contact")
        result = execute_condition(
            dataset,
            "fictional-request",
            "baseline",
            prepared,
            ScriptedProvider(alias),
            root=root,
            run_id="fictional-software-demo",
            binding=binding,
        )
        if result.row.status != "ok":
            raise RuntimeError(f"fictional validation failed: {result.row.failure}")
        return {
            "fictional_demo": True,
            "provider": "scripted; no model inference",
            "measured_study_result": False,
            "semantic_success": "not assessed",
            "technical_status": result.row.status,
            "provider_calls": result.provider_calls,
            "retrieved_record_ids": packet.search.candidates.source_record_ids,
            "decision": result.row.decision.model_dump(),
            "messages_sent": 0,
        }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, ensure_ascii=False))
