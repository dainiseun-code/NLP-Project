"""Archive-independent Stage 1 DATA gate with explicit installed integration.

Run: uv run --locked --offline --no-sync python tests/test_study_data.py -v
SelectedDataIntegrity and its oracle use stdlib only. InstalledPublicIntegration
imports the installed distribution without source-path injection. The manifest
is a byte inventory, not semantic annotation acceptance. Original
source comparisons cannot be repeated after deletion of the original release;
immutable selected-source hashes below anchor the retained migration evidence.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
EXPORT = Path("data/stage1-v1")
SOURCE = "evaluator/provenance/source/"
VARIANTS = {"clean", "typo", "asr"}
REQUEST_FIELDS = {
    "message_id",
    "text",
    "owner_id",
    "request_time",
    "time_zone",
    "context_snapshot_id",
}
TEMP_ROOT = Path(tempfile.gettempdir()).resolve() / "opencode"
# Verified pause-snapshot source bytes, not values derived from the code under test.
# Intentional source/gold changes require separate primary authorization/review.
SOURCE_HASHES = {
    "public/calendar.jsonl": "69b99261e1344c2e6b88989d8b6b79f3d79f7b16b9a7456d96452402fcbac369",
    "public/contacts.jsonl": "06d3820eab1cd1d16ceeb936641b446e19939d372e5ccaee954818d86b2dfc2a",
    "public/history.jsonl": "eaee503ed48b77cfa8edda3b37889dd569390a462c756ba72905b8d08bf9237b",
    "public/snapshots.jsonl": "38016c6c539812bb497f6e9bf66bd6600de3e83d2d54fc7c7521704d03df7a83",
    SOURCE
    + "processed/messages.jsonl": "9517b7ae62b2eadcfe74851322c142078a10140fb33f492ffdae495a177cc28f",
    SOURCE
    + "processed/scenarios.jsonl": "753cd196d11ba30a661b955db027a36ed1b883cb59345ec7d31959f92029748b",
    SOURCE
    + "processed/families.jsonl": "436a4f43f191b49a9d5e4f92e9072ae266343f7f90dbdb7d2b69a58812528c51",
    SOURCE
    + "review/eligibility.jsonl": "0f153c69bf9c0ede3013dd43bcfc0fe2509a48febcfb3cb54b5182c4f26a845a",
    SOURCE
    + "review/provenance.jsonl": "8b2a9d8d00d0854f587b76b93a086dc357b9889d4a3eb88c9282657ea1a87226",
    "evaluator/gold/entities.jsonl": "c8dfc16a30df10828b0f944ca5ec2b0f73c72943ad633ed60d63e1914f42e09e",
    "evaluator/gold/annotation_status.jsonl": "9fcbfaf3a5f8283ab5638e7313f2a464ac7da25546c8918995357acf25e5e7e8",
}
OPENING_HASH = "3f7a172dd44af30695f9171555f13070d7408bf1e2ba88d4c2e8e6d17af4f67d"
TZIF_HASH = "e182e20ec67c46a7dd0aedb3d45fb1395edcb14ff2af2ac694a5776b307bed31"
HISTORICAL_PENDING_CONTENT_SCENARIOS = {"s0109", "s0124", "s0209"}


def annotation_status(scenario_id):
    return "model_annotated_primary_adjudicated_human_pending"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def strict_json(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="strict")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def number(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON number")
        return value

    def constant(text):
        raise ValueError(f"nonfinite JSON number: {text}")

    return json.loads(
        raw, object_pairs_hook=pairs, parse_float=number, parse_constant=constant
    )


def indexed(rows, key):
    result = {}
    for row in rows:
        value = tuple(row[k] for k in key) if isinstance(key, tuple) else row[key]
        require(value not in result, f"duplicate ID: {value}")
        result[value] = row
    return result


def instant(text):
    value = datetime.fromisoformat(text)
    require(
        value.tzinfo is not None and value.utcoffset() is not None,
        "timestamp must be offset-bearing",
    )
    return value.astimezone(timezone.utc)


def validate_inventory(root):
    export = root / EXPORT
    manifest = strict_json((export / "selection-manifest.json").read_bytes())
    actual = {
        p.relative_to(export).as_posix()
        for p in export.rglob("*")
        if p.is_file() and p != export / "selection-manifest.json"
    }
    require(actual == set(manifest["files"]), "manifest retained-file coverage")
    for name, receipt in manifest["files"].items():
        path = export / name
        require(
            not path.is_symlink() and path.resolve().is_relative_to(export.resolve()),
            "manifest path escapes export",
        )
        raw = path.read_bytes()
        require(digest(raw) == receipt["sha256"], f"file hash mismatch: {name}")
        require(len(raw) == receipt["bytes"], f"file length mismatch: {name}")
        if path.suffix == ".jsonl":
            lines = raw.splitlines(keepends=True)
            require(receipt["rows"] == len(lines), f"row count mismatch: {name}")
            rows = []
            for line in lines:
                row = strict_json(line)
                require(isinstance(row, dict), f"nonobject row: {name}")
                rows.append(row)
            if rows:
                if name.endswith("temporal-normalizations.jsonl"):
                    key = ("source_public_file", "id", "field")
                elif name.endswith("annotation_status.jsonl"):
                    key = ("message_id", "task")
                else:
                    key = next(
                        (
                            key
                            for key in (
                                "entity_id",
                                "record_id",
                                "message_id",
                                "scenario_id",
                                "context_snapshot_id",
                                "family_id",
                            )
                            if key in rows[0]
                        ),
                        None,
                    )
                    require(key is not None, f"unspecified row identity: {name}")
                indexed(rows, key)
        elif path.suffix == ".json":
            strict_json(raw)
        if name in SOURCE_HASHES:
            require(digest(raw) == SOURCE_HASHES[name], f"source bytes changed: {name}")
    require(
        "evaluator/integration-validation.json" in actual, "missing integration receipt"
    )
    return manifest


def pinned_zone(root, policy, zone_key):
    binding = policy["temporal"]["timezone_identity"]
    identity = strict_json((root / binding["identity_file"]).read_bytes())
    require(
        identity["time_zone"] == zone_key == "America/Vancouver",
        "timezone IANA identity mismatch",
    )
    for field in ("tzif_path", "tzif_sha256", "tzdb_version"):
        require(identity[field] == binding[field], f"timezone identity {field}")
    raw = (root / identity["tzif_path"]).read_bytes()
    require(digest(raw) == identity["tzif_sha256"] == TZIF_HASH, "timezone hash")
    require(identity["tzdb_version"] == "2026e", "timezone database version")
    return ZoneInfo.from_file(io.BytesIO(raw), key=zone_key)


def eligible_bundle(public, request, owner):
    """Independent requirement oracle: membership + both clocks + global versions.

    This is not the public loader. It deliberately reads plain dictionaries and
    does not repair invalid snapshots or map null message endpoints to the owner.
    """
    require(request["owner_id"] == owner, "request owner")
    snapshots = indexed(public["snapshots"], "context_snapshot_id")
    snapshot = snapshots[request["context_snapshot_id"]]
    cutoff = min(instant(request["request_time"]), instant(snapshot["timestamp"]))
    records = indexed(
        public["contacts"] + public["calendar"] + public["history"], "record_id"
    )
    ids = snapshot["record_ids"]
    require(len(ids) == len(set(ids)), "duplicate snapshot member")
    require(set(ids) <= records.keys(), "snapshot foreign key")
    latest = {}
    for group, field in (("contacts", "contact_id"), ("calendar", "event_id")):
        indexed(public[group], (field, "version"))
        by_identity = defaultdict(list)
        for row in public[group]:
            require(type(row["version"]) is int and row["version"] >= 1, "version")
            by_identity[row[field]].append(row)
        for identity, versions in by_identity.items():
            ordered = sorted(versions, key=lambda row: row["version"])
            require(
                all(
                    instant(a["available_at"]) <= instant(b["available_at"])
                    for a, b in itertools.pairwise(ordered)
                ),
                "version availability",
            )
            available = [r for r in versions if instant(r["available_at"]) <= cutoff]
            if available:
                latest[group, identity] = max(available, key=lambda r: r["version"])
        for rid in ids:
            row = records[rid]
            if field in row:
                require(latest.get((group, row[field])) == row, "stale/future version")
    selected = {rid: records[rid] for rid in ids}
    contacts = {r["contact_id"] for r in selected.values() if "contact_id" in r}
    require(
        owner not in {r["contact_id"] for r in public["contacts"]},
        "owner aliases a contact",
    )
    identities = contacts | {owner}
    for row in selected.values():
        require(instant(row["available_at"]) <= cutoff, "future availability")
        if "timestamp" in row:
            require(instant(row["timestamp"]) <= cutoff, "future message timestamp")
            refs = [row["sender_contact_id"], row["recipient_contact_id"]]
            require(None in refs or owner in refs, "exchange lacks owner")
        else:
            refs = row.get("participant_contact_ids", [])
            require(len(refs) == len(set(refs)), "duplicate participant")
        require(
            all(ref is None or ref in identities for ref in refs),
            "endpoint/participant missing eligible contact",
        )
    return selected


class SelectedDataIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export = ROOT / EXPORT
        cls.policy = strict_json((ROOT / "configs/study-policy.json").read_bytes())
        cls.manifest = strict_json(
            (cls.export / "selection-manifest.json").read_bytes()
        )
        cls.public = {
            name: cls.rows(f"public/{name}.jsonl")
            for name in ("requests", "contacts", "calendar", "history", "snapshots")
        }

    @classmethod
    def rows(cls, name):
        return [
            strict_json(line) for line in (cls.export / name).read_bytes().splitlines()
        ]

    def test_manifest_all_retained_bytes_and_strict_json(self):
        validate_inventory(ROOT)
        self.assertFalse(self.manifest["source_runtime_dependency"])
        self.assertEqual(self.manifest["status"], "stage1_accepted_not_run_frozen")

    def test_exact_public_boundary_and_original_request_lines(self):
        self.assertEqual(
            {p.name for p in (self.export / "public").iterdir()},
            {f"{name}.jsonl" for name in self.public},
        )
        source_lines = (
            self.export / (SOURCE + "processed/messages.jsonl")
        ).read_bytes()
        lines = indexed(
            [
                {"message_id": strict_json(raw)["message_id"], "raw": raw}
                for raw in source_lines.splitlines(keepends=True)
            ],
            "message_id",
        )
        requests = indexed(self.public["requests"], "message_id")
        roster = indexed(self.rows("evaluator/roster.jsonl"), "message_id")
        self.assertEqual(requests.keys(), lines.keys())
        self.assertEqual(requests.keys(), roster.keys())
        original_line_numbers = []
        for mid, request in requests.items():
            source = strict_json(lines[mid]["raw"])
            self.assertEqual(set(request), REQUEST_FIELDS)
            self.assertEqual(
                request,
                {
                    "message_id": mid,
                    "text": source["text"],
                    "owner_id": "owner_001",
                    "request_time": source["timestamp"],
                    "time_zone": source["timezone"],
                    "context_snapshot_id": source["context_snapshot_id"],
                },
            )
            self.assertEqual(
                roster[mid]["source_line_sha256"], digest(lines[mid]["raw"])
            )
            self.assertEqual(
                roster[mid]["text_sha256"], digest(request["text"].encode())
            )
            self.assertGreater(roster[mid]["source_line"], 0)
            original_line_numbers.append(roster[mid]["source_line"])
        self.assertEqual(len(set(original_line_numbers)), len(requests))

    def test_complete_snapshot_source_bytes_membership_and_version_closure(self):
        source = (self.export / (SOURCE + "context/snapshots.jsonl")).read_bytes()
        self.assertEqual(source, (self.export / "public/snapshots.jsonl").read_bytes())
        receipts = indexed(
            self.rows("evaluator/provenance/snapshot-projections.jsonl"),
            "context_snapshot_id",
        )
        snapshot_ids = {r["context_snapshot_id"] for r in self.public["requests"]}
        self.assertEqual(set(receipts), snapshot_ids)
        union = set()
        for raw in source.splitlines(keepends=True):
            row = strict_json(raw)
            receipt = receipts[row["context_snapshot_id"]]
            self.assertEqual(receipt["source_line_sha256"], digest(raw))
            self.assertEqual(receipt["removed_record_ids"], [])
            self.assertEqual(len(row["record_ids"]), len(set(row["record_ids"])))
            union.update(row["record_ids"])
        records = indexed(
            self.public["contacts"] + self.public["calendar"] + self.public["history"],
            "record_id",
        )
        # No answer-selected context: the full snapshot union, including distractors.
        self.assertEqual(union, records.keys())
        for request in self.public["requests"]:
            selected = eligible_bundle(self.public, request, "owner_001")
            self.assertTrue(selected)
        # All exported versions of retained identities through the latest cutoff
        # must be reachable in the union, not dropped in favor of an answer version.
        cutoffs = [
            min(instant(r["request_time"]), instant(s["timestamp"]))
            for r in self.public["requests"]
            for s in self.public["snapshots"]
            if s["context_snapshot_id"] == r["context_snapshot_id"]
        ]
        for group in ("contacts", "calendar"):
            for row in self.public[group]:
                if instant(row["available_at"]) <= max(cutoffs):
                    self.assertIn(row["record_id"], union)

    def test_standalone_development_selection_and_complete_variant_links(self):
        messages = indexed(self.rows(SOURCE + "processed/messages.jsonl"), "message_id")
        scenarios = indexed(
            self.rows(SOURCE + "processed/scenarios.jsonl"), "scenario_id"
        )
        families = indexed(self.rows(SOURCE + "processed/families.jsonl"), "family_id")
        eligibility = indexed(
            self.rows(SOURCE + "review/eligibility.jsonl"), "message_id"
        )
        provenance = indexed(
            self.rows(SOURCE + "review/provenance.jsonl"), "message_id"
        )
        self.assertEqual(messages.keys(), eligibility.keys())
        self.assertEqual(messages.keys(), provenance.keys())
        groups = defaultdict(list)
        for mid, message in messages.items():
            scenario = scenarios[message["scenario_id"]]
            family = families[scenario["family_id"]]
            evidence = eligibility[mid]
            self.assertEqual(scenario["split"], "development")
            self.assertEqual(family["split"], "development")
            self.assertEqual(message["turn_index"], 0)
            self.assertIs(evidence["development_eval_eligible"], True)
            self.assertIs(evidence["quarantined"], False)
            self.assertEqual(evidence["quarantine_reasons"], [])
            for field in ("scenario_id", "variant_type"):
                self.assertEqual(message[field], evidence[field])
                self.assertEqual(message[field], provenance[mid][field])
            self.assertEqual(evidence["family_id"], scenario["family_id"])
            self.assertEqual(evidence["split"], "development")
            self.assertEqual(evidence["text_sha256"], digest(message["text"].encode()))
            # Historical release rewrites are provenance, not Stage 1 rewrites.
            # The selected *released* bytes above are what Stage 1 preserves.
            self.assertEqual(
                provenance[mid]["message_id_changed"],
                provenance[mid]["original_message_id"] != mid,
            )
            self.assertEqual(
                evidence["original_message_id"], provenance[mid]["original_message_id"]
            )
            groups[message["scenario_id"]].append(message)
        self.assertEqual(groups.keys(), scenarios.keys())
        for group in groups.values():
            variants = indexed(group, "variant_type")
            self.assertEqual(set(variants), VARIANTS)
            clean = variants["clean"]
            self.assertIsNone(clean["source_message_id"])
            for variant in ("typo", "asr"):
                self.assertEqual(
                    variants[variant]["source_message_id"], clean["message_id"]
                )
                self.assertEqual(
                    variants[variant]["context_snapshot_id"],
                    clean["context_snapshot_id"],
                )
        members = indexed(
            self.rows(SOURCE + "processed/family-members.jsonl"), "family_id"
        )
        contacts = {r["contact_id"] for r in self.public["contacts"]}
        self.assertLessEqual(members.keys(), families.keys())
        for family in list(families.values()) + list(members.values()):
            self.assertLessEqual(set(family.get("contact_ids", [])), contacts)
            if "contact_id" in family:
                self.assertIn(family["contact_id"], contacts)

    def test_roster_category_pairing_order_and_cardinalities(self):
        roster = indexed(self.rows("evaluator/roster.jsonl"), "message_id")
        messages = indexed(self.rows(SOURCE + "processed/messages.jsonl"), "message_id")
        scenarios = indexed(
            self.rows(SOURCE + "processed/scenarios.jsonl"), "scenario_id"
        )
        eligibility = indexed(
            self.rows(SOURCE + "review/eligibility.jsonl"), "message_id"
        )
        permutations = list(itertools.permutations(["baseline", "B", "A+B"]))
        self.assertEqual(
            self.policy["conditions"]["permutations"], [list(p) for p in permutations]
        )
        offsets = {"clean": 0, "typo": 2, "asr": 4}
        self.assertEqual(self.policy["conditions"]["variant_offsets"], offsets)
        categories = defaultdict(list)
        for sid, source in sorted(scenarios.items()):
            for tag in source["capability_tags"]:
                categories[tag].append(sid)
        self.assertEqual(dict(categories), self.manifest["category_scenarios"])
        scenario_order = {sid: i for i, sid in enumerate(sorted(scenarios))}
        for mid, row in roster.items():
            source = messages[mid]
            scenario = scenarios[source["scenario_id"]]
            for field in (
                "scenario_id",
                "variant_type",
                "source_message_id",
                "information_loss",
            ):
                self.assertEqual(row[field], source[field])
            self.assertEqual(row["family_id"], scenario["family_id"])
            self.assertEqual(row["category_tags"], scenario["capability_tags"])
            self.assertEqual(row["source_episode"]["position"], 0)
            self.assertEqual(
                row["source_episode"]["episode_id"], eligibility[mid]["episode_id"]
            )
            i = (scenario_order[row["scenario_id"]] + offsets[row["variant_type"]]) % 6
            self.assertEqual(row["condition_permutation_index"], i)
            self.assertEqual(row["condition_order"], list(permutations[i]))
            self.assertEqual(
                row["requirement_status"], annotation_status(row["scenario_id"])
            )
            self.assertEqual(row["human_review_status"], "pending_not_human_annotated")
        counts = self.manifest["counts"]
        self.assertEqual((len(scenarios), len(roster)), (41, 123))
        self.assertEqual(counts["selected_scenarios"], len(scenarios))
        self.assertEqual(counts["selected_requests"], len(roster))
        self.assertEqual(counts["conditions_per_request"], 3)
        self.assertEqual(counts["planned_qwen_rows"], 3 * len(roster))
        for variant in VARIANTS:
            orders = Counter(
                r["condition_permutation_index"]
                for r in roster.values()
                if r["variant_type"] == variant
            )
            self.assertEqual(set(orders), set(range(6)))
            self.assertLessEqual(max(orders.values()) - min(orders.values()), 1)

    def test_exclusions_accounting_without_heldout_task_content(self):
        selected = indexed(
            self.rows(SOURCE + "processed/scenarios.jsonl"), "scenario_id"
        )
        excluded = indexed(self.rows("evaluator/exclusions.jsonl"), "scenario_id")
        self.assertFalse(selected.keys() & excluded.keys())
        self.assertEqual(len(excluded), self.manifest["counts"]["excluded_scenarios"])
        self.assertEqual(len(selected) + len(excluded), 324)
        all_message_ids = [r["message_id"] for r in self.public["requests"]]
        for row in excluded.values():
            self.assertTrue(row["reasons"])
            self.assertEqual(len(row["message_ids"]), 3)
            all_message_ids.extend(row["message_ids"])
            if row["split"] != "development":
                self.assertIn(
                    "outside_development_split_content_not_inspected", row["reasons"]
                )
            else:
                self.assertTrue(
                    set(row["reasons"])
                    & {
                        "dependent_episode_followup_prior_state_required",
                        "source_quarantined",
                        "source_development_ineligible",
                    }
                )
                evidence = row.get("source_eligibility_evidence", [])
                self.assertEqual(
                    {r["message_id"] for r in evidence}, set(row["message_ids"])
                )
                for entry in evidence:
                    if "source_quarantined" in row["reasons"]:
                        self.assertTrue(entry["quarantined"])
                        self.assertTrue(entry["quarantine_reasons"])
        self.assertEqual(len(all_message_ids), len(set(all_message_ids)))

    def test_original_noisy_entity_spans_ids_and_honest_historical_status(self):
        requests = indexed(self.public["requests"], "message_id")
        entities = indexed(self.rows("evaluator/gold/entities.jsonl"), "entity_id")
        spans = set()
        for eid, entity in entities.items():
            text = requests[entity["message_id"]]["text"]
            self.assertTrue(eid.startswith(entity["message_id"] + "_e"))
            self.assertIn(entity["type"], {"PERSON", "EVENT", "DATE", "TOPIC"})
            self.assertIs(type(entity["start"]), int)
            self.assertIs(type(entity["end"]), int)
            self.assertTrue(0 <= entity["start"] < entity["end"] <= len(text))
            self.assertEqual(text[entity["start"] : entity["end"]], entity["text"])
            key = (entity["message_id"], entity["start"], entity["end"], entity["type"])
            self.assertNotIn(key, spans)
            spans.add(key)
        statuses = indexed(
            self.rows("evaluator/gold/annotation_status.jsonl"), ("message_id", "task")
        )
        self.assertEqual(
            set(statuses),
            set(itertools.product(requests, ["extraction", "downstream"])),
        )
        eligibility = indexed(
            self.rows(SOURCE + "review/eligibility.jsonl"), "message_id"
        )
        for (mid, _), row in statuses.items():
            self.assertEqual(row["status"], "model_reviewed")
            self.assertEqual(row["human_review_status"], "pending")
            self.assertEqual(
                row["review_record_id"], eligibility[mid]["review_record_id"]
            )
            self.assertEqual(
                row["adjudication_id"], eligibility[mid]["adjudication_id"]
            )
        self.assertEqual(self.policy["data"]["human_review_status"], "pending")
        self.assertEqual(self.policy["data"]["optional_gold_diagnostic"], "deferred")

    def test_requirements_hashes_supports_and_normalized_facts_not_semantic_judgments(
        self,
    ):
        requirements = indexed(
            self.rows("evaluator/task-requirements.jsonl"), "message_id"
        )
        roster = indexed(self.rows("evaluator/roster.jsonl"), "message_id")
        requests = indexed(self.public["requests"], "message_id")
        self.assertEqual(requirements.keys(), requests.keys())
        zone = pinned_zone(ROOT, self.policy, "America/Vancouver")
        for mid, row in requirements.items():
            request = requests[mid]
            selected = eligible_bundle(self.public, request, "owner_001")
            self.assertEqual(
                row["request_text_sha256"], digest(request["text"].encode())
            )
            self.assertEqual(row["context_snapshot_id"], request["context_snapshot_id"])
            for field in ("scenario_id", "family_id", "variant_type", "category_tags"):
                self.assertEqual(row[field], roster[mid][field])
            self.assertEqual(row["requirement_version"], "stage1-source-authored-v4")
            self.assertEqual(
                row["annotation_status"], annotation_status(row["scenario_id"])
            )
            self.assertEqual(row["human_review_status"], "pending_not_human_annotation")
            self.assertIs(row["historical_outcomes_used"], False)
            self.assertIn(row["expected_action"], {"draft", "clarify", "unsupported"})
            self.assertTrue(row["required_meaning"])
            self.assertTrue(row["forbidden_meaning"])
            if row["expected_action"] == "draft":
                self.assertEqual(row["actual_blockers"], [])
                self.assertIsNone(row["clarification_slot"])
                self.assertIsNone(row["unsupported_reason"])
            elif row["expected_action"] == "clarify":
                self.assertTrue(row["actual_blockers"])
                self.assertTrue(row["clarification_slot"])
                self.assertIsNone(row["unsupported_reason"])
            else:
                self.assertTrue(row["actual_blockers"])
                self.assertTrue(row["unsupported_reason"])
                self.assertIsNone(row["clarification_slot"])
            contact_ids = {
                r["contact_id"] for r in selected.values() if "contact_id" in r
            }
            self.assertLessEqual(set(row["recipient_contact_ids"]), contact_ids)
            self.assertTrue(row["support_alternatives"])
            for alternative in row["support_alternatives"]:
                self.assertEqual(len(alternative), len(set(alternative)))
                self.assertLessEqual(set(alternative), selected.keys())
            for fact in row["normalized_facts"]:
                self.assertTrue(fact["source_record_ids"])
                self.assertLessEqual(set(fact["source_record_ids"]), selected.keys())
                for rid in fact["source_record_ids"]:
                    source = selected[rid]
                    if fact["kind"] == "selected_event":
                        self.assertEqual(fact["event_id"], source["event_id"])
                    elif fact["kind"] == "event_start":
                        self.assertEqual(fact["source_value"], source["start"])
                        self.assertEqual(fact["time_zone"], request["time_zone"])
                        value = instant(source["start"])
                        self.assertEqual(
                            fact["local_value"], value.astimezone(zone).isoformat()
                        )
                        if "instant_utc" in fact:
                            self.assertEqual(fact["instant_utc"], value.isoformat())
                    elif fact["kind"] == "owner_nonattendance_date":
                        tomorrow = instant(request["request_time"]).astimezone(
                            zone
                        ).date() + timedelta(days=1)
                        self.assertEqual(fact["value"], tomorrow.isoformat())
                        self.assertEqual(
                            fact["value"],
                            instant(source["start"])
                            .astimezone(zone)
                            .date()
                            .isoformat(),
                        )
                    elif fact["kind"] == "book_title":
                        self.assertIn(fact["value"], source["text"])
                    elif fact["kind"] in {"book_owner", "current_holder"}:
                        self.assertIn(fact["contact_id"], contact_ids)
                    else:
                        self.fail(f"unvalidated normalized fact kind: {fact['kind']}")
        # Contact/holder associations and prose adequacy require source-only
        # annotation review; membership is not a semantic endorsement.

    def test_rejected_v1_annotations_retained_only_as_provenance(self):
        old = indexed(
            self.rows("evaluator/provenance/requirements/v1-proposed.jsonl"),
            "message_id",
        )
        active = indexed(
            self.rows("evaluator/provenance/requirements/v2-proposed.jsonl"),
            "message_id",
        )
        rejections = indexed(
            self.rows("evaluator/provenance/requirements/rejections-v1-to-v2.jsonl"),
            "message_id",
        )
        self.assertEqual(old.keys(), active.keys())
        self.assertTrue(rejections)
        for mid, receipt in rejections.items():
            self.assertEqual(receipt["old_annotation_status"], "rejected")
            self.assertEqual(receipt["old_requirement"], old[mid])
            self.assertEqual(
                receipt["old_requirement_version"], "stage1-source-authored-v1"
            )
            self.assertEqual(
                receipt["new_requirement_version"], active[mid]["requirement_version"]
            )
            self.assertTrue(receipt["reason"])
            # Historical receipt binds compact insertion-order JSON, not sorted
            # canonical JSON. Preserve its actual serialization convention.
            raw = json.dumps(
                active[mid], separators=(",", ":"), ensure_ascii=False
            ).encode()
            self.assertEqual(receipt["new_requirement_sha256"], digest(raw))
        self.assertEqual(
            self.policy["data"]["task_requirements"],
            "data/stage1-v1/evaluator/task-requirements.jsonl",
        )

    def test_historical_v3_comparison_and_adjudication_bindings(self):
        base = "evaluator/provenance/requirements/"
        v2raw = (self.export / (base + "v2-proposed.jsonl")).read_bytes()
        self.assertEqual(
            digest(v2raw),
            "3488091008ce9ea7621981d5061a377e810ae6626a0f0d0d29cc41759b448b2c",
        )
        old = {
            strict_json(raw)["message_id"]: raw
            for raw in v2raw.splitlines(keepends=True)
        }
        current = {
            strict_json(raw)["message_id"]: raw
            for raw in (self.export / (base + "v3-proposed.jsonl"))
            .read_bytes()
            .splitlines(keepends=True)
        }
        requests = {
            strict_json(raw)["message_id"]: raw
            for raw in (self.export / "public/requests.jsonl")
            .read_bytes()
            .splitlines(keepends=True)
        }
        snapshots = {
            strict_json(raw)["context_snapshot_id"]: raw
            for raw in (self.export / "public/snapshots.jsonl")
            .read_bytes()
            .splitlines(keepends=True)
        }
        comparisons = indexed(
            self.rows(base + "source-only-comparison-v3.jsonl"), "message_id"
        )
        changes = indexed(self.rows(base + "v2-to-v3-changes.jsonl"), "message_id")
        decision_raw = (
            self.export / (base + "v2-to-v3-adjudication.json")
        ).read_bytes()
        decision = strict_json(decision_raw)
        auditor = self.export / "evaluator/provenance/annotation-audit"
        original = (auditor / "original-annotations.md").read_bytes()
        receipt = strict_json((auditor / "receipt.json").read_bytes())
        self.assertEqual(receipt["original_annotations_sha256"], digest(original))
        self.assertEqual(
            receipt["input_manifest_sha256"],
            digest((auditor / "inputs.json").read_bytes()),
        )
        self.assertEqual(
            receipt["assignment_sha256"],
            digest((auditor / "assignment.txt").read_bytes()),
        )
        self.assertEqual(
            receipt["model"],
            {"id": "deepseek-flash", "providerID": "deepseek", "variant": "high"},
        )
        self.assertEqual(receipt["finish"], "stop")
        self.assertFalse(receipt["proposed_labels_supplied"])
        self.assertFalse(receipt["outputs_judged"])
        access = strict_json((auditor / "input-access.json").read_bytes())
        self.assertEqual(access["session_id"], receipt["session_id"])
        self.assertEqual(access["models_observed"], [receipt["model"]])
        self.assertEqual(len(access["tools"]), 11)
        allowed_suffixes = {
            "AGENTS.md",
            "handover.md",
            "docs/agents/data-auditor.md",
            "nlp3-source-audit-v2-tzuhox75/manifest.json",
            "nlp3-source-audit-v2-tzuhox75/policy.json",
            "nlp3-source-audit-v2-tzuhox75/variant-links.jsonl",
            *{
                f"nlp3-source-audit-v2-tzuhox75/public/{name}.jsonl"
                for name in self.public
            },
        }
        observed = set()
        for tool in access["tools"]:
            self.assertEqual(tool["name"], "read")
            self.assertEqual(tool["status"], "completed")
            matches = {
                suffix
                for suffix in allowed_suffixes
                if tool["input"]["path"].endswith("/" + suffix)
            }
            self.assertEqual(len(matches), 1)
            observed.update(matches)
        self.assertEqual(observed, allowed_suffixes)
        inputs = strict_json((auditor / "inputs.json").read_bytes())
        for path, binding in inputs["files"].items():
            raw = (ROOT / path).read_bytes()
            self.assertEqual(digest(raw), binding["sha256"])
            self.assertEqual(len(raw), binding["bytes"])
        self.assertEqual(decision["v2_source_sha256"], digest(v2raw))
        self.assertEqual(decision["original_annotation_sha256"], digest(original))
        self.assertEqual(
            decision["decision_section_sha256"],
            digest(decision["decision_section_text"].encode()),
        )
        # Preserve authority bytes, not a runtime dependency on a future mutable log.
        self.assertEqual(
            digest(decision["decision_document_utf8"].encode()),
            decision["decision_document_sha256"],
        )
        self.assertIn(
            decision["decision_section_text"], decision["decision_document_utf8"]
        )
        self.assertEqual(current.keys(), comparisons.keys())
        self.assertEqual(current.keys(), changes.keys())
        self.assertEqual(current.keys(), old.keys())
        dimensions = {
            "expected_action",
            "recipient_contact_ids",
            "required_meaning",
            "forbidden_meaning",
            "support_alternatives",
            "normalized_facts",
            "actual_blockers",
        }
        for mid, row in comparisons.items():
            active, previous = strict_json(current[mid]), strict_json(old[mid])
            self.assertEqual(row["status"], active["annotation_status"])
            self.assertEqual(row["v2_requirement_source_line_sha256"], digest(old[mid]))
            self.assertEqual(
                row["v3_requirement_source_line_sha256"], digest(current[mid])
            )
            self.assertEqual(row["request_jsonl_line_sha256"], digest(requests[mid]))
            self.assertEqual(
                row["snapshot_source_line_sha256"],
                digest(snapshots[row["context_snapshot_id"]]),
            )
            self.assertIn(
                row["original_auditor_table_row"].encode(),
                original.splitlines(keepends=True),
            )
            self.assertEqual(
                row["original_auditor_table_row_sha256"],
                digest(row["original_auditor_table_row"].encode()),
            )
            self.assertEqual(
                row["primary_adjudication_receipt_sha256"], digest(decision_raw)
            )
            self.assertEqual(
                row["auditor_receipt_sha256"],
                digest((auditor / "receipt.json").read_bytes()),
            )
            self.assertEqual(set(row["dimensions"]), dimensions)
            for field, comparison in row["dimensions"].items():
                self.assertEqual(comparison["v2"], previous[field])
                self.assertEqual(comparison["v3"], active[field])
                self.assertTrue(comparison["comparison"])
            if row["scenario_id"] in HISTORICAL_PENDING_CONTENT_SCENARIOS:
                self.assertEqual(
                    row["dimensions"]["required_meaning"]["comparison"],
                    "pending_additional_primary_content_decision",
                )
            self.assertEqual(
                changes[mid]["old_requirement_source_line_sha256"], digest(old[mid])
            )
            self.assertEqual(
                changes[mid]["new_requirement_source_line_sha256"], digest(current[mid])
            )
            for field in changes[mid]["changed_semantic_fields"]:
                self.assertNotEqual(previous[field], active[field])
        for variant in VARIANTS:
            ambiguous = strict_json(
                current[f"bb_f_development_ambiguous_match_{variant}"]
            )
            self.assertEqual(ambiguous["expected_action"], "draft")
            self.assertEqual(ambiguous["recipient_contact_ids"], ["bb_victor"])
            self.assertEqual(ambiguous["actual_blockers"], [])
            in_person = strict_json(current[f"bb_f_development_in_person_{variant}"])
            self.assertEqual(in_person["expected_action"], "unsupported")
            self.assertEqual(len(in_person["required_meaning"]), 1)
        self.assertEqual(strict_json(current["m0105_asr"])["expected_action"], "draft")
        self.assertEqual(
            comparisons["m0105_asr"]["dimensions"]["expected_action"][
                "auditor_interpretation"
            ],
            "clarify",
        )

    def test_active_v4_content_decision_history_and_all_variant_bindings(self):
        base = "evaluator/provenance/requirements/"

        def raw_index(name):
            return {
                strict_json(raw)["message_id"]: raw
                for raw in (self.export / name).read_bytes().splitlines(keepends=True)
            }

        old = raw_index(base + "v3-proposed.jsonl")
        current = raw_index("evaluator/task-requirements.jsonl")
        historical_ledger = raw_index(base + "source-only-comparison-v3.jsonl")
        ledger = indexed(
            self.rows(base + "source-only-comparison-v4.jsonl"), "message_id"
        )
        changes = indexed(self.rows(base + "v3-to-v4-changes.jsonl"), "message_id")
        authority_raw = (
            self.export / (base + "v3-to-v4-adjudication.json")
        ).read_bytes()
        authority = strict_json(authority_raw)
        self.assertEqual(
            authority["primary_decision_sha256"],
            digest(authority["primary_decision_utf8"].encode()),
        )
        self.assertIn("examples, not exclusive", authority["primary_decision_utf8"])
        self.assertEqual(authority["pending_content_decisions"], [])
        self.assertFalse(authority["execution_freeze"])
        self.assertFalse(authority["outcome_judging"])
        for name, field in (
            ("v3-proposed.jsonl", "preserved_v3_requirements_sha256"),
            ("v3-roster.jsonl", "preserved_v3_roster_sha256"),
            (
                "source-only-comparison-v3.jsonl",
                "preserved_v3_comparison_ledger_sha256",
            ),
            ("v2-to-v3-adjudication.json", "preserved_v3_primary_adjudication_sha256"),
        ):
            self.assertEqual(
                digest((self.export / (base + name)).read_bytes()), authority[field]
            )
        self.assertEqual(len(changes), 9)
        self.assertEqual(current.keys(), old.keys())
        self.assertEqual(current.keys(), ledger.keys())
        roster = indexed(self.rows("evaluator/roster.jsonl"), "message_id")
        old_roster = indexed(self.rows(base + "v3-roster.jsonl"), "message_id")
        requests = raw_index("public/requests.jsonl")
        snapshots = {
            strict_json(raw)["context_snapshot_id"]: raw
            for raw in (self.export / "public/snapshots.jsonl")
            .read_bytes()
            .splitlines(keepends=True)
        }
        for mid, raw in current.items():
            active, previous = strict_json(raw), strict_json(old[mid])
            entry = ledger[mid]
            self.assertEqual(active["requirement_version"], "stage1-source-authored-v4")
            self.assertEqual(entry["status"], annotation_status(active["scenario_id"]))
            self.assertEqual(entry["pending_content_decisions"], [])
            self.assertEqual(
                entry["v3_requirement_source_line_sha256"], digest(old[mid])
            )
            self.assertEqual(entry["v4_requirement_source_line_sha256"], digest(raw))
            self.assertEqual(
                entry["historical_v3_comparison_row_sha256"],
                digest(historical_ledger[mid]),
            )
            self.assertEqual(
                entry["primary_adjudication_receipt_sha256"], digest(authority_raw)
            )
            self.assertEqual(entry["request_jsonl_line_sha256"], digest(requests[mid]))
            self.assertEqual(
                entry["snapshot_source_line_sha256"],
                digest(snapshots[entry["context_snapshot_id"]]),
            )
            historical_entry = strict_json(historical_ledger[mid])
            for field in (
                "original_annotations_sha256",
                "original_auditor_table_row",
                "original_auditor_table_row_sha256",
                "auditor_receipt_sha256",
                "auditor_input_access_sha256",
                "source_packet_manifest_sha256",
            ):
                self.assertEqual(entry[field], historical_entry[field])
            for field, comparison in entry["dimensions"].items():
                self.assertEqual(comparison["v3"], previous[field])
                self.assertEqual(comparison["v4"], active[field])
            content_change = (
                active["scenario_id"] in HISTORICAL_PENDING_CONTENT_SCENARIOS
            )
            allowed_changes = {
                "requirement_version",
                "annotation_status",
                "annotation_review_id",
            }
            if content_change:
                allowed_changes |= {"required_meaning", "forbidden_meaning"}
                self.assertIn(mid, changes)
                self.assertEqual(
                    changes[mid]["old_requirement_source_line_sha256"], digest(old[mid])
                )
                self.assertEqual(
                    changes[mid]["new_requirement_source_line_sha256"], digest(raw)
                )
                self.assertIn("genuine question", active["required_meaning"][0])
                self.assertIn(
                    "not the only accepted kinds", active["required_meaning"][0]
                )
                self.assertNotIn(
                    "Invent an update instead of asking for one.",
                    active["forbidden_meaning"],
                )
                self.assertEqual(
                    previous["annotation_status"],
                    "model_annotated_pending_additional_primary_content_decision",
                )
            else:
                self.assertNotIn(mid, changes)
            for field in previous.keys() | active.keys():
                if field not in allowed_changes:
                    self.assertEqual(previous[field], active[field], (mid, field))
            for field in old_roster[mid].keys() | roster[mid].keys():
                if field not in {"requirement_status", "annotation_review_id"}:
                    self.assertEqual(
                        old_roster[mid][field], roster[mid][field], (mid, field)
                    )
        self.assertEqual(self.policy["data"]["pending_content_scenarios"], [])
        self.assertEqual(
            self.policy["data"]["task_requirement_version"], "stage1-source-authored-v4"
        )
        self.assertEqual(
            self.policy["data"]["source_only_comparison_ledger"],
            "data/stage1-v1/evaluator/provenance/requirements/source-only-comparison-v4.jsonl",
        )

    def test_timezone_identity_and_all_retained_temporal_normalizations(self):
        zone = pinned_zone(ROOT, self.policy, "America/Vancouver")
        self.assertEqual(
            instant("2026-11-03T15:00:00-07:00").astimezone(zone).isoformat(),
            "2026-11-03T15:00:00-07:00",
        )
        expected = {}
        fields = {
            "requests": ("request_time",),
            "calendar": ("start", "available_at"),
            "history": ("timestamp", "available_at"),
            "snapshots": ("timestamp",),
        }
        for group, names in fields.items():
            for row in self.public[group]:
                rid = row.get(
                    "record_id", row.get("message_id", row.get("context_snapshot_id"))
                )
                for field in names:
                    expected[f"public/{group}.jsonl", rid, field] = row[field]
        receipts = indexed(
            self.rows("evaluator/provenance/temporal-normalizations.jsonl"),
            ("source_public_file", "id", "field"),
        )
        self.assertEqual(receipts.keys(), expected.keys())
        for key, receipt in receipts.items():
            self.assertEqual(receipt["source_value"], expected[key])
            value = instant(expected[key])
            self.assertEqual(receipt["instant_utc"], value.isoformat())
            self.assertEqual(receipt["local_value"], value.astimezone(zone).isoformat())
            self.assertEqual(receipt["time_zone"], "America/Vancouver")
            self.assertEqual(
                receipt["source_offset_differs_from_zone_at_instant"],
                datetime.fromisoformat(expected[key]).utcoffset()
                != value.astimezone(zone).utcoffset(),
            )

    def test_budget_opening_hash_existing_charges_caps_and_versions_preserved(self):
        resources = self.policy["resources"]
        raw = (ROOT / resources["opening_record"]).read_bytes()
        self.assertEqual(digest(raw), resources["opening_record_sha256"])
        self.assertEqual(digest(raw), OPENING_HASH)
        opening = strict_json(raw)
        reservations = indexed(opening["reservations"], "reservation_id")
        self.assertEqual(len(reservations), 416)
        usage = defaultdict(lambda: Counter(seconds=0, generations=0, bytes=0))
        for row in reservations.values():
            for unit in ("seconds", "generations", "bytes"):
                planned, actual = row[f"planned_{unit}"], row[f"actual_{unit}"]
                self.assertGreaterEqual(planned, 0)
                self.assertGreaterEqual(actual, 0)
                charged = (
                    0
                    if row["status"] == "cancelled"
                    else actual
                    if row["status"] == "finished"
                    else max(planned, actual)
                )
                usage[row["stage"]][unit] += charged
        for account, totals in opening["prior_usage_by_historical_account"].items():
            self.assertEqual(dict(usage[account]), totals)
        self.assertEqual(
            dict(usage["stage3"]),
            {"seconds": 11075, "generations": 2054, "bytes": 78719238},
        )
        self.assertEqual(opening["caps"]["stage3_max_generations"], 3000)
        self.assertEqual(opening["caps"]["stage3_max_seconds"], 21600)
        self.assertEqual(opening["caps"]["max_storage_bytes"], 5368709120)
        self.assertEqual(
            resources["remaining_generation_ceiling_at_opening"],
            opening["caps"]["stage3_max_generations"] - usage["stage3"]["generations"],
        )
        self.assertEqual(
            resources["remaining_seconds_ceiling_at_opening"],
            opening["caps"]["stage3_max_seconds"] - usage["stage3"]["seconds"],
        )
        self.assertEqual(
            opening["prior_global_seconds"], sum(u["seconds"] for u in usage.values())
        )
        self.assertEqual(
            opening["prior_storage_bytes"], sum(u["bytes"] for u in usage.values())
        )
        self.assertEqual(
            len(opening["registered_prompt_versions"]["assistant_prompt"]), 2
        )
        self.assertEqual(len(opening["registered_prompt_versions"]["extraction"]), 1)
        self.assertFalse(opening["new_study_policy"]["source_lock_ownership_imported"])
        for flag in (
            "new_cap_expansion",
            "borrow_training_account",
            "reset_prior_charges",
            "stage1_model_execution_authorized",
        ):
            self.assertIs(resources[flag], False)


class ValidationOracleFaultTests(unittest.TestCase):
    """Fictional boundary cases, specified without using research requirements."""

    def test_nested_duplicate_json_keys_nonfinite_utf8_and_duplicate_ids(self):
        for raw in (
            b'{"x":1,"x":2}',
            b'{"nested":[{"id":1,"id":2}]}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":1e999}',
            b'{"x":"\xff"}',
            '{"x":1}'.encode("utf-16"),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json(raw)
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            indexed([{"id": "same"}, {"id": "same"}], "id")

    def fixture(self):
        before, now, after = (
            "2031-01-01T00:00:00+00:00",
            "2031-01-02T00:00:00+00:00",
            "2031-01-03T00:00:00+00:00",
        )
        contact = {
            "record_id": "person-v1",
            "contact_id": "person",
            "version": 1,
            "available_at": before,
        }
        event = {
            "record_id": "event-v1",
            "event_id": "event",
            "version": 1,
            "available_at": before,
            "participant_contact_ids": ["person", "account"],
            "status": "cancelled",
        }
        history = {
            "record_id": "letter",
            "sender_contact_id": None,
            "recipient_contact_id": "account",
            "timestamp": before,
            "available_at": before,
        }
        public = {
            "contacts": [contact],
            "calendar": [event],
            "history": [history],
            "snapshots": [
                {
                    "context_snapshot_id": "snapshot",
                    "timestamp": now,
                    "record_ids": ["person-v1", "event-v1", "letter"],
                }
            ],
        }
        request = {
            "owner_id": "account",
            "context_snapshot_id": "snapshot",
            "request_time": now,
        }
        return public, request, before, now, after

    def test_null_endpoint_is_unknown_and_cancellation_not_repaired(self):
        public, request, *_ = self.fixture()
        original = copy.deepcopy(public)
        bundle = eligible_bundle(public, request, "account")
        self.assertIsNone(bundle["letter"]["sender_contact_id"])
        self.assertEqual(bundle["event-v1"]["status"], "cancelled")
        self.assertEqual(public, original)

    def test_foreign_keys_owner_and_membership_fail_closed(self):
        mutations = [
            lambda p, r: r.update(owner_id="person"),
            lambda p, r: p["contacts"][0].update(contact_id="account"),
            lambda p, r: p["history"][0].update(sender_contact_id="ghost"),
            lambda p, r: p["history"][0].update(
                sender_contact_id="person", recipient_contact_id="person"
            ),
            lambda p, r: p["calendar"][0].update(participant_contact_ids=["ghost"]),
            lambda p, r: p["snapshots"][0]["record_ids"].append("missing"),
            lambda p, r: p["snapshots"][0]["record_ids"].append("letter"),
            lambda p, r: p["snapshots"][0]["record_ids"].remove("person-v1"),
            lambda p, r: p["history"].append(copy.deepcopy(p["history"][0])),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                public, request, *_ = self.fixture()
                mutate(public, request)
                with self.assertRaises(ValueError):
                    eligible_bundle(public, request, "account")

    def test_both_message_clocks_and_earlier_request_snapshot_cutoff(self):
        for field in ("timestamp", "available_at"):
            for early_clock in ("request", "snapshot"):
                with self.subTest(field=field, early_clock=early_clock):
                    public, request, _, now, after = self.fixture()
                    public["history"][0][field] = after
                    if early_clock == "request":
                        public["snapshots"][0]["timestamp"] = after
                    else:
                        request["request_time"] = after
                    with self.assertRaises(ValueError):
                        eligible_bundle(public, request, "account")
                    public["history"][0][field] = now  # inclusive cutoff
                    eligible_bundle(public, request, "account")

    def test_global_unlisted_newer_available_version_not_future_version(self):
        for group in ("contacts", "calendar"):
            with self.subTest(group=group):
                public, request, before, now, after = self.fixture()
                newer = dict(
                    public[group][0],
                    record_id="unlisted-v2",
                    version=2,
                    available_at=after,
                )
                public[group].append(newer)
                eligible_bundle(
                    public, request, "account"
                )  # future version is not evidence
                newer["available_at"] = now
                with self.assertRaisesRegex(ValueError, "stale/future version"):
                    eligible_bundle(public, request, "account")
                newer["available_at"] = before
                public["snapshots"][0]["record_ids"].remove(
                    public[group][0]["record_id"]
                )
                public["snapshots"][0]["record_ids"].append("unlisted-v2")
                eligible_bundle(public, request, "account")

    def test_timezone_hash_and_iana_key_cannot_be_relabelled(self):
        policy = strict_json((ROOT / "configs/study-policy.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "IANA identity"):
            pinned_zone(ROOT, policy, "Europe/London")
        wrong = copy.deepcopy(policy)
        wrong["temporal"]["timezone_identity"]["tzif_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "timezone identity"):
            pinned_zone(ROOT, wrong, "America/Vancouver")


class InstalledPublicIntegration(unittest.TestCase):
    def test_installed_public_loader_matches_independent_bundles_without_gold_reads(
        self,
    ):
        from messaging_nlp.context import load_public_dataset

        self.assertEqual(importlib.metadata.version("messaging-nlp"), "0.1.0")
        public_root = ROOT / EXPORT / "public"
        public = {
            name: [
                strict_json(line)
                for line in (public_root / f"{name}.jsonl").read_bytes().splitlines()
            ]
            for name in ("requests", "contacts", "calendar", "history", "snapshots")
        }
        allowed = {public_root / f"{name}.jsonl" for name in public}
        opened = []
        real_open = Path.open

        def public_only(path, *args, **kwargs):
            self.assertIn(
                path, allowed, f"public loader attempted nonpublic read: {path}"
            )
            opened.append(path)
            return real_open(path, *args, **kwargs)

        with patch.object(Path, "open", public_only):
            dataset = load_public_dataset(public_root, owner_id="owner_001")
            for request in public["requests"]:
                expected = eligible_bundle(public, request, "owner_001")
                bundle = dataset.bundle(request["message_id"])
                actual = {
                    row.record_id: row.model_dump(exclude_unset=True)
                    for group in (
                        bundle.context.contacts,
                        bundle.context.calendar,
                        bundle.context.history,
                    )
                    for row in group
                }
                self.assertEqual(actual, expected)
                self.assertEqual(bundle.request.model_dump(), request)
        self.assertEqual(set(opened), allowed)

    def test_installed_timezone_requires_trusted_name_path_hash_binding(self):
        from messaging_nlp.contracts import Request
        from messaging_nlp.temporal import TimezoneBinding, local_week, request_zone

        policy = strict_json((ROOT / "configs/study-policy.json").read_bytes())
        # Authenticate the identity and reconcile policy before constructing binding.
        pinned_zone(ROOT, policy, "America/Vancouver")
        identity = strict_json(
            (
                ROOT / policy["temporal"]["timezone_identity"]["identity_file"]
            ).read_bytes()
        )
        binding = TimezoneBinding(
            identity["time_zone"], ROOT / identity["tzif_path"], identity["tzif_sha256"]
        )
        request = Request.model_validate(
            strict_json(
                (ROOT / EXPORT / "public/requests.jsonl").read_bytes().splitlines()[0]
            )
        )
        self.assertEqual(
            request_zone(request, binding=binding).key, "America/Vancouver"
        )
        start, end = local_week(request, binding=binding)
        self.assertEqual(start.isoformat(), "2026-11-01T00:00:00-07:00")
        self.assertEqual(end.isoformat(), "2026-11-08T00:00:00-07:00")
        wrong_zone = Request.model_validate(
            dict(request.model_dump(), time_zone="Europe/London")
        )
        with self.assertRaisesRegex(ValueError, "trusted timezone binding"):
            request_zone(wrong_zone, binding=binding)
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            request_zone(
                request,
                binding=TimezoneBinding(binding.time_zone, binding.tzif_path, "0" * 64),
            )
        with self.assertRaises(TypeError):
            request_zone(
                request, tzif_path=binding.tzif_path, tzif_sha256=binding.tzif_sha256
            )


class PortabilityTests(unittest.TestCase):
    def test_relocated_data_gate_without_archive_package_models_or_old_builders(self):
        # Use the authorized host location when present; remain runnable on a
        # different machine rather than depending on this host's user-directory ID.
        temp_parent = TEMP_ROOT if TEMP_ROOT.is_dir() else None
        with tempfile.TemporaryDirectory(
            prefix="stage1-data-", dir=temp_parent
        ) as temp:
            root = Path(temp)
            shutil.copytree(ROOT / EXPORT, root / EXPORT)
            (root / "configs").mkdir()
            (root / "docs").mkdir()
            (root / "tests").mkdir()
            shutil.copy2(
                ROOT / "configs/study-policy.json", root / "configs/study-policy.json"
            )
            shutil.copy2(
                ROOT / "docs/stage1-budget-opening.json",
                root / "docs/stage1-budget-opening.json",
            )
            script = root / "tests/test_study_data.py"
            shutil.copy2(Path(__file__), script)
            # Same interpreter, no installation or PYTHONPATH/source injection.
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(script),
                    "SelectedDataIntegrity",
                    "InstalledPublicIntegration",
                    "-v",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((root / "Old-version").exists())
            self.assertFalse((root / "src").exists())
            self.assertFalse((root / "models").exists())
            # Inventory mutation checks operate on copies, never research inputs.
            original = (root / EXPORT / "public/requests.jsonl").read_bytes()
            (root / EXPORT / "public/requests.jsonl").write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_inventory(root)
            (root / EXPORT / "public/requests.jsonl").write_bytes(original)
            (root / EXPORT / "evaluator/unbound-receipt.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "retained-file coverage"):
                validate_inventory(root)


if __name__ == "__main__":
    unittest.main()
