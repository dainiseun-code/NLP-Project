"""Isolated SOFTWARE acceptance tests; never part of the root pure test gate.

Acceptance design (written from the Stage 2 contract before runtime inspection):
* Construct only fictional, local, tiny masked-LM safetensors and exactly the
  final-block/head component. Bind expected identity at fixture construction;
  mutated adjacent metadata is never an authority for that expected identity.
* Compare assembled full-forward logits with an independent full encoder whose
  final block and head were replaced, using a nontrivial attention mask.
* Require CPU float32, eval, frozen parameters, and inference-mode/no-grad calls.
* Reject incomplete/extra base tensors (including arbitrary MLM extras), component
  keys/shapes, wrong labels/hidden size/block index, stale bytes and inconsistent
  metadata/receipt/checkpoint identities, and custom-code configuration hooks.
* Use a fake tokenizer only; no pretrained tokenizer or retained asset is opened.

Execution belongs exclusively to the lead holding the local Torch-test lease:
uv run --locked --offline --project environments/encoder python -m unittest \
    discover -s environments/encoder/tests -p 'test_encoder_tiny.py' -v
These tiny synthetic checks establish software behavior, not model quality.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM

from messaging_nlp.encoder_cache import EncoderIdentity, FileIdentity
from messaging_nlp.encoder_decode import LABELS


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


class FakeTokenizer:
    """No real tokenizer assets are interpreted; offsets address exact input."""

    is_fast = True
    model_max_length = 256

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {
            "input_ids": [0, 4, 5, 2],
            "attention_mask": [1, 1, 1, 1],
            "offset_mapping": [(0, 0), (0, 3), (4, 7), (0, 0)],
            "special_tokens_mask": [1, 0, 0, 1],
        }


class TinyFixture:
    """Trusted fictional construction; identity is not inferred from corrupt files."""

    base_directory = "fixture/base"
    component_directory = "fixture/component"
    receipt_path = "fixture/receipts/base.json"
    repo = "fictional/tiny-xlmr"
    revision = "fixture-revision-1"

    def __init__(self, root: Path):
        self.root = root
        self.base = root / self.base_directory
        self.component = root / self.component_directory
        self.base.mkdir(parents=True)
        self.component.mkdir(parents=True)
        (root / self.receipt_path).parent.mkdir(parents=True)
        self.config = XLMRobertaConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=32,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            attn_implementation="eager",
        )
        self.config.architectures = ["XLMRobertaForMaskedLM"]
        torch.manual_seed(1729)
        self.reference = XLMRobertaForMaskedLM(self.config).cpu().float().eval()
        self.base_state = {
            name: tensor.detach().clone()
            for name, tensor in self.reference.state_dict().items()
        }
        # A genuinely different final block, not the base block in disguise.
        replacement = XLMRobertaForMaskedLM(self.config)
        self.final_state = {
            name: tensor.detach().clone()
            for name, tensor in replacement.roberta.encoder.layer[-1]
            .state_dict()
            .items()
        }
        self.head = torch.nn.Linear(8, len(LABELS)).cpu().float().eval()
        self.head_state = {
            name: tensor.detach().clone()
            for name, tensor in self.head.state_dict().items()
        }
        self.reference.roberta.encoder.layer[-1].load_state_dict(self.final_state)
        self.component_state = {
            "final_block": self.final_state,
            "head": self.head_state,
        }
        self.config_data = self.config.to_dict()
        self.write_base()
        self.write_component()

    def write_base(self):
        (self.base / "config.json").write_bytes(json_bytes(self.config_data))
        # Clones remove tied-storage ambiguity while preserving the complete MLM
        # checkpoint namespace, including its explicitly permitted MLM tensors.
        save_file(
            {key: value.clone() for key, value in self.base_state.items()},
            str(self.base / "model.safetensors"),
            metadata={"format": "pt"},
        )
        (self.base / "tokenizer.json").write_bytes(b'{"fictional":true}\n')
        (self.base / "tokenizer_config.json").write_bytes(b'{"model_max_length":256}\n')
        (self.base / "sentencepiece.bpe.model").write_bytes(
            b"fictional-not-a-tokenizer\n"
        )

    def write_component(self):
        torch.save(self.component_state, self.component / "block-head.pt")

    def file_identities(self):
        return {
            name: FileIdentity(bytes=len(data), sha256=digest(data))
            for name in (
                "config.json",
                "model.safetensors",
                "sentencepiece.bpe.model",
                "tokenizer.json",
                "tokenizer_config.json",
            )
            for data in [(self.base / name).read_bytes()]
        }

    def seal(self, *, metadata_changes=None, receipt_changes=None):
        """Seal intentional fixture bytes once, before the tested corruption.

        Metadata/receipt wire keys must be aligned by the lead with the runtime
        owner's format if it differs; no behavioral assertion may be weakened.
        """
        files = self.file_identities()
        receipt = {
            "repo": self.repo,
            "revision": self.revision,
            "directory": self.base_directory,
            "total_bytes": sum(item.bytes for item in files.values()),
            "files": [
                {"file": name, **item.model_dump(), "verified": True}
                for name, item in files.items()
            ],
        }
        receipt.update(receipt_changes or {})
        receipt_data = json_bytes(receipt)
        (self.root / self.receipt_path).write_bytes(receipt_data)
        component_hash = digest((self.component / "block-head.pt").read_bytes())
        metadata = {
            "format": "nlp3-xlmr-final-block-bio-head-v1",
            "labels": list(LABELS),
            "hidden_size": 8,
            "final_block_index": 1,
            "settings": {
                "max_length": 256,
                "threads": 2,
                "device": "cpu",
                "dtype": "float32",
                "offset_policy": "whitespace_edges_unowned_marks_v1",
                "decoder": "maximum_score_valid_BIO_path_no_gold",
                "encoder_trainable": "final_block_only",
                "encoder_mode": "eval",
                "block_dropout": "disabled_eval_mode",
            },
            "base_model": {
                "model_files_sha256": {
                    name: item.sha256 for name, item in files.items()
                },
                "config": self.config_data,
                "verified_download_identity": {
                    "repo": self.repo,
                    "revision": self.revision,
                    "receipt_sha256": digest(receipt_data),
                    "verification": "pinned_five_file_sizes_and_sha256_plus_receipt_metadata",
                },
            },
            "checkpoint_sha256": component_hash,
        }
        metadata.update(metadata_changes or {})
        metadata_data = json_bytes(metadata)
        (self.component / "block-head-config.json").write_bytes(metadata_data)
        self.identity = EncoderIdentity(
            repo=self.repo,
            revision=self.revision,
            base_directory=self.base_directory,
            base_files=files,
            receipt_path=self.receipt_path,
            receipt_sha256=digest(receipt_data),
            component_directory=self.component_directory,
            component_config_sha256=digest(metadata_data),
            component_sha256=component_hash,
            hidden_size=8,
            final_block_index=1,
            labels=list(LABELS),
        )
        return self.identity


class TinyEncoderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="nlp3-tiny-software-",
            dir=Path(tempfile.gettempdir()).resolve() / "opencode",
        )
        self.addCleanup(self.temporary.cleanup)
        self.fixture = TinyFixture(Path(self.temporary.name))
        self.tokenizer = FakeTokenizer()
        self.factory_calls = []

    def load(self):
        from messaging_nlp.encoder import _load_encoder

        def tokenizer_factory(*args, **kwargs):
            self.factory_calls.append((args, kwargs))
            return self.tokenizer

        return _load_encoder(
            self.fixture.root,
            self.fixture.identity,
            tokenizer_factory=tokenizer_factory,
        )

    def reject(self):
        # Import errors, signature errors, and unrelated failures do not count as
        # validation success. A successful positive load is a separate gate.
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            self.load()

    def modules(self, runtime):
        """Inspect assembled modules without imposing a public forward API."""
        roots = [
            value
            for value in vars(runtime._runtime).values()
            if isinstance(value, torch.nn.Module)
        ]
        modules = {id(module): module for root in roots for module in root.modules()}
        self.assertTrue(
            modules, "runtime must expose its assembled modules for this software check"
        )
        return list(modules.values())

    def test_full_forward_matches_independent_replacement_with_attention_mask(self):
        from transformers import XLMRobertaModel

        self.fixture.seal()
        runtime = self.load()
        modules = self.modules(runtime)
        encoders = [module for module in modules if isinstance(module, XLMRobertaModel)]
        heads = [
            module
            for module in modules
            if isinstance(module, torch.nn.Linear)
            and module.in_features == 8
            and module.out_features == len(LABELS)
        ]
        self.assertEqual(len(encoders), 1)
        self.assertEqual(len(heads), 1)
        ids = torch.tensor([[0, 4, 5, 2, 1], [0, 7, 2, 1, 1]])
        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
        with torch.inference_mode():
            expected = self.fixture.head(
                self.fixture.reference.roberta(
                    input_ids=ids, attention_mask=mask
                ).last_hidden_state
            )
            actual = heads[0](
                encoders[0](input_ids=ids, attention_mask=mask).last_hidden_state
            )
        self.assertTrue(torch.equal(actual, expected))
        for key, expected_tensor in self.fixture.final_state.items():
            torch.testing.assert_close(
                encoders[0].encoder.layer[-1].state_dict()[key], expected_tensor
            )
        for key, expected_tensor in self.fixture.head_state.items():
            torch.testing.assert_close(heads[0].state_dict()[key], expected_tensor)

    def test_cpu_eval_frozen_and_extract_inference_mode(self):
        self.fixture.seal()
        runtime = self.load()
        modules = self.modules(runtime)
        before = {
            (index, name): tensor.clone()
            for index, module in enumerate(modules)
            for name, tensor in module.state_dict().items()
        }
        calls = []

        def before_forward(module, args):
            calls.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))

        handles = []
        for module in modules:
            self.assertFalse(module.training)
            for parameter in module.parameters(recurse=False):
                self.assertEqual(parameter.device.type, "cpu")
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
            handles.append(module.register_forward_pre_hook(before_forward))
        try:
            artifact = runtime.extract("fictional-request", "Ada met")
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(
            calls, "extract must actually exercise the tiny assembled forward"
        )
        self.assertTrue(all(not grad and inference for grad, inference in calls))
        self.assertEqual(artifact.result.status, "ok")
        self.assertIsNone(artifact.result.failure)
        self.assertEqual(artifact.text, "Ada met")
        self.assertEqual(artifact.identity, self.fixture.identity)
        self.assertEqual(artifact.result.message_id, "fictional-request")
        self.assertEqual(artifact.result.input_sha256, digest(b"Ada met"))
        self.assertEqual(
            artifact.result.extractor_sha256, self.fixture.identity.digest()
        )
        self.assertEqual(len(self.tokenizer.calls), 1)
        self.assertEqual(self.tokenizer.calls[0][0], "Ada met")
        self.assertFalse(self.tokenizer.calls[0][1].get("truncation", False))
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual(torch.get_num_interop_threads(), 2)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        for index, module in enumerate(modules):
            for name, tensor in module.state_dict().items():
                self.assertTrue(torch.equal(tensor, before[index, name]))

    def test_masked_lm_safetensors_may_omit_only_verified_tied_aliases(self):
        del self.fixture.base_state["lm_head.decoder.weight"]
        del self.fixture.base_state["lm_head.decoder.bias"]
        self.fixture.write_base()
        self.fixture.seal()
        runtime = self.load()
        artifact = runtime.extract("fictional-request", "Ada met")
        self.assertEqual(artifact.result.status, "ok")

    def test_base_and_tokenizer_loading_are_local_safe_and_explicit(self):
        self.fixture.seal()
        with patch.object(
            XLMRobertaForMaskedLM,
            "from_pretrained",
            wraps=XLMRobertaForMaskedLM.from_pretrained,
        ) as load:
            self.load()
        self.assertEqual(load.call_count, 1)
        kwargs = load.call_args.kwargs
        self.assertIs(kwargs["local_files_only"], True)
        self.assertIs(kwargs["trust_remote_code"], False)
        self.assertIs(kwargs["use_safetensors"], True)
        self.assertIs(kwargs["output_loading_info"], True)
        self.assertEqual(kwargs["dtype"], torch.float32)
        self.assertEqual(kwargs["attn_implementation"], "eager")
        self.assertEqual(
            self.factory_calls[0][1],
            {
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )

    def test_missing_base_tensor_is_rejected(self):
        del self.fixture.base_state[
            "roberta.encoder.layer.0.attention.self.query.weight"
        ]
        self.fixture.write_base()
        self.fixture.seal()
        self.reject()

    def test_changed_eval_grad_or_parameter_state_fails_before_forward(self):
        self.fixture.seal()
        for kind in ("training", "grad", "inplace"):
            with self.subTest(kind=kind):
                runtime = self.load()
                head = runtime._runtime.head
                if kind == "training":
                    head.train()
                elif kind == "grad":
                    head.weight.requires_grad_(True)
                else:
                    with torch.no_grad():
                        head.weight.add_(1)
                artifact = runtime.extract("fictional-request", "Ada met")
                self.assertEqual(artifact.result.status, "failed")
                self.assertEqual(artifact.result.failure.code, "runtime_error")
                self.assertFalse(artifact.diagnostics["forward_performed"])

    def test_state_change_during_forward_is_a_preserved_failure(self):
        self.fixture.seal()
        runtime = self.load()

        def mutate(module, args, output):
            module.bias.add_(1)

        handle = runtime._runtime.head.register_forward_hook(mutate)
        try:
            artifact = runtime.extract("fictional-request", "Ada met")
        finally:
            handle.remove()
        self.assertEqual(artifact.result.status, "failed")
        self.assertEqual(artifact.result.failure.code, "runtime_error")
        self.assertIn("state changed", artifact.result.failure.message)
        self.assertEqual(len(self.tokenizer.calls), 1)

    def test_extra_encoder_tensor_is_rejected(self):
        self.fixture.base_state["roberta.unexpected.weight"] = torch.zeros(8)
        self.fixture.write_base()
        self.fixture.seal()
        self.reject()

    def test_missing_known_mlm_tensor_is_rejected(self):
        del self.fixture.base_state["lm_head.dense.weight"]
        self.fixture.write_base()
        self.fixture.seal()
        self.reject()

    def test_arbitrary_mlm_tensor_is_not_allowlisted_by_prefix(self):
        self.fixture.base_state["lm_head.arbitrary_unexpected"] = torch.zeros(8)
        self.fixture.write_base()
        self.fixture.seal()
        self.reject()

    def test_wrong_base_tensor_dimension_is_rejected(self):
        self.fixture.base_state["roberta.embeddings.word_embeddings.weight"] = (
            torch.zeros(31, 8)
        )
        self.fixture.write_base()
        self.fixture.seal()
        self.reject()

    def test_component_requires_exact_top_level_keys(self):
        self.fixture.component_state["optimizer"] = {}
        self.fixture.write_component()
        self.fixture.seal()
        self.reject()

    def test_missing_component_dictionary_is_rejected(self):
        del self.fixture.component_state["head"]
        self.fixture.write_component()
        self.fixture.seal()
        self.reject()

    def test_component_missing_state_key_is_rejected(self):
        del self.fixture.head_state["bias"]
        self.fixture.write_component()
        self.fixture.seal()
        self.reject()

    def test_component_extra_state_key_is_rejected(self):
        self.fixture.final_state["unexpected"] = torch.zeros(8)
        self.fixture.write_component()
        self.fixture.seal()
        self.reject()

    def test_component_dimension_mismatch_is_rejected(self):
        self.fixture.head_state["weight"] = torch.zeros(len(LABELS), 7)
        self.fixture.write_component()
        self.fixture.seal()
        self.reject()

    def test_metadata_semantics_are_checked_against_trusted_identity(self):
        mutations = [
            {"labels": list(reversed(LABELS))},
            {"hidden_size": 16},
            {"final_block_index": 0},
            {"base_model": {}},
            {"checkpoint_sha256": "0" * 64},
            {"format": "fictional-unsupported-format"},
        ]
        for changes in mutations:
            with self.subTest(changes=changes):
                self.fixture.seal(metadata_changes=changes)
                self.reject()

    def test_receipt_semantics_are_checked_even_when_receipt_hash_matches(self):
        for changes in (
            {"repo": "fictional/wrong"},
            {"revision": "wrong"},
            {"files": {}},
        ):
            with self.subTest(changes=changes):
                self.fixture.seal(receipt_changes=changes)
                self.reject()

    def test_every_asset_is_bound_to_expected_bytes(self):
        identity = self.fixture.seal()
        paths = [
            *(self.fixture.base / name for name in identity.base_files),
            self.fixture.component / "block-head-config.json",
            self.fixture.component / "block-head.pt",
            self.fixture.root / self.fixture.receipt_path,
        ]
        for path in paths:
            original = path.read_bytes()
            with self.subTest(path=path.relative_to(self.fixture.root)):
                try:
                    path.write_bytes(original + b"changed")
                    self.reject()
                finally:
                    path.write_bytes(original)

    def test_missing_asset_is_rejected(self):
        self.fixture.seal()
        path = self.fixture.base / "tokenizer.json"
        path.unlink()
        self.reject()

    def test_untrusted_metadata_rewrite_does_not_rebind_expected_identity(self):
        identity = self.fixture.seal()
        path = self.fixture.component / "block-head-config.json"
        value = json.loads(path.read_bytes())
        value["hidden_size"] = 16
        path.write_bytes(json_bytes(value))
        self.assertIs(self.fixture.identity, identity)
        self.reject()

    def test_unsafe_custom_configuration_is_rejected(self):
        for hook in (
            {"auto_map": {"AutoModel": "evil.CustomModel"}},
            {"trust_remote_code": True},
            {"model_type": "fictional-custom-model"},
            {"architectures": ["CustomRemoteModel"]},
        ):
            with self.subTest(hook=hook):
                original = copy.deepcopy(self.fixture.config_data)
                self.fixture.config_data.update(hook)
                self.fixture.write_base()
                self.fixture.seal()
                self.reject()
                self.fixture.config_data = original

    def test_component_deserialization_is_cpu_weights_only(self):
        self.fixture.seal()
        with patch.object(torch, "load", wraps=torch.load) as load:
            self.load()
        self.assertTrue(load.call_args_list)
        for call in load.call_args_list:
            self.assertIs(call.kwargs.get("weights_only"), True)
            self.assertEqual(str(call.kwargs.get("map_location")), "cpu")


if __name__ == "__main__":
    unittest.main()
