"""Stage5 header-derived compatibility requirements, fictional tensors only."""

import tempfile
import unittest
from pathlib import Path

import torch
from test_encoder_tiny import FakeTokenizer, TinyFixture


class PoolerCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="nlp3-pooler-software-",
            dir=Path(tempfile.gettempdir()).resolve() / "opencode",
        )
        self.addCleanup(temporary.cleanup)
        self.fixture = TinyFixture(Path(temporary.name))
        self.pooler()

    def pooler(self):
        self.fixture.base_state.update(
            {
                "roberta.pooler.dense.weight": torch.eye(8),
                "roberta.pooler.dense.bias": torch.zeros(8),
            }
        )

    def load(self):
        from messaging_nlp.encoder import _load_encoder

        self.fixture.write_base()
        self.fixture.seal()
        return _load_encoder(
            self.fixture.root,
            self.fixture.identity,
            tokenizer_factory=lambda *args, **kwargs: FakeTokenizer(),
        )

    def test_known_unused_pair_and_verified_omitted_aliases_preserve_forward(self):
        del self.fixture.base_state["lm_head.decoder.weight"]
        del self.fixture.base_state["lm_head.decoder.bias"]
        runtime = self.load()
        ids = torch.tensor([[0, 4, 5, 2]])
        mask = torch.tensor([[1, 1, 1, 1]])
        with torch.inference_mode():
            expected = self.fixture.head(
                self.fixture.reference.roberta(
                    input_ids=ids, attention_mask=mask
                ).last_hidden_state
            )[0]
        actual = torch.tensor(
            runtime._runtime.forward(ids[0].tolist(), mask[0].tolist())
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(runtime._runtime.base.pooler)

    def test_partial_pooler_is_not_a_compatible_checkpoint(self):
        del self.fixture.base_state["roberta.pooler.dense.bias"]
        with self.assertRaises(ValueError):
            self.load()

    def test_pooler_wrong_shape_is_rejected(self):
        self.fixture.base_state["roberta.pooler.dense.weight"] = torch.zeros(7, 8)
        with self.assertRaises(ValueError):
            self.load()

    def test_pooler_dtype_is_checked_before_unused_tensor_is_excluded(self):
        self.fixture.base_state["roberta.pooler.dense.bias"] = torch.zeros(
            8, dtype=torch.float64
        )
        with self.assertRaises(ValueError):
            self.load()

    def test_pooler_nonfinite_values_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                self.pooler()
                self.fixture.base_state["roberta.pooler.dense.bias"][0] = value
                with self.assertRaises(ValueError):
                    self.load()

    def test_unrelated_pooler_prefix_does_not_receive_an_exception(self):
        self.fixture.base_state["roberta.pooler.other.weight"] = torch.zeros(8)
        with self.assertRaises(ValueError):
            self.load()

    def test_missing_encoder_weight_is_still_rejected_with_valid_pooler(self):
        del self.fixture.base_state[
            "roberta.encoder.layer.0.attention.self.query.weight"
        ]
        with self.assertRaises(ValueError):
            self.load()

    def test_missing_original_is_not_repaired_from_a_tied_alias(self):
        del self.fixture.base_state["roberta.embeddings.word_embeddings.weight"]
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
