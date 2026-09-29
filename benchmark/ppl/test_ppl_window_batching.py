import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from benchmark.quant.eval_quant import compute_ppl


class _ExactLogitModel:
    """CPU fake whose rows are independent and exactly reproducible."""

    device = torch.device("cpu")

    def __init__(self, vocab=11):
        values = torch.arange(vocab * vocab, dtype=torch.float32)
        self.table = values.reshape(vocab, vocab) / 17.0
        self.call_shapes = []

    def __call__(self, *, input_ids, labels=None):
        self.call_shapes.append(tuple(input_ids.shape))
        logits = self.table[input_ids]
        out = SimpleNamespace(logits=logits)
        if labels is not None:
            shifted = torch.nn.functional.pad(
                labels, (0, 1), value=-100)[..., 1:].contiguous()
            out.loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                shifted.view(-1), ignore_index=-100, reduction="mean")
        return out


class PplWindowBatchingTest(unittest.TestCase):
    def setUp(self):
        self.ctx = 4
        self.enc = torch.tensor([
            0, 1, 2, 3,
            4, 5, 6, 7,
            7, 6, 5, 4,
        ], dtype=torch.long)

    def test_batched_losses_and_ppl_equal_historical_path(self):
        legacy_model = _ExactLogitModel()
        legacy_losses = []
        legacy = compute_ppl(
            legacy_model, self.enc, self.ctx, self.ctx,
            window_losses=legacy_losses, window_batch_size=1)

        batched_model = _ExactLogitModel()
        batched_losses = []
        batched = compute_ppl(
            batched_model, self.enc, self.ctx, self.ctx,
            window_losses=batched_losses, window_batch_size=2)

        self.assertEqual(legacy[:2], batched[:2])
        self.assertEqual(legacy_losses, batched_losses)
        self.assertEqual(legacy_model.call_shapes, [(1, 4)] * 3)
        self.assertEqual(batched_model.call_shapes, [(2, 4), (1, 4)])
        self.assertTrue(math.isfinite(batched[0]))

    def test_environment_opt_in(self):
        model = _ExactLogitModel()
        with mock.patch.dict(os.environ, {"PPL_WINDOW_BATCH_SIZE": "3"}):
            compute_ppl(model, self.enc, self.ctx, self.ctx)
        self.assertEqual(model.call_shapes, [(3, 4)])

    def test_rejects_overlap_or_partial_windows(self):
        with self.assertRaisesRegex(ValueError, "stride == ctx"):
            compute_ppl(
                _ExactLogitModel(), self.enc, self.ctx, 2,
                window_batch_size=2)
        with self.assertRaisesRegex(ValueError, "full ctx windows"):
            compute_ppl(
                _ExactLogitModel(), self.enc[:-1], self.ctx, self.ctx,
                window_batch_size=2)


if __name__ == "__main__":
    unittest.main()
