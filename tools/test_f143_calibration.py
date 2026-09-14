"""Focused invariants for the reused historical calibration algorithms."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from f143 import (
    accumulate_input_moments,
    block_hessians,
    equalize_layernorm_group,
    f143_tensor_scale,
    project_f143,
    reconstruct_linear,
    scale_for_maximum,
)


class CalibrationTests(unittest.TestCase):
    def test_block_moments_match_dense_reference_with_partial_batches(self):
        torch.manual_seed(3)
        x = torch.randn(2, 5, 21, requires_grad=True)
        hessians = block_hessians(21, 8, x.device)
        total = torch.zeros(21)
        for batch in x:
            rows = accumulate_input_moments(batch, hessians, total)
            self.assertFalse(rows.requires_grad)
        flat = x.detach().reshape(-1, 21)
        self.assertTrue(torch.allclose(total, flat.sum(0), atol=1e-6))
        for start, hessian in zip(range(0, 21, 8), hessians):
            part = flat[:, start : start + 8]
            self.assertTrue(torch.allclose(hessian, part.T @ part, atol=2e-6))
        for maximum in [0, 240, 241, 120, 1e-20, 1e20]:
            self.assertEqual(
                scale_for_maximum(maximum),
                f143_tensor_scale(torch.tensor([maximum], dtype=torch.float64)),
            )

    def test_gptq_keeps_one_tensor_scale_with_partial_hessian_blocks(self):
        torch.manual_seed(1)
        x = torch.randn(40, 21)
        weight = torch.randn(13, 21) * 0.1
        module = SimpleNamespace(
            weight=weight.clone(), bias=torch.zeros(13), in_features=21, out_features=13
        )
        hessians = [x[:, i : i + 8].T @ x[:, i : i + 8] for i in range(0, 21, 8)]
        reconstruct_linear(
            module, hessians, 8, 0.01, "gptq", x.mean(0), tensor_scale=-8
        )
        self.assertTrue(torch.equal(module.weight, project_f143(module.weight, -8)))
        self.assertTrue(
            torch.allclose(module.bias, (weight - module.weight) @ x.mean(0))
        )

    def test_equalization_preserves_layernorm_and_linear_composition(self):
        torch.manual_seed(2)
        x = torch.randn(9, 11)
        norm = SimpleNamespace(weight=torch.randn(11), bias=torch.randn(11))
        consumer = SimpleNamespace(weight=torch.randn(7, 11))

        def output():
            return F.layer_norm(x, (11,), norm.weight, norm.bias) @ consumer.weight.T

        expected = output()
        equalize_layernorm_group(
            norm, [consumer], torch.rand(11) + 0.01, 8, 16.0, tensor_scales=True
        )
        self.assertTrue(torch.allclose(expected, output(), atol=2e-6, rtol=2e-6))


if __name__ == "__main__":
    unittest.main()
