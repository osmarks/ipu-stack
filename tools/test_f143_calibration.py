"""Focused invariants for the reused historical calibration algorithms."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from quantize_siglip_f143 import (
    equalize_layernorm_group,
    project_f143,
    reconstruct_linear,
)


class CalibrationTests(unittest.TestCase):
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
