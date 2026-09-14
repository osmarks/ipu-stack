"""Calibration hooks must not survive a failed model forward."""

import unittest

import torch
from quantize_siglip_f143 import collect_block_hessians


class CalibrationFrontendTests(unittest.TestCase):
    def test_failed_forward_removes_observation_hooks(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)

            def forward(self, pixel_values, interpolate_pos_encoding):
                self.linear(pixel_values)
                raise RuntimeError("injected forward failure")

        model = Model()
        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
            collect_block_hessians(
                model, {"linear": model.linear}, [torch.ones(2, 3)], 2
            )
        self.assertFalse(model.linear._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
