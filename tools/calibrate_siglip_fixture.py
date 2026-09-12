#!/usr/bin/env python3
"""Measure tensor-wide FP8 scales and historical GPTQ on an exported SigLIP fixture.

Scales may be independent or shared between operands and repeated positions.
Accumulation and attention remain FP32 in this numerical probe. This does not simulate IPU accumulation.
The Hessian block size controls reconstruction work, never scale granularity.
"""

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from quantize_siglip_f143 import (
    equalize_layernorm_group,
    project_f143,
    reconstruct_linear,
)
from safetensors.torch import save_file


def scale_for(maximum):
    return max(-32, min(31, math.ceil(math.log2(maximum / 240)))) if maximum else 0


class Siglip:
    def __init__(self, root, device, block):
        self.root, self.device, self.block = root, device, block
        self.manifest = json.loads((root / "manifest.json").read_text())
        config = self.manifest["config"]
        if any(
            config[k] != v
            for k, v in {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "patch_size": 14,
            }.items()
        ):
            raise ValueError("this calibration frontend requires SigLIP So400m/14")
        self.parameters = {
            k.removeprefix("vit."): self.load(e)
            for k, e in self.manifest["parameters"].items()
        }
        self.stats, self.scales, self.clipping = {}, {}, {}
        self.collect, self.quantize = False, False
        self.quantize_names = None
        self.fp16_embedding = False

    def load(self, entry):
        return torch.from_numpy(
            np.fromfile(self.root / entry["file"], dtype="<f4").reshape(entry["shape"])
        ).to(self.device)

    def dense(self, x, name):
        weight, bias = (
            self.parameters[name + ".weight"],
            self.parameters[name + ".bias"],
        )
        if self.collect:
            rows = x.reshape(-1, x.shape[-1]).float()
            if name not in self.stats:
                self.stats[name] = {
                    "maximum": 0.0,
                    "count": 0,
                    "sum": torch.zeros(rows.shape[1], device=self.device),
                    "hessian": [
                        torch.zeros(
                            (min(self.block, rows.shape[1] - i),) * 2,
                            device=self.device,
                        )
                        for i in range(0, rows.shape[1], self.block)
                    ],
                }
            stats = self.stats[name]
            stats["maximum"] = max(stats["maximum"], rows.abs().max().item())
            stats["count"] += len(rows)
            stats["sum"].add_(rows.sum(0))
            for i, h in enumerate(stats["hessian"]):
                part = rows[:, i * self.block : (i + 1) * self.block]
                h.addmm_(part.T, part)
        if self.quantize and (
            self.quantize_names is None or name in self.quantize_names
        ):
            scale = self.scales[name]["activation"]
            clipped = (x.abs() > 240 * 2.0**scale).sum().item()
            self.clipping[name] = self.clipping.get(name, 0) + int(clipped)
            x = project_f143(x, scale)
        if self.quantize and self.fp16_embedding and name == "embedding":
            return (
                (x.half().float() @ weight.half().float() + bias.half().float())
                .half()
                .float()
            )
        return x @ weight + bias

    def norm(self, x, name):
        return F.layer_norm(
            x,
            (1152,),
            self.parameters[name + ".scale"].flatten(),
            self.parameters[name + ".bias"].flatten(),
            1e-6,
        )

    def mlp(self, x, name):
        return self.dense(
            F.gelu(self.dense(x, name + ".up"), approximate="tanh"), name + ".down"
        )

    def attention(self, q, k, v, name):
        def split(x):
            return x.reshape(1, -1, 16, 72).transpose(1, 2)

        x = (
            F.scaled_dot_product_attention(split(q), split(k), split(v))
            .transpose(1, 2)
            .reshape(1, -1, 1152)
        )
        return self.dense(x, name + ".output")

    def forward(self, case):
        x = (
            self.dense(self.load(case["inputs"]["vit.image.patches"]), "embedding")
            + self.parameters["position"]
        )
        for i in range(27):
            name = f"encoder.layer{i}"
            q, k, v = self.dense(
                self.norm(x, name + ".attention_norm"), name + ".attention.qkv"
            ).chunk(3, -1)
            x = x + self.attention(q, k, v, name + ".attention")
            x = x + self.mlp(self.norm(x, name + ".mlp_norm"), name + ".mlp")
        x = self.norm(x, "encoder.final_norm")
        q = self.dense(self.parameters["map.probe"], "map.attention.query")
        k, v = self.dense(x, "map.attention.kv").chunk(2, -1)
        x = self.attention(q, k, v, "map.attention")
        return x + self.mlp(self.norm(x, "map.norm"), "map.mlp")

    def evaluate(self, cases):
        report = []
        for case in cases:
            self.clipping = {}
            actual = self.forward(case).flatten().double()
            expected = self.load(case["expected"]).flatten().double()
            item = {
                "name": case["name"],
                "cosine": F.cosine_similarity(actual, expected, dim=0).item(),
                "maximum_absolute_error": (actual - expected).abs().max().item(),
                "clipped_activation_elements": self.clipping.copy(),
            }
            report.append(item)
            print(
                f"{case['name']}: cosine={item['cosine']:.9f} clipped={sum(self.clipping.values())}",
                flush=True,
            )
        return report


def save_weights(model, path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "vit." + name: value.detach().cpu().contiguous()
                for name, value in model.parameters.items()
            },
            path,
            metadata={
                "format": "pt",
                "layout": "ipu-graph-logical",
                "scale_granularity": "tensor",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--output-weights",
        type=Path,
        help="Save reconstructed logical parameters as FP32 SafeTensors",
    )
    parser.add_argument("--calibration-count", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nearest-only", action="store_true")
    parser.add_argument("--equalize-layernorm", action="store_true")
    parser.add_argument("--fp16-embedding", action="store_true")
    parser.add_argument(
        "--scale-sharing", choices=("tensor", "repeat-role"), default="tensor"
    )
    parser.add_argument("--shared-operand-scale", action="store_true")
    args = parser.parse_args()
    if args.block_size <= 0 or args.damp < 0:
        raise ValueError("block size must be positive and damping nonnegative")
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Siglip(args.fixture, args.device, args.block_size)
    cases = model.manifest["cases"]
    if not 0 < args.calibration_count < len(cases):
        raise ValueError("reserve at least one held-out image")
    report = {
        "calibration": [c["name"] for c in cases[: args.calibration_count]],
        "held_out": [c["name"] for c in cases[args.calibration_count :]],
        "accumulation": "FP32",
        "block_size": args.block_size,
        "scale_granularity": "tensor",
    }
    print("FP32 mapped graph", flush=True)
    report["fp32"] = model.evaluate(cases)
    assert min(r["cosine"] for r in report["fp32"]) > 0.99999
    model.collect = True
    for case in cases[: args.calibration_count]:
        model.forward(case)
    model.collect = False
    if args.equalize_layernorm:
        report["equalization"] = {}
        for i in range(27):
            for norm_suffix, linear_suffix in [
                ("attention_norm", "attention.qkv"),
                ("mlp_norm", "mlp.up"),
            ]:
                norm_name, name = (
                    f"encoder.layer{i}." + norm_suffix,
                    f"encoder.layer{i}." + linear_suffix,
                )
                stats = model.stats[name]
                moment = (
                    torch.cat([h.diagonal() for h in stats["hessian"]]) / stats["count"]
                )
                norm = SimpleNamespace(
                    weight=model.parameters[norm_name + ".scale"],
                    bias=model.parameters[norm_name + ".bias"],
                )
                consumer = SimpleNamespace(weight=model.parameters[name + ".weight"].T)
                report["equalization"][name] = equalize_layernorm_group(
                    norm, [consumer], moment, args.block_size, 16.0, tensor_scales=True
                )
        print("Equalized FP32 graph", flush=True)
        report["equalized_fp32"] = model.evaluate(cases)
        assert min(r["cosine"] for r in report["equalized_fp32"]) > 0.99999
        model.stats.clear()
        model.collect = True
        for case in cases[: args.calibration_count]:
            model.forward(case)
        model.collect = False
    for name, stats in model.stats.items():
        model.scales[name] = {
            "activation": scale_for(stats["maximum"]),
            "weight": scale_for(model.parameters[name + ".weight"].abs().max().item()),
            "activation_maximum": stats["maximum"],
        }
    if args.scale_sharing == "repeat-role":
        groups = {}
        for name, scales in model.scales.items():
            role = re.sub(r"encoder\.layer\d+\.", "encoder.body.", name)
            groups.setdefault(role, []).append(scales)
        for members in groups.values():
            activation = max(s["activation"] for s in members)
            weight = max(s["weight"] for s in members)
            for scales in members:
                scales.update(activation=activation, weight=weight)
    if args.shared_operand_scale:
        for scales in model.scales.values():
            scale = max(scales["activation"], scales["weight"])
            scales.update(activation=scale, weight=scale)
    report["scale_sharing"] = args.scale_sharing
    report["shared_operand_scale"] = args.shared_operand_scale
    report["scales"] = model.scales
    report["fp16_embedding"] = args.fp16_embedding
    model.fp16_embedding = args.fp16_embedding
    if args.fp16_embedding:
        model.quantize_names = set(model.scales) - {"embedding"}
    originals = {
        name: value.clone()
        for name, value in model.parameters.items()
        if name.endswith((".weight", ".bias"))
    }
    for name, scales in model.scales.items():
        if args.fp16_embedding and name == "embedding":
            continue
        model.parameters[name + ".weight"].copy_(
            project_f143(originals[name + ".weight"], scales["weight"])
        )
    model.quantize = True
    print("Tensor scales, nearest weights", flush=True)
    report["nearest"] = model.evaluate(cases)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.nearest_only:
        save_weights(model, args.output_weights)
        return
    for name, value in originals.items():
        model.parameters[name].copy_(value)
    report["reconstruction"] = {}
    for name, stats in model.stats.items():
        if args.fp16_embedding and name == "embedding":
            continue
        weight = model.parameters[name + ".weight"]
        module = SimpleNamespace(
            weight=weight.T,
            bias=model.parameters[name + ".bias"].flatten(),
            in_features=weight.shape[0],
            out_features=weight.shape[1],
        )
        nearest, rebuilt, correction = reconstruct_linear(
            module,
            stats["hessian"],
            args.block_size,
            args.damp,
            "gptq",
            stats["sum"] / stats["count"],
            tensor_scale=model.scales[name]["weight"],
        )
        report["reconstruction"][name] = {
            "nearest_objective": nearest,
            "gptq_objective": rebuilt,
            "bias_correction_rms": correction,
        }
        print(f"GPTQ {name}: {rebuilt / nearest:.5f} weighted error ratio", flush=True)
    print("Tensor scales, GPTQ weights and bias correction", flush=True)
    report["gptq"] = model.evaluate(cases)
    save_weights(model, args.output_weights)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
