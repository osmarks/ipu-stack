#!/usr/bin/env python3
"""Reconstruct SigLIP encoder weights for IPU21 F143 storage."""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import SiglipConfig, SiglipVisionModel


LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--calibration",
        type=Path,
        action="append",
        required=True,
        help="SafeTensors file containing pixel_values; may be repeated",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--algorithm", choices=("nearest", "gptq"), default="gptq")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def load_model(path: Path) -> tuple[SiglipVisionModel, SiglipConfig]:
    config = SiglipConfig.from_pretrained(path, local_files_only=True)
    model = SiglipVisionModel(config.vision_config)
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as source:
        state = {
            name: source.get_tensor(name)
            for name in source.keys()
            if name.startswith("vision_model.")
        }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, config


def calibration_pixels(paths: list[Path]) -> list[torch.Tensor]:
    batches = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as source:
            batches.append(source.get_tensor("pixel_values").float())
    return batches


def encoder_linears(model: SiglipVisionModel, layers: int) -> dict[str, torch.nn.Linear]:
    selected = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith("vision_model.encoder.layers."):
            continue
        layer = int(name.split(".")[3])
        if layer < layers and name.endswith(LINEAR_SUFFIXES):
            selected[name] = module
    return selected


def collect_block_hessians(
    model: SiglipVisionModel,
    modules: dict[str, torch.nn.Linear],
    batches: list[torch.Tensor],
    block_size: int,
) -> dict[str, list[torch.Tensor]]:
    hessians = {
        name: [
            torch.zeros(
                (min(block_size, module.in_features - start),) * 2,
                device=module.weight.device,
            )
            for start in range(0, module.in_features, block_size)
        ]
        for name, module in modules.items()
    }
    hooks = []
    for name, module in modules.items():
        def accumulate(_module, inputs, name=name):
            values = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            for index, start in enumerate(range(0, values.shape[1], block_size)):
                block = values[:, start : start + block_size]
                hessians[name][index].addmm_(block.T, block)

        hooks.append(module.register_forward_pre_hook(accumulate))
    with torch.inference_mode():
        for pixels in batches:
            model(pixel_values=pixels.to(next(model.parameters()).device), interpolate_pos_encoding=False)
    for hook in hooks:
        hook.remove()
    return hessians


def f143_scales_by_row_block(values: torch.Tensor, block_size: int) -> torch.Tensor:
    rows = values.shape[0]
    padded_rows = math.ceil(rows / block_size) * block_size
    if padded_rows != rows:
        values = torch.nn.functional.pad(values, (0, 0, 0, padded_rows - rows))
    maximum = values.abs().reshape(-1, block_size, values.shape[1]).amax(dim=(1, 2))
    scales = torch.where(
        maximum == 0.0,
        torch.zeros_like(maximum),
        torch.ceil(torch.log2(maximum / 240.0)),
    ).clamp(-32.0, 31.0)
    return scales.repeat_interleave(block_size)[:rows]


def project_f143(values: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    factor = torch.exp2(torch.as_tensor(-scale, device=values.device, dtype=values.dtype))
    magnitude = values.abs() * factor
    subnormal = torch.round(magnitude * 1024.0).clamp(max=8.0) * 2.0**-10
    exponent = torch.floor(torch.log2(magnitude.clamp_min(torch.finfo(torch.float32).tiny)))
    unit = torch.exp2(exponent)
    mantissa = torch.round((magnitude / unit - 1.0) * 8.0)
    carry = mantissa == 8.0
    exponent += carry
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    normal = (1.0 + mantissa / 8.0) * torch.exp2(exponent)
    projected = torch.where(magnitude < 2.0**-7, subnormal, normal).clamp(max=240.0)
    return torch.copysign(projected / factor, values)


def inverse_hessian_factor(hessian: torch.Tensor, damp: float) -> torch.Tensor:
    hessian = hessian.double()
    diagonal_mean = torch.diagonal(hessian).mean()
    hessian.diagonal().add_(max(torch.finfo(torch.float64).eps, damp * diagonal_mean.item()))
    try:
        return torch.linalg.cholesky(torch.linalg.inv(hessian), upper=True).float()
    except torch.linalg.LinAlgError:
        return torch.diag(torch.diagonal(hessian).rsqrt()).float()


def gptq_block(
    weight: torch.Tensor, inverse_factor: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    width = weight.shape[1]

    working = weight.float().clone()
    output = torch.empty_like(working)
    for column in range(width):
        quantized = project_f143(working[:, column], scales)
        output[:, column] = quantized
        divisor = inverse_factor[column, column].clamp_min(torch.finfo(torch.float32).eps)
        error = (working[:, column] - quantized) / divisor
        working[:, column:].sub_(error[:, None] * inverse_factor[column, column:][None, :])
    return output


def reconstruct_linear(
    module: torch.nn.Linear,
    hessians: list[torch.Tensor],
    block_size: int,
    damp: float,
    algorithm: str,
) -> tuple[float, float]:
    original = module.weight.detach()
    reconstructed = torch.empty_like(original)
    nearest_objective = 0.0
    reconstructed_objective = 0.0
    for input_index, input_start in enumerate(range(0, module.in_features, block_size)):
        input_end = min(input_start + block_size, module.in_features)
        hessian = hessians[input_index]
        inverse_factor = (
            inverse_hessian_factor(hessian, damp) if algorithm == "gptq" else None
        )
        block = original[:, input_start:input_end]
        scales = f143_scales_by_row_block(block, block_size)
        nearest = project_f143(block, scales[:, None])
        rebuilt = (
            gptq_block(block, inverse_factor, scales)
            if inverse_factor is not None
            else nearest
        )
        reconstructed[:, input_start:input_end] = rebuilt
        nearest_error = block.float() - nearest
        rebuilt_error = block.float() - rebuilt
        nearest_objective += torch.sum((nearest_error @ hessian) * nearest_error).item()
        reconstructed_objective += torch.sum(
            (rebuilt_error @ hessian) * rebuilt_error
        ).item()
    module.weight.copy_(reconstructed)
    return nearest_objective, reconstructed_objective


def output_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    error = (actual - reference).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(actual.shape[0], -1), reference.reshape(reference.shape[0], -1)
    ).mean()
    return {
        "mean_absolute_error": error.mean().item(),
        "maximum_absolute_error": error.max().item(),
        "cosine_similarity": cosine.item(),
    }


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.block_size <= 0 or args.damp < 0.0:
        raise ValueError("block size must be positive and damping must be non-negative")

    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, config = load_model(args.model)
    model.to(device)
    layer_count = args.layers or config.vision_config.num_hidden_layers
    if not 1 <= layer_count <= config.vision_config.num_hidden_layers:
        raise ValueError("layer count is outside the model")
    batches = calibration_pixels(args.calibration)
    modules = encoder_linears(model, layer_count)

    with torch.inference_mode():
        references = [
            model(pixel_values=pixels.to(device), interpolate_pos_encoding=False)
            .last_hidden_state.clone()
            for pixels in batches
        ]
    hessians = collect_block_hessians(model, modules, batches, args.block_size)
    objectives = {}
    with torch.no_grad():
        for index, (name, module) in enumerate(modules.items(), 1):
            nearest, reconstructed = reconstruct_linear(
                module, hessians.pop(name), args.block_size, args.damp, args.algorithm
            )
            objectives[name] = {
                "nearest": nearest,
                "reconstructed": reconstructed,
                "ratio": reconstructed / nearest if nearest else 0.0,
            }
            print(
                f"[{index}/{len(modules)}] reconstructed {name} "
                f"(weighted error {objectives[name]['ratio']:.4f}x)",
                flush=True,
            )

    with torch.inference_mode():
        metrics = [
            output_metrics(
                reference,
                model(pixel_values=pixels.to(device), interpolate_pos_encoding=False)
                .last_hidden_state,
            )
            for pixels, reference in zip(batches, references)
        ]

    report = {
        "algorithm": f"block-diagonal-f143-{args.algorithm}",
        "block_size": args.block_size,
        "calibration": [str(path) for path in args.calibration],
        "damp": args.damp,
        "layers": layer_count,
        "metrics": metrics,
        "objectives": objectives,
    }
    if not args.no_save:
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.model / "config.json", args.output / "config.json")
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            args.output / "model.safetensors",
            metadata={"format": "pt", "ipu_f143_reconstruction": args.algorithm},
        )
        (args.output / "quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
