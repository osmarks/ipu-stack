#!/usr/bin/env python3
"""Reconstruct SigLIP encoder weights for IPU21 F143 storage."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from f143 import (
    accumulate_input_moments,
    block_hessians,
    equalize_layernorm_group,
    f143_tensor_scale,
    reconstruct_linear,
)
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoImageProcessor, SiglipConfig, SiglipVisionModel

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
        default=[],
        help="SafeTensors file containing pixel_values; may be repeated",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Image processed with the model's official processor; may be repeated",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--scale-granularity", choices=("block", "tensor"), default="block")
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--algorithm", choices=("nearest", "gptq"), default="gptq")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--no-bias-correction", action="store_false", dest="bias_correction"
    )
    parser.add_argument(
        "--sequential-layers",
        action="store_true",
        help="recollect each layer after reconstructing its predecessors",
    )
    parser.add_argument(
        "--equalize-layernorm",
        action="store_true",
        help="fold activation-aware channel scales into LayerNorm and its consumers",
    )
    parser.add_argument("--equalization-scale-limit", type=float, default=16.0)
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
    if not hasattr(model, "vision_model"):
        state = {name.removeprefix("vision_model."): value for name, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, config


def calibration_pixels(
    paths: list[Path], image_paths: list[Path], model_path: Path
) -> list[torch.Tensor]:
    batches = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as source:
            batches.append(source.get_tensor("pixel_values").float())
    if image_paths:
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        batches.append(processor(images=images, return_tensors="pt")["pixel_values"].float())
    return batches


def encoder_linears(model: SiglipVisionModel, layers: int) -> dict[str, torch.nn.Linear]:
    selected = {}
    vision = getattr(model, "vision_model", model)
    for name, module in vision.named_modules():
        name = "vision_model." + name
        if not isinstance(module, torch.nn.Linear) or not name.startswith("vision_model.encoder.layers."):
            continue
        layer = int(name.split(".")[3])
        if layer < layers and name.endswith(LINEAR_SUFFIXES):
            selected[name] = module
    return selected


def run_calibration(model, batches, hooks):
    try:
        with torch.inference_mode():
            for pixels in batches:
                model(pixel_values=pixels.to(next(model.parameters()).device),
                      interpolate_pos_encoding=False)
    finally:
        for hook in hooks:
            hook.remove()


def collect_block_hessians(
    model: SiglipVisionModel,
    modules: dict[str, torch.nn.Linear],
    batches: list[torch.Tensor],
    block_size: int,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor], dict[str, int]]:
    hessians = {
        name: block_hessians(module.in_features, block_size, module.weight.device)
        for name, module in modules.items()
    }
    sums = {
        name: torch.zeros(module.in_features, device=module.weight.device)
        for name, module in modules.items()
    }
    counts = dict.fromkeys(modules, 0)
    hooks = []
    for name, module in modules.items():
        def accumulate(_module, inputs, name=name):
            values = accumulate_input_moments(inputs[0], hessians[name], sums[name])
            counts[name] += values.shape[0]

        hooks.append(module.register_forward_pre_hook(accumulate))
    run_calibration(model, batches, hooks)
    return hessians, sums, counts


def layernorm_consumer_groups(
    model: SiglipVisionModel, layers: int
) -> list[tuple[str, torch.nn.LayerNorm, list[torch.nn.Linear]]]:
    groups = []
    for index, layer in enumerate(getattr(model, "vision_model", model).encoder.layers[:layers]):
        groups.append(
            (
                f"encoder_layer_{index:02}.norm1_qkv",
                layer.layer_norm1,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            )
        )
        groups.append(
            (f"encoder_layer_{index:02}.norm2_fc1", layer.layer_norm2, [layer.mlp.fc1])
        )
    return groups


def equalize_layernorm_consumers(
    model: SiglipVisionModel,
    batches: list[torch.Tensor],
    layers: int,
    block_size: int,
    scale_limit: float,
    tensor_scales: bool = False,
) -> dict[str, dict[str, float]]:
    groups = layernorm_consumer_groups(model, layers)
    second_moments = {
        name: torch.zeros(consumers[0].in_features, device=consumers[0].weight.device)
        for name, _norm, consumers in groups
    }
    counts = dict.fromkeys(second_moments, 0)
    hooks = []
    for name, _norm, consumers in groups:
        def accumulate(_module, inputs, name=name):
            values = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            second_moments[name].add_(values.square().sum(dim=0))
            counts[name] += values.shape[0]

        hooks.append(consumers[0].register_forward_pre_hook(accumulate))
    run_calibration(model, batches, hooks)

    report = {}
    with torch.no_grad():
        for name, norm, consumers in groups:
            moment = second_moments.pop(name) / counts.pop(name)
            report[name] = equalize_layernorm_group(norm, consumers, moment, block_size, scale_limit, tensor_scales)
    return report


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


def reconstruct_modules(
    model: SiglipVisionModel,
    modules: dict[str, torch.nn.Linear],
    batches: list[torch.Tensor],
    args: argparse.Namespace,
    objectives: dict,
    progress: list[int],
) -> None:
    hessians, input_sums, input_counts = collect_block_hessians(
        model, modules, batches, args.block_size
    )
    with torch.no_grad():
        for name, module in modules.items():
            input_mean = (
                input_sums.pop(name) / input_counts.pop(name)
                if args.bias_correction
                else None
            )
            nearest, reconstructed, correction_rms = reconstruct_linear(
                module,
                hessians.pop(name),
                args.block_size,
                args.damp,
                args.algorithm,
                input_mean,
                tensor_scale=(
                    f143_tensor_scale(module.weight)
                    if args.scale_granularity == "tensor" else None
                ),
            )
            objectives[name] = {
                "nearest": nearest,
                "reconstructed": reconstructed,
                "ratio": reconstructed / nearest if nearest else 0.0,
                "bias_correction_rms": correction_rms,
            }
            progress[0] += 1
            print(
                f"[{progress[0]}/{progress[1]}] reconstructed {name} "
                f"(weighted error {objectives[name]['ratio']:.4f}x)",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.block_size <= 0 or args.damp < 0.0 or args.equalization_scale_limit < 1.0:
        raise ValueError(
            "block size must be positive, damping non-negative, and scale limit at least one"
        )

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
    if not args.calibration and not args.image:
        raise ValueError("at least one --calibration or --image input is required")
    batches = calibration_pixels(args.calibration, args.image, args.model)
    modules = encoder_linears(model, layer_count)

    with torch.inference_mode():
        references = [
            model(pixel_values=pixels.to(device), interpolate_pos_encoding=False)
            .last_hidden_state.clone()
            for pixels in batches
        ]
    equalization = (
        equalize_layernorm_consumers(
            model,
            batches,
            layer_count,
            args.block_size,
            args.equalization_scale_limit,
            tensor_scales=args.scale_granularity == "tensor",
        )
        if args.equalize_layernorm
        else {}
    )
    with torch.inference_mode():
        equalization_metrics = [
            output_metrics(
                reference,
                model(pixel_values=pixels.to(device), interpolate_pos_encoding=False)
                .last_hidden_state,
            )
            for pixels, reference in zip(batches, references)
        ]
    objectives = {}
    progress = [0, len(modules)]
    if args.sequential_layers:
        for layer in range(layer_count):
            marker = f"vision_model.encoder.layers.{layer}."
            layer_modules = {
                name: module for name, module in modules.items() if name.startswith(marker)
            }
            reconstruct_modules(model, layer_modules, batches, args, objectives, progress)
    else:
        reconstruct_modules(model, modules, batches, args, objectives, progress)

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
        "scale_granularity": args.scale_granularity,
        "calibration": [str(path) for path in args.calibration],
        "images": [str(path) for path in args.image],
        "damp": args.damp,
        "bias_correction": args.bias_correction,
        "sequential_layers": args.sequential_layers,
        "equalize_layernorm": args.equalize_layernorm,
        "equalization_scale_limit": args.equalization_scale_limit,
        "equalization_metrics": equalization_metrics,
        "equalization": equalization,
        "layers": layer_count,
        "metrics": metrics,
        "objectives": objectives,
    }
    report_text = json.dumps(report, indent=2) + "\n"
    if not args.no_save:
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.model / "config.json", args.output / "config.json")
        save_file(
            {
                (name if name.startswith("vision_model.") else "vision_model." + name): value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            args.output / "model.safetensors",
            metadata={"format": "pt", "ipu_f143_reconstruction": args.algorithm},
        )
        (args.output / "quantization.json").write_text(report_text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "objectives"
            }
            | {
                "objective_ratio": sum(
                    objective["reconstructed"] for objective in objectives.values()
                )
                / sum(objective["nearest"] for objective in objectives.values())
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
