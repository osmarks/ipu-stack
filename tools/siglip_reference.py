#!/usr/bin/env python3
"""Generate deterministic SigLIP vision reference tensors with Hugging Face."""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import SiglipConfig, SiglipVisionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0x51_61_1A)
    parser.add_argument(
        "--checkpoints",
        choices=("none", "layers", "all"),
        default="layers",
    )
    return parser.parse_args()


def load_model(path: Path) -> tuple[SiglipVisionModel, SiglipConfig]:
    config = SiglipConfig.from_pretrained(path, local_files_only=True)
    model = SiglipVisionModel(config.vision_config)
    state = {}
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith("vision_model."):
                state[name] = source.get_tensor(name)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, config


def parameter_manifest(path: Path) -> dict:
    groups: dict[str, int] = {}
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as source:
        for name in source.keys():
            if not name.startswith("vision_model."):
                continue
            if ".encoder.layers." in name:
                layer = name.split(".encoder.layers.", 1)[1].split(".", 1)[0]
                group = f"encoder_layer_{int(layer):02d}"
            elif ".embeddings." in name:
                group = "embeddings"
            elif ".post_layernorm." in name:
                group = "post_layernorm"
            elif ".head." in name:
                group = "map_head"
            else:
                group = "other"
            groups[group] = groups.get(group, 0) + math.prod(source.get_slice(name).get_shape())
    return {
        "groups": {
            name: {"parameters": count, "fp16_bytes": count * 2}
            for name, count in sorted(groups.items())
        },
        "vision_parameters": sum(groups.values()),
        "vision_fp16_bytes": sum(groups.values()) * 2,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    model, config = load_model(args.model)
    vision = config.vision_config
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    pixels = torch.randn(
        args.batch_size,
        vision.num_channels,
        vision.image_size,
        vision.image_size,
        generator=generator,
        dtype=torch.float32,
    ) * 0.25

    tensors = {"pixel_values": pixels}
    hooks = []
    if args.checkpoints in ("layers", "all"):
        for index, layer in enumerate(model.vision_model.encoder.layers):
            hooks.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, index=index: tensors.__setitem__(
                        f"encoder_layer_{index:02d}", output.detach().clone()
                    )
                )
            )
    if args.checkpoints == "all":
        first_layer = model.vision_model.encoder.layers[0]
        hooks.extend(
            [
                model.vision_model.embeddings.register_forward_hook(
                    lambda _module, _inputs, output: tensors.__setitem__(
                        "patch_and_position", output.detach().clone()
                    )
                ),
                model.vision_model.post_layernorm.register_forward_hook(
                    lambda _module, _inputs, output: tensors.__setitem__(
                        "post_layernorm", output.detach().clone()
                    )
                ),
                first_layer.layer_norm1.register_forward_hook(
                    lambda _module, _inputs, output: tensors.__setitem__(
                        "encoder_layer_00_norm1", output.detach().clone()
                    )
                ),
                first_layer.self_attn.q_proj.register_forward_hook(
                    lambda _module, _inputs, output: tensors.__setitem__(
                        "encoder_layer_00_query", output.detach().clone()
                    )
                ),
                first_layer.self_attn.k_proj.register_forward_hook(
                    lambda _module, _inputs, output: tensors.__setitem__(
                        "encoder_layer_00_key", output.detach().clone()
                    )
                ),
                first_layer.self_attn.v_proj.register_forward_hook(
                    lambda _module, _inputs, output: tensors.__setitem__(
                        "encoder_layer_00_value", output.detach().clone()
                    )
                ),
            ]
        )

    with torch.inference_mode():
        output = model(pixel_values=pixels, interpolate_pos_encoding=False)
    for hook in hooks:
        hook.remove()
    if args.checkpoints == "all":
        heads = vision.num_attention_heads
        head_dimension = vision.hidden_size // heads
        query, key, value = (
            tensors[f"encoder_layer_00_{name}"]
            .reshape(args.batch_size, -1, heads, head_dimension)
            .transpose(1, 2)
            for name in ("query", "key", "value")
        )
        tensors["encoder_layer_00_attention_heads"] = (
            torch.nn.functional.scaled_dot_product_attention(query, key, value)
            .detach()
            .clone()
        )
        attention_hidden = (
            tensors["encoder_layer_00_attention_heads"]
            .transpose(1, 2)
            .reshape(args.batch_size, -1, vision.hidden_size)
        )
        tensors["encoder_layer_00_attention_residual"] = (
            tensors["patch_and_position"]
            + first_layer.self_attn.out_proj(attention_hidden)
        ).detach().clone()
    tensors["last_hidden_state"] = output.last_hidden_state
    tensors["pooler_output"] = output.pooler_output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = parameter_manifest(args.model)
    metadata.update(
        {
            "batch_size": args.batch_size,
            "image_size": vision.image_size,
            "patch_size": vision.patch_size,
            "patch_grid": vision.image_size // vision.patch_size,
            "sequence_length": (vision.image_size // vision.patch_size) ** 2,
            "hidden_size": vision.hidden_size,
            "intermediate_size": vision.intermediate_size,
            "layers": vision.num_hidden_layers,
            "heads": vision.num_attention_heads,
            "layer_norm_epsilon": vision.layer_norm_eps,
            "seed": args.seed,
            "preprocessing": "none; pixel_values are already-prepared model inputs",
        }
    )
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        args.output,
        metadata={"siglip": json.dumps(metadata, sort_keys=True)},
    )
    print(json.dumps({**metadata, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
