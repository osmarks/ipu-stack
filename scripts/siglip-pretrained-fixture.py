#!/usr/bin/env python3
"""Export official SigLIP vision weights, real images and independent FP32 embeddings.

Dependencies: torch, transformers, safetensors, huggingface_hub, Pillow, numpy.
The output is consumed by ipu-trivial-test --reference-fixture DIRECTORY.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

MODEL = "google/siglip-so400m-patch14-384"
REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
IMAGES = [
    "authors.jpg",
    "siglip.jpg",
    "caffeine.jpg",
    "robosign.jpg",
    "fried_fish.jpeg",
    "cow_beach2.jpg",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    root = args.output
    (root / "parameters").mkdir(parents=True, exist_ok=True)
    checkpoint = hf_hub_download(MODEL, "model.safetensors", revision=REVISION)
    with safe_open(checkpoint, framework="pt") as f:
        names = f.keys()
        state = {k: f.get_tensor(k) for k in names if k.startswith("vision_model.")}
    config = SiglipVisionConfig.from_pretrained(MODEL, revision=REVISION)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = SiglipVisionModel(config)
    model.load_state_dict(
        {k.removeprefix("vision_model."): v for k, v in state.items()},
        strict=True,
        assign=True,
    )
    model.embeddings.position_ids = torch.arange(729).expand(1, -1)
    model.eval()
    processor = SiglipImageProcessor.from_pretrained(MODEL, revision=REVISION)
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "parameters": {},
        "cases": [],
        "config": config.to_dict(),
        "preprocessor": processor.to_dict(),
    }

    def write(path, tensor):
        array = tensor.detach().cpu().float().contiguous().numpy().astype("<f4")
        array.tofile(root / path)
        return {
            "file": str(path),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    def put(name, tensor):
        manifest["parameters"]["vit." + name] = write(
            Path("parameters") / (name + ".f32"), tensor
        )

    def get(name):
        return state["vision_model." + name]

    def dense(dst, src):
        put(dst + ".weight", get(src + ".weight").T)
        put(dst + ".bias", get(src + ".bias").reshape(1, 1, -1))

    def norm(dst, src):
        put(dst + ".scale", get(src + ".weight").reshape(1, 1, -1))
        put(dst + ".bias", get(src + ".bias").reshape(1, 1, -1))

    put(
        "embedding.weight",
        get("embeddings.patch_embedding.weight").permute(2, 3, 1, 0).reshape(588, 1152),
    )
    put("embedding.bias", get("embeddings.patch_embedding.bias").reshape(1, 1, -1))
    put("position", get("embeddings.position_embedding.weight").unsqueeze(0))
    for i in range(27):
        dst, src = f"encoder.layer{i}", f"encoder.layers.{i}"
        norm(dst + ".attention_norm", src + ".layer_norm1")
        norm(dst + ".mlp_norm", src + ".layer_norm2")
        for suffix in ("weight", "bias"):
            fused = torch.cat(
                [get(src + f".self_attn.{q}_proj.{suffix}") for q in "qkv"]
            )
            put(
                dst + ".attention.qkv." + suffix,
                fused.T if suffix == "weight" else fused.reshape(1, 1, -1),
            )
        dense(dst + ".attention.output", src + ".self_attn.out_proj")
        dense(dst + ".mlp.up", src + ".mlp.fc1")
        dense(dst + ".mlp.down", src + ".mlp.fc2")
    norm("encoder.final_norm", "post_layernorm")
    put("map.probe", get("head.probe"))
    for suffix in ("weight", "bias"):
        q, k, v = get("head.attention.in_proj_" + suffix).chunk(3)
        for name, value in [("query", q), ("kv", torch.cat([k, v]))]:
            put(
                "map.attention." + name + "." + suffix,
                value.T if suffix == "weight" else value.reshape(1, 1, -1),
            )
    dense("map.attention.output", "head.attention.out_proj")
    norm("map.norm", "head.layernorm")
    dense("map.mlp.up", "head.mlp.fc1")
    dense("map.mlp.down", "head.mlp.fc2")
    for filename in IMAGES:
        directory = root / Path(filename).stem
        directory.mkdir(exist_ok=True)
        url = "https://storage.googleapis.com/big_vision/siglip/" + filename
        original = directory / filename
        if not original.exists():
            original.write_bytes(urlopen(url).read())
        pixels = processor(
            images=Image.open(original).convert("RGB"), return_tensors="pt"
        ).pixel_values
        expected = model(pixel_values=pixels).pooler_output.unsqueeze(1)
        assert torch.isfinite(expected).all(), "nonfinite independent reference"
        # Conv14 stride14 on 384 pixels ignores the last six pixels, after resize.
        patches = (
            pixels[:, :, :378, :378]
            .reshape(1, 3, 27, 14, 27, 14)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(1, 729, 588)
        )
        case = {
            "name": directory.name,
            "url": url,
            "image_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "inputs": {
                "vit.image.patches": write(
                    Path(directory.name) / "patches.f32", patches
                )
            },
            "expected": write(Path(directory.name) / "expected.f32", expected),
        }
        manifest["cases"].append(case)
        print(
            f"{filename}: FP32 reference norm={expected.norm().item():.6f}", flush=True
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
