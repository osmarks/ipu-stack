#!/usr/bin/env python3
"""Prepare image-sample patch bindings and Hugging Face pooler references."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from siglip_reference import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def pack_patches(pixels: np.ndarray, patch_size: int) -> bytes:
    channels, image_height, image_width = pixels.shape
    patch_grid = image_height // patch_size
    rows = patch_grid * patch_grid
    patch_elements = channels * patch_size * patch_size
    padded_inner = ((patch_elements + 63) // 64) * 64
    matrix = np.zeros((rows, padded_inner), dtype=np.float16)
    for patch_y in range(patch_grid):
        for patch_x in range(patch_grid):
            row = patch_y * patch_grid + patch_x
            patch = pixels[
                :,
                patch_y * patch_size : (patch_y + 1) * patch_size,
                patch_x * patch_size : (patch_x + 1) * patch_size,
            ]
            matrix[row, :patch_elements] = patch.reshape(-1).astype(np.float16)

    row_block_dimension = 13
    row_grid = (rows + row_block_dimension - 1) // row_block_dimension
    base_rows, larger_shards = divmod(rows, row_grid)
    packed = []
    row_start = 0
    for block_row in range(row_grid):
        block_rows = base_rows + int(block_row < larger_shards)
        for column_start in range(0, padded_inner, 64):
            block = matrix[
                row_start : row_start + block_rows,
                column_start : column_start + 64,
            ]
            packed.append(block.reshape(block_rows, 4, 16).transpose(1, 0, 2).tobytes())
        row_start += block_rows
    result = b"".join(packed)
    assert len(result) == rows * padded_inner * 2
    return result


def main() -> None:
    args = parse_args()
    paths = sorted(
        path
        for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not paths:
        raise RuntimeError("no sample images found")
    args.output.mkdir(parents=True, exist_ok=True)
    patch_directory = args.output / "patches"
    reference_directory = args.output / "references"
    patch_directory.mkdir(exist_ok=True)
    reference_directory.mkdir(exist_ok=True)

    model, config = load_model(args.model)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    device = torch.device(args.device)
    model.to(device)

    manifest = []
    for batch_start in range(0, len(paths), args.batch_size):
        batch_paths = paths[batch_start : batch_start + args.batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].float()
        with torch.inference_mode():
            references = model(
                pixel_values=pixels.to(device), interpolate_pos_encoding=False
            ).pooler_output.float().cpu().numpy()
        for path, pixel_values, reference in zip(batch_paths, pixels.numpy(), references):
            key = f"{len(manifest):03d}-{path.stem}"
            patch_path = patch_directory / f"{key}.bin"
            reference_path = reference_directory / f"{key}.npy"
            patch_path.write_bytes(pack_patches(pixel_values, config.vision_config.patch_size))
            np.save(reference_path, reference)
            manifest.append(
                {
                    "key": key,
                    "image": str(path),
                    "patches": str(patch_path),
                    "reference": str(reference_path),
                }
            )
        print(f"prepared {min(batch_start + args.batch_size, len(paths))}/{len(paths)}")

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
