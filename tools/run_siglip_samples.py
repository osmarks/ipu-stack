#!/usr/bin/env python3
"""Run prepared SigLIP samples through a saved IPU package and verify outputs."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


PATCH_BYTES = 729 * 640 * 2
POOL_ROWS = 12
POOL_COLUMNS = 1152


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("base_input", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("bootloader", type=Path)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--cli", type=Path, default=Path("target/release/ipu-stack"))
    parser.add_argument("--device", default="/dev/ipu0")
    parser.add_argument("--minimum-cosine", type=float, default=0.995)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def decode_output(path: Path) -> np.ndarray:
    packed = np.fromfile(path, dtype="<f2")
    expected = POOL_ROWS * POOL_COLUMNS
    if packed.size != expected:
        raise RuntimeError(f"output contains {packed.size} values, expected {expected}")
    return packed.reshape(POOL_COLUMNS // 16, POOL_ROWS, 16).transpose(1, 0, 2).reshape(
        POOL_ROWS, POOL_COLUMNS
    ).astype(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.samples / "manifest.json").read_text())
    manifest = manifest[args.start :]
    if args.limit is not None:
        manifest = manifest[: args.limit]
    work = args.samples / "device-input.bin"
    output_directory = args.samples / "device-outputs"
    output_directory.mkdir(exist_ok=True)
    report_path = args.samples / "results.jsonl"
    shutil.copyfile(args.base_input, work)

    failures = 0
    report_mode = "w" if args.start == 0 else "a"
    with report_path.open(report_mode, buffering=1) as report:
        for offset, sample in enumerate(manifest):
            index = args.start + offset
            patches = Path(sample["patches"]).read_bytes()
            if len(patches) != PATCH_BYTES:
                raise RuntimeError(f"{sample['key']} patch binding has {len(patches)} bytes")
            with work.open("r+b", buffering=0) as input_file:
                input_file.write(patches)
            output = output_directory / f"{sample['key']}.bin"
            command = [
                str(args.cli),
                "host-run",
                str(args.package),
                str(args.bootloader),
                str(args.configuration),
                "graph",
                "--device",
                args.device,
                "--input",
                f"graph={work}",
                "--output",
                f"graph={output}",
            ]
            started = time.monotonic()
            completed = subprocess.run(command, text=True, capture_output=True)
            elapsed = time.monotonic() - started
            result = {
                "index": index,
                "key": sample["key"],
                "image": sample["image"],
                "seconds": elapsed,
                "returncode": completed.returncode,
            }
            if completed.returncode == 0:
                rows = decode_output(output)
                reference = np.load(sample["reference"]).astype(np.float32)
                row_cosines = [cosine(row, reference) for row in rows]
                result.update(
                    cosine=min(row_cosines),
                    max_error=float(np.max(np.abs(rows - reference[None, :]))),
                    duplicate_max_error=float(np.max(np.abs(rows - rows[0]))),
                    finite=bool(np.isfinite(rows).all()),
                )
                passed = result["finite"] and result["cosine"] >= args.minimum_cosine
            else:
                result["stderr"] = completed.stderr[-4000:]
                passed = False
            result["passed"] = passed
            failures += int(not passed)
            report.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"[{index + 1}/{args.start + len(manifest)}] {sample['key']} "
                f"pass={passed} cosine={result.get('cosine')} "
                f"max_error={result.get('max_error')} seconds={elapsed:.2f}",
                flush=True,
            )

    if failures:
        raise SystemExit(f"{failures} sample(s) failed")


if __name__ == "__main__":
    main()
