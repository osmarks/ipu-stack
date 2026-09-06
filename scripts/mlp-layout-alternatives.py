#!/usr/bin/env python3
"""Controlled result-grid, tile-placement and bounded-reduction experiments."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

spec = importlib.util.spec_from_file_location("sweep", Path(__file__).with_name("mlp-layout-sweep.py"))
sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="c600-init.ipucfg")
    parser.add_argument("--binary", default="target/release/ipu-trivial-test")
    parser.add_argument("--cli", default="target/release/ipu-stack")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "device.lock"
    if not lock.exists():
        lock.symlink_to(Path("artifacts/layout-sweep/device.lock").resolve())
    up, down = sweep.UP, dataclasses.replace(sweep.DOWN, memory="standard")
    cases = [("control", up, down)]
    for rr, rc in [(1, 2), (2, 1), (2, 2), (1, 4)]:
        cases.append((f"up-result-{rr}x{rc}", dataclasses.replace(up, result_grid=(rr, rc)), down))
    for rr, rc in [(1, 3), (3, 1), (5, 1), (5, 3), (3, 5), (1, 15)]:
        cases.append((f"down-result-{rr}x{rc}", up, dataclasses.replace(down, result_grid=(rr, rc))))
    for side, limits in [("up", [1, 2]), ("down", [1, 2, 4, 7])]:
        for limit in limits:
            plan = dataclasses.replace(up if side == "up" else down, reduction=f"batch-{limit}")
            cases.append((f"{side}-batch-{limit}", plan if side == "up" else up,
                          plan if side == "down" else down))
    args.tile_mappings = {}
    # Transpose the embedding while retaining every ownership relationship.
    # This changes source-bus/pair membership without changing kernel shapes.
    for width in [2, 4, 16, 32, 46, 92, 184, 368]:
        name = f"mapping-transpose-{width}"
        args.tile_mappings[name] = [(tile % width) * (1472 // width) + tile // width
                                    for tile in range(1472)]
        cases.append((name, up, down))
    if args.followup:
        mixed_down = dataclasses.replace(down, result_grid=(5, 3))
        cases = [("mixed-control", up, mixed_down),
                 ("mixed-repeat", up, mixed_down),
                 ("both-mixed", dataclasses.replace(up, result_grid=(2, 2)), mixed_down)]
        for rr, rc in [(2, 3), (3, 3), (4, 3), (6, 2), (7, 2)]:
            cases.append((f"down-result-{rr}x{rc}", up,
                          dataclasses.replace(down, result_grid=(rr, rc))))
        # Preserve each GEMM compute-row group while changing its K/C embedding.
        for block, width in [(368, 92), (368, 4), (360, 24), (360, 15), (92, 2)]:
            mapping = []
            for tile in range(1472):
                base, local = divmod(tile, block)
                mapping.append(base * block + ((local % width) * (block // width) + local // width
                               if (base + 1) * block <= 1472 else local))
            for label, selected_down in [("base", down), ("mixed", mixed_down)]:
                name = f"mapping-block-{block}-{width}-{label}"
                args.tile_mappings[name] = mapping
                cases.append((name, up, selected_down))
        for width in [4, 92]:
            name = f"mapping-transpose-{width}-mixed"
            args.tile_mappings[name] = args.tile_mappings[f"mapping-transpose-{width}"]
            cases.append((name, up, mixed_down))
    if args.confirm:
        mixed_down = dataclasses.replace(down, result_grid=(5, 3))
        cases = [("winner-repeat-1", up, mixed_down), ("winner-repeat-2", up, mixed_down)]
        mapping = [(tile // 92) * 92 + (tile % 2) * 46 + (tile % 92) // 2
                   for tile in range(1472)]
        args.tile_mappings.update({name: mapping for name, _, _ in cases})
    cases = [case for case in cases if not args.only or case[0] in args.only]
    binary = args.output / "cohort-binary"
    digest = hashlib.file_digest(Path(args.binary).open("rb"), "sha256").hexdigest()
    manifest = dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    binary_sha256=digest, cases=[dict(name=n, up=dataclasses.asdict(u),
                    down=dataclasses.asdict(d), mapping=args.tile_mappings.get(n)) for n, u, d in cases])
    path = args.output / "manifest.json"
    manifest = json.loads(json.dumps(manifest))
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous[key] != manifest[key] for key in ["binary_sha256", "cases"]):
            raise RuntimeError("Changed experiment manifest; use a fresh output directory")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not binary.exists():
        shutil.copy2(args.binary, binary)
    args.binary = str(binary)
    sweep.run_cases(args, cases, [])


if __name__ == "__main__":
    main()
