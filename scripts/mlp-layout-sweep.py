#!/usr/bin/env python3
"""Stratified, resumable SigLIP MLP hardware sweep; one device owner at a time.

Screening deliberately uses a small independent work/memory proxy, not the
planner's shortlist. Every selected case gets a complete build and numerical
check. Logs, profiles, predictions, failures and device timings are retained.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import dataclasses
import hashlib
import json
import itertools
import math
from pathlib import Path
import re
import subprocess
import time


@dataclasses.dataclass(frozen=True)
class Plan:
    r: int
    c: int
    k: int
    columns: int
    memory: str = "interleaved"
    orientation: str = "normal"
    distributed: bool = True
    reduction: str = "complete"
    local: str = "direct"

    def constraint(self, operation):
        result = self.k if self.distributed else 1
        return (f"{operation}:{self.r}x{self.c}x{self.k}:{result}x1:"
                f"{self.columns}:{self.memory}:{self.orientation}:{self.reduction}:{self.local}")


UP = Plan(4, 92, 4, 48)
DOWN = Plan(4, 24, 15, 48)


def grids(inner, columns, orientation="normal"):
    rows = 729
    if orientation == "swapped":
        rows, columns = columns, rows
    result = []
    for k in range(2, min(65, math.ceil(inner / 16) + 1)):
        local_k = math.ceil(math.ceil(inner / 16) / k) * 16
        if (k - 1) * local_k >= inner:
            continue
        for c in range(1, min(math.ceil(columns / 16), 1472 // k) + 1):
            r = 1472 // (k * c)
            local_r = math.ceil(math.ceil(rows / r) / 2) * 2
            local_c = math.ceil(math.ceil(columns / 16) / c) * 16
            if r * k > math.ceil(rows / 2):
                continue
            left = 2 * local_r * local_k
            right = 2 * local_c * local_k
            partial = 2 * local_r * local_c
            # Generous screens: leave real storage feasibility to the compiler.
            if left + right + partial > 360448 or right + partial > 262144:
                continue
            compute = (local_r * 4 + 160) * (local_k // 16) * (local_c // 16)
            proxy = compute + (left + partial * (k - 1)) / 8
            result.append((proxy, Plan(r, c, k, local_c, orientation=orientation)))
    return sorted(result, key=lambda item: (item[0], item[1].constraint(0)))


def stratified(inner, columns):
    ranked = grids(inner, columns)
    chosen = []
    # Include optima in different split/width strata, including deliberately
    # low-ranked plans. This makes cost calibration less selection-biased.
    for attr, targets in [("k", [2, 3, 4, 6, 8, 12, 15, 18, 24, 27, 36]),
                          ("r", [1, 2, 3, 4, 6, 8]),
                          ("columns", [16, 32, 48, 64, 96, 128])]:
        for target in targets:
            match = next((plan for _, plan in ranked if getattr(plan, attr) == target), None)
            if match is not None and match not in chosen:
                chosen.append(match)
    swapped = grids(inner, columns, "swapped")
    for target in [4, 8, 16]:
        match = next((plan for _, plan in swapped if plan.k == target), None)
        if match is not None:
            chosen.append(match)
    return chosen, len(ranked), len(swapped)


def manifest():
    cases = [("historical", UP, DOWN),
             ("default", UP, Plan(3, 18, 27, 64, memory="standard"))]
    counts = {}
    for side, inner, columns, anchor in [("up", 1152, 4304, UP), ("down", 4304, 1152, DOWN)]:
        plans, normal_count, swapped_count = stratified(inner, columns)
        counts[side] = dict(normal=normal_count, swapped=swapped_count)
        plans += [dataclasses.replace(anchor, memory="standard"),
                  dataclasses.replace(anchor, distributed=False),
                  dataclasses.replace(anchor, reduction="streamed"),
                  dataclasses.replace(anchor, distributed=False, reduction="streamed")]
        for index, plan in enumerate(plans):
            pair = (plan, DOWN) if side == "up" else (UP, plan)
            if not any(pair == (up, down) for _, up, down in cases):
                cases.append((f"{side}-{index:02d}", *pair))
    # Independent coordinate sweeps can miss a pair whose boundary is only
    # cheap when both sides change. Cover shared row grids and matching the
    # downprojection compute rows to the up-projection's scattered result.
    up_grids, down_grids = grids(1152, 4304), grids(4304, 1152)
    for rows in [3, 6, 8]:
        up = next(plan for _, plan in up_grids if plan.r == rows)
        down = next(plan for _, plan in down_grids if plan.r == rows)
        cases.append((f"joint-rows-{rows}", up, down))
    for name, up in [("joint-result-rows-4", UP), ("joint-result-rows-8", Plan(8, 90, 2, 48))]:
        cases.append((name, up, Plan(16, 23, 4, 64)))
    return cases, counts


def extract(log):
    result = {}
    patterns = {
        "compact_cycles": r"retained operator-plan finalist.*?estimated_cycles=(\d+)",
        "compact_exchange": r"retained operator-plan finalist.*?estimated_exchange_cycles=(\d+)",
        "expanded_cycles": r"selected analytical operator plan estimated_cycles=(\d+)",
        "expanded_exchange": r"selected analytical operator plan.*?estimated_exchange_cycles=(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, log)
        if match and match.lastindex:
            result[key] = int(match[1])
    benchmark = next((line for line in log.splitlines() if "effectiveGemmTflops=" in line), "")
    for key in ["cycles", "minimumTileCycles", "maximumAbsoluteError"]:
        match = re.search(rf"\b{key}=([\d.]+)", benchmark)
        if match:
            result[key] = float(match[1]) if "." in match[1] else int(match[1])
    result["hardware_pass"] = "hardwareTest=PASS" in log
    return result


def run_case(args, name, up, down):
    folder = args.output / name
    folder.mkdir(exist_ok=True)
    record_path = folder / "result.json"
    constraints = [up.constraint(0), down.constraint(2)]
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous["constraints"] != constraints:
            raise RuntimeError(f"stale manifest in {folder}")
        return previous
    command = [args.binary, args.config, "--sdk", args.sdk,
               "--device-lock", str(args.output.resolve() / "device.lock"),
               "--workload", "siglip-mlp-benchmark", "--mlp-batch", "1",
               "--package", str(folder / "model.ipuexe"),
               "--profile-output", str(folder / "execution.ipuprofile")]
    for constraint in constraints:
        command += ["--gemm-plan-constraint", constraint]
    log_path = folder / "build.log"
    # Recover a completed run if the coordinator was stopped between process
    # completion and writing its summary. Never recover a partial/failing run.
    recovered = log_path.exists() and "hardwareTest=PASS" in log_path.read_text()
    if recovered:
        old_command = json.loads((folder / "command.json").read_text())
        old_constraints = [old_command[i + 1] for i, arg in enumerate(old_command)
                           if arg == "--gemm-plan-constraint"]
        if old_constraints != constraints:
            raise RuntimeError(f"stale completed build in {folder}")
    else:
        (folder / "command.json").write_text(json.dumps(command, indent=2))
    print(f"START {name}: {' '.join(constraints)}", flush=True)
    start = time.monotonic()
    status = 0
    if not recovered:
        with log_path.open("w") as output:
            try:
                completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                           timeout=args.timeout)
                status = completed.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
    log = log_path.read_text()
    result = dict(name=name, constraints=constraints, up=dataclasses.asdict(up),
                  down=dataclasses.asdict(down), status=status,
                  seconds=None if recovered else time.monotonic() - start, **extract(log))
    if result["hardware_pass"]:
        for report, options in [("barriers", ["profile-barriers"]),
                                ("kernels", ["profile-query", "--group-by", "kernel", "--limit", "1000"]),
                                ("operations", ["profile-query", "--group-by", "operation", "--limit", "1000"])]:
            data = subprocess.check_output([args.cli, *options, str(folder / "execution.ipuprofile"), "--json"])
            (folder / f"{report}.json").write_bytes(data)
        barriers = json.loads((folder / "barriers.json").read_text())
        result["scheduled_exchange"] = sum(b["scheduledEventCycles"] for b in barriers)
        result["exchange_after_arrival"] = sum(b["afterLastArrivalCycles"] for b in barriers)
        if "expanded_cycles" in result:
            result["refined_cycles"] = (result["expanded_cycles"] - result["expanded_exchange"]
                                        + result["scheduled_exchange"])
    record_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {name}: status={status}, cycles={result.get('cycles')}, seconds={result['seconds']}", flush=True)
    # Never silently continue a hardware correctness failure as a slow layout.
    if not result["hardware_pass"] and "application loaded" in log:
        raise RuntimeError(f"hardware failure: inspect {folder / 'build.log'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="c600-init.ipucfg")
    parser.add_argument("--binary", default="target/release/ipu-trivial-test")
    parser.add_argument("--cli", default="target/release/ipu-stack")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent builds; hardware access is serialized")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cases, counts = manifest()
    inventory = dict(screened_grids=counts, cases=[dict(name=name, up=dataclasses.asdict(up),
                     down=dataclasses.asdict(down)) for name, up, down in cases])
    inventory["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    inventory["binary_sha256"] = hashlib.file_digest(Path(args.binary).open("rb"), "sha256").hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"{len(cases)} initial cases; screened grids: {counts}", flush=True)
    if args.dry_run:
        return
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        # Bound in-flight work and stop submitting cases after a hardware error.
        remaining = iter(cases)
        pending = {pool.submit(run_case, args, *case)
                   for case in itertools.islice(remaining, args.jobs)}
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                results.append(future.result())
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            # Inspect every completed result before replacing finished work.
            for case in itertools.islice(remaining, len(completed)):
                pending.add(pool.submit(run_case, args, *case))
    # Explore coupling: cross the three fastest independent alternatives on
    # each side. Preserve the initial stratified sample for calibration.
    winners = {}
    for side in ["up", "down"]:
        eligible = [r for r in results if r["hardware_pass"] and
                    (r["name"].startswith(side + "-") or r["name"] == "historical")]
        winners[side] = [Plan(**r[side]) for r in sorted(eligible, key=lambda r: r["cycles"])[:3]]
    pairs = {(tuple(r["constraints"])) for r in results}
    for i, up in enumerate(winners["up"]):
        for j, down in enumerate(winners["down"]):
            if (up.constraint(0), down.constraint(2)) in pairs:
                continue
            results.append(run_case(args, f"cross-{i}-{j}", up, down))
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
