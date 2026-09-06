#!/usr/bin/env python3
"""Run and resume a hardware batch sweep of three-iteration structured repeats."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time


def run_case(args, family, batch):
    path = args.output / f"{family}-b{batch}-n{args.blocks}"
    path.mkdir(parents=True, exist_ok=True)
    result_path = path / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    command = [str(args.binary), str(args.configuration), "--device-lock", str(args.device_lock),
               "--package", str(path / "model.ipuexe"), "--profile-output", str(path / "execution.ipuprofile")]
    if family == "mlp":
        command += ["--workload", "siglip-mlp-benchmark", "--mlp-batch", str(batch), "--mlp-blocks", str(args.blocks)]
    else:
        command += ["--workload", "siglip-attention-benchmark", "--attention-strategy", family,
                    "--attention-batch", str(batch), "--attention-blocks", str(args.blocks)]
    (path / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads))
    with (path / "run.log").open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    log = (path / "run.log").read_text()
    status = "pass" if process.returncode == 0 and "hardwareTest=PASS" in log else "failed"
    # Keep every non-memory failure visible; never classify a generic planner
    # rejection as OOM without investigating its cause.
    if status == "failed" and ("out of memory" in log.lower() or "OutOfMemory" in log or "no candidate within tile SRAM" in log):
        status = "oom"
    result = dict(family=family, batch=batch, blocks=args.blocks, status=status,
                  returncode=process.returncode, seconds=round(time.monotonic() - started, 1))
    if status == "pass":
        query = subprocess.run([str(args.cli), "profile-query", str(path / "execution.ipuprofile"), "--json"],
                               capture_output=True, text=True, check=True)
        (path / "summary.json").write_text(query.stdout)
        result["cycles"] = json.loads(query.stdout)["profileSpanCycles"]
        subprocess.run([str(args.cli), "profile-render", str(path / "execution.ipuprofile"),
                        "--output", str(path / "profile.html")], check=True, capture_output=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/repeat-sweep"))
    parser.add_argument("--binary", type=Path, default=Path("target/release/ipu-trivial-test"))
    parser.add_argument("--cli", type=Path, default=Path("target/release/ipu-stack"))
    parser.add_argument("--configuration", type=Path, default=Path("c600-init.ipucfg"))
    parser.add_argument("--device-lock", type=Path, default=Path("artifacts/layout-sweep/device.lock"))
    parser.add_argument("--families", nargs="+", choices=["flash", "materialized", "mlp"], default=["flash", "materialized", "mlp"])
    parser.add_argument("--batches", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if min(args.blocks, args.jobs, args.threads, *args.batches) < 1:
        parser.error("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case, args, family, batch) for batch in args.batches for family in args.families]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "results.json").write_text(json.dumps(sorted(results, key=lambda r: (r["family"], r["batch"])), indent=2) + "\n")


if __name__ == "__main__":
    main()
