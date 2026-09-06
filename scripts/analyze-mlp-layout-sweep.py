#!/usr/bin/env python3
"""Report measured layout ranking and cost calibration from an MLP sweep."""
import argparse
import csv
import json
import math
import re
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def calibration(rows, key, actual="renderer_cycles"):
    samples = [r for r in rows if key in r and actual in r]
    predicted = np.array([r[key] for r in samples], dtype=float)
    measured = np.array([r[actual] for r in samples], dtype=float)
    if not len(samples):
        return {}
    selected = min(samples, key=lambda r: r[key])
    comparisons = inversions = 0
    for i in range(len(samples)):
        for j in range(i):
            # Ignore differences smaller than run-to-run measurement noise.
            if abs(measured[i] - measured[j]) <= 24:
                continue
            comparisons += 1
            inversions += int((predicted[i] - predicted[j]) * (measured[i] - measured[j]) < 0)
    return dict(samples=len(samples), median_ratio=float(np.median(predicted / measured)),
                median_absolute_percent_error=float(np.median(abs(predicted / measured - 1)) * 100),
                spearman=float(spearmanr(predicted, measured).statistic) if len(samples) > 2 else None,
                inversions=inversions, comparable_pairs=comparisons,
                selected=selected["name"], selected_cycles=selected[actual],
                regret_cycles=int(selected[actual] - measured.min()))


def instruction_model_report(database):
    """Offline checks against instruction counts; do not change planner prices."""
    families = {}
    for sample in database["measurements"]:
        key = sample["key"]
        dimensions = key["dimensions"]
        spec = dimensions.get("kernelSpec", "")
        n = int(dimensions.get("outputElements", 0))
        gemm = re.search(r"gemm_f16_init_(small|large)_rows_interleaved_k(\d+)_c(\d+)_r(\d+)_r(\d+)", key["kernel"])
        if spec == "Gelu":
            family = "GELU (logical size; padding may add waves)"
            predicted = 300 + 558 * math.ceil(n / 96)
        elif spec.startswith("ReductionSum"):
            family = "F16 reduction"
            partials = int(re.search(r"partials: (\d+)", spec)[1])
            predicted = 282 + 6 * math.ceil(n / 48) * (9 + 6 * (partials - 1))
        elif gemm:
            family = "Interleaved F16 GEMM"
            size, inner, columns, small, large = gemm.groups()
            rows = int(small if size == "small" else large)
            predicted = 294 + math.ceil(int(inner) / 16) * math.ceil(int(columns) / 16) * (4 * rows + 160)
        else:
            continue
        actual = sample["medianCycles"]
        families.setdefault(family, []).append((abs(predicted - actual), abs(predicted / actual - 1)))
    lines = ["", "## Instruction-count checks", "",
             "Offline diagnostics only; these formulas were not used to select or cost the sweep.",
             "Each row counts distinct exported kernel/metadata keys, not independent hardware runs.", "",
             "| Kernel | Keys | Median absolute error (%) | Maximum error (cycles) |",
             "|---|---:|---:|---:|"]
    for family, errors in families.items():
        lines.append(f"| {family} | {len(errors)} | {np.median([e[1] for e in errors]) * 100:.3f} | {max(e[0] for e in errors)} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cli", default="target/release/ipu-stack")
    args = parser.parse_args()
    records = []
    for path in sorted(args.directory.glob("*/result.json")):
        record = json.loads(path.read_text())
        records.append(record)
        if record["hardware_pass"] and "renderer_cycles" not in record:
            report = json.loads(subprocess.check_output([
                args.cli, "profile-query", str(path.parent / "execution.ipuprofile"),
                "--limit", "0", "--json"]))
            record["renderer_cycles"] = report["profileSpanCycles"]
            path.write_text(json.dumps(record, indent=2) + "\n")
    passed = [r for r in records if r["hardware_pass"]]
    if not passed:
        raise SystemExit("no passing runs yet")
    initial = [r for r in passed if not r["name"].startswith("cross-")]
    keys = ["compact_cycles", "expanded_cycles", "refined_cycles"]
    stats = {key: calibration(initial, key) for key in keys}
    stats.update({key: calibration(initial, key, "scheduled_exchange")
                  for key in ["compact_exchange", "expanded_exchange"]})
    (args.directory / "calibration-summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    fields = ["name", "status", "renderer_cycles", "cycles", "minimumTileCycles", *keys,
              "compact_exchange", "expanded_exchange", "scheduled_exchange", "seconds"]
    with (args.directory / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: r.get("renderer_cycles", float("inf"))))
    colors = {"up": "#2977b8", "down": "#ce6728", "joint": "#874bb6",
              "cross": "#378449", "baseline": "#30343a"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, key, title in zip(axes.flat, keys + ["expanded_exchange"],
                               ["Compact plan", "Expanded plan", "Expanded + final exchange schedule",
                                "Expanded exchange estimate"]):
        actual = "scheduled_exchange" if key == "expanded_exchange" else "renderer_cycles"
        for family in colors:
            rows = [r for r in passed if r["name"].split("-")[0] == family
                    or (family == "baseline" and r["name"] in ["default", "historical"])]
            rows = [r for r in rows if key in r]
            if rows:
                axis.scatter([r[actual] for r in rows], [r[key] for r in rows],
                             label=family, color=colors[family], alpha=.8, s=30)
        values = [r[k] for r in passed for k in [key, actual] if k in r]
        low, high = min(values) * .85, max(values) * 1.15
        axis.plot([low, high], [low, high], color="#666666", linewidth=1, linestyle="--")
        axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
                 title=title, xlabel="Measured cycles" if actual == "renderer_cycles" else "Scheduled event cycles",
                 ylabel="Predicted cycles")
        axis.grid(alpha=.2)
    axes[0, 0].legend()
    figure.savefig(args.directory / "cost-calibration.svg")
    figure.savefig(args.directory / "cost-calibration.png", dpi=170)
    plt.close(figure)
    best = sorted(passed, key=lambda r: r["renderer_cycles"])
    lines = ["# MLP layout sweep", "", f"{len(records)} completed; {len(passed)} hardware passes.", "",
             "Calibration statistics exclude adaptive cross-products; differences ≤24 cycles are ties.", "",
             "The third estimate replaces expanded exchange cost with the final placed schedule.",
             "It is not the earlier provisional score used for finalist selection.", "",
             "Runtime starts at the renderer’s default cut: the latest tile’s initial profile entry.", "", "| Plan | Renderer cycles | Compact | Expanded | With final exchange |", "|---|---:|---:|---:|---:|"]
    for row in best[:15]:
        lines.append(f"| {row['name']} | {row['renderer_cycles']:,} | {row.get('compact_cycles', 0):,} | "
                     f"{row.get('expanded_cycles', 0):,} | {row.get('refined_cycles', 0):,} |")
    lines += ["", "## Calibration", "", "| Estimate | Median predicted/measured | Rank correlation | Selection regret |",
              "|---|---:|---:|---:|"]
    for key, values in stats.items():
        if values:
            rho = values['spearman']
            lines.append(f"| {key} | {values['median_ratio']:.3f} | {rho if rho is None else round(rho, 3)} | "
                         f"{values['regret_cycles']:,} |")
    lines += ["", "## Best constraints", "", "```", *best[0]["constraints"], "```", "",
              "## Rejected or failed builds", ""]
    for row in records:
        if not row["hardware_pass"]:
            lines.append(f"- {row['name']}: {row['status']}; see `{row['name']}/build.log`.")
    (args.directory / "report.md").write_text("\n".join(lines) + "\n")
    # Existing profiler retains kernel geometry and cycle distributions; keep
    # these measurements available for calibration without refitting the model
    # against the same samples used to judge its current ranking.
    build_id = subprocess.check_output([args.cli, "kernel-build-id", "device"], text=True).splitlines()[0]
    subprocess.run([args.cli, "profile-calibrate", *[str(args.directory / r["name"] / "execution.ipuprofile")
                    for r in passed], "--build-id", build_id,
                    "--output", str(args.directory / "kernel-measurements.json")], check=True)
    database = json.loads((args.directory / "kernel-measurements.json").read_text())
    lines += instruction_model_report(database)
    (args.directory / "report.md").write_text("\n".join(lines) + "\n")
    print(f"best={best[0]['name']} cycles={best[0]['renderer_cycles']} completed={len(records)} passes={len(passed)}")


if __name__ == "__main__":
    main()
