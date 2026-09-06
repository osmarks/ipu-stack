#!/usr/bin/env python3
"""Report measured layout ranking and cost calibration from an MLP sweep."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def calibration(rows, key, actual="cycles"):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cli", default="target/release/ipu-stack")
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in sorted(args.directory.glob("*/result.json"))]
    passed = [r for r in records if r["hardware_pass"]]
    if not passed:
        raise SystemExit("no passing runs yet")
    initial = [r for r in passed if not r["name"].startswith("cross-")]
    keys = ["compact_cycles", "expanded_cycles", "refined_cycles"]
    stats = {key: calibration(initial, key) for key in keys}
    stats.update({key: calibration(initial, key, "scheduled_exchange")
                  for key in ["compact_exchange", "expanded_exchange"]})
    (args.directory / "calibration-summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    fields = ["name", "status", "cycles", "minimumTileCycles", *keys,
              "compact_exchange", "expanded_exchange", "scheduled_exchange", "seconds"]
    with (args.directory / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: r.get("cycles", float("inf"))))
    colors = {"up": "#2977b8", "down": "#ce6728", "joint": "#874bb6",
              "cross": "#378449", "baseline": "#30343a"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, key, title in zip(axes.flat, keys + ["expanded_exchange"],
                               ["Compact plan", "Expanded plan", "Expanded + final exchange schedule",
                                "Expanded exchange estimate"]):
        actual = "scheduled_exchange" if key == "expanded_exchange" else "cycles"
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
                 title=title, xlabel="Measured cycles" if actual == "cycles" else "Scheduled event cycles",
                 ylabel="Predicted cycles")
        axis.grid(alpha=.2)
    axes[0, 0].legend()
    figure.savefig(args.directory / "cost-calibration.svg")
    figure.savefig(args.directory / "cost-calibration.png", dpi=170)
    plt.close(figure)
    best = sorted(passed, key=lambda r: r["cycles"])
    lines = ["# MLP layout sweep", "", f"{len(records)} completed; {len(passed)} hardware passes.", "",
             "Calibration statistics exclude adaptive cross-products; differences ≤24 cycles are ties.", "",
             "The third estimate replaces expanded exchange cost with the final placed schedule.",
             "It is not the earlier provisional score used for finalist selection.", "",
             "| Plan | Cycles | Compact | Expanded | With final exchange |", "|---|---:|---:|---:|---:|"]
    for row in best[:15]:
        lines.append(f"| {row['name']} | {row['cycles']:,} | {row.get('compact_cycles', 0):,} | "
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
    print(f"best={best[0]['name']} cycles={best[0]['cycles']} completed={len(records)} passes={len(passed)}")


if __name__ == "__main__":
    main()
