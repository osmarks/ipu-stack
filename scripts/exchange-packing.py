#!/usr/bin/env python3
"""Analyze packed transport alternatives on captured physical exchange phases.

No payload duplication, recipient changes, overfetch, placement or hardware runs.
Synthetic snapshots are for scheduler experiments, not executable model plans.
"""

import argparse
import bisect
import html
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_TRANSFER_WORDS = 4148  # ipu_exchange::MAX_TRANSFER_WORDS


def has_read_write_overlap(phase):
    """A whole-phase gather/scatter cannot assume these transfers independent."""
    writes = defaultdict(list)
    for transfer in phase["transfers"]:
        for destination in transfer["destinations"]:
            start = destination["address"]
            writes[destination["tile"]].append((start, start + 4 * transfer["words"]))
    for tile, intervals in writes.items():
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        writes[tile] = merged
    ends = {tile: [end for _, end in intervals] for tile, intervals in writes.items()}
    for transfer in phase["transfers"]:
        tile = transfer["source"]
        intervals = writes.get(tile, [])
        for address in transfer["source_addresses"]:
            index = bisect.bisect_right(ends.get(tile, []), address)
            if (
                index < len(intervals)
                and intervals[index][0] < address + 4 * transfer["words"]
            ):
                return True
    return False


def route(t):
    return t["source"], tuple(d["tile"] for d in t["destinations"])


def mergeable(a, b):
    size = 4 * a["words"]
    return (
        route(a) == route(b)
        and len(a["source_addresses"]) == len(b["source_addresses"])
        and all(
            x + size == y for x, y in zip(a["source_addresses"], b["source_addresses"])
        )
        and all(
            x["address"] + size == y["address"]
            for x, y in zip(a["destinations"], b["destinations"])
        )
    )


def affine_tasks(copies, *, forward_only=False):
    """Group copies in their given order; hardware tasks require forward strides."""
    i = 0
    while i < len(copies):
        source, destination, size = copies[i]
        rows, ss, ds = 1, size, size
        if i + 1 < len(copies) and copies[i + 1][2] == size:
            ss = copies[i + 1][0] - source
            ds = copies[i + 1][1] - destination
            if not forward_only or (ss >= 0 and ds >= 0):
                rows = 2
                while i + rows < len(copies) and copies[i + rows] == (
                    source + rows * ss,
                    destination + rows * ds,
                    size,
                ):
                    rows += 1
        yield {
            "source": source,
            "destination": destination,
            "row_bytes": size,
            "rows": rows,
            "source_stride": ss if rows > 1 else size,
            "destination_stride": ds if rows > 1 else size,
        }
        i += rows


def copy_loops(copies):
    """Optimistic affine-loop count, NOT measured generated kernel counts."""
    return sum(1 for _ in affine_tasks(copies))


def append_copy(copies, source, destination, size):
    if (
        copies
        and copies[-1][0] + copies[-1][2] == source
        and copies[-1][1] + copies[-1][2] == destination
    ):
        old = copies[-1]
        copies[-1] = (old[0], old[1], old[2] + size)
    else:
        copies.append((source, destination, size))


def transform(phase, mode, include_copies=False):
    """Retain each multicast recipient set; pack whole phase into fresh buffers."""
    pack = mode in ("source", "both")
    unpack = mode in ("destination", "both")
    groups = defaultdict(list)
    for t in phase["transfers"]:
        t = dict(t, destinations=sorted(t["destinations"], key=lambda d: d["tile"]))
        groups[route(t)].append(t)
    # Separate hypothetical SRAM ranges. Their coexistence with model storage
    # is deliberately not claimed. Large footprints are not exported for replay.
    src_cursor, dst_cursor = defaultdict(int), defaultdict(int)
    local = {"source": defaultdict(list), "destination": defaultdict(list)}
    result = []
    for (source, receivers), group in sorted(groups.items()):
        key = (
            (lambda t: tuple(d["address"] for d in t["destinations"]))
            if mode == "source"
            else (lambda t: tuple(t["source_addresses"]))
        )
        group.sort(key=key)
        src_cursor[source] = (src_cursor[source] + 7) & ~7
        for tile in receivers:
            dst_cursor[tile] = (dst_cursor[tile] + 7) & ~7
        for t in group:
            size = t["words"] * 4
            src = 0x10000 + src_cursor[source]
            addresses = [src] if pack else t["source_addresses"][:]
            if pack:
                append_copy(
                    local["source"][source], t["source_addresses"][0], src, size
                )
                src_cursor[source] += size
            destinations = []
            for d in t["destinations"]:
                tile = d["tile"]
                dst = (0x80000 if pack else 0x10000) + dst_cursor[tile]
                destinations.append(
                    {"tile": tile, "address": dst if unpack else d["address"]}
                )
                if unpack:
                    append_copy(local["destination"][tile], dst, d["address"], size)
                    dst_cursor[tile] += size
            item = {
                "source": source,
                "source_addresses": addresses,
                "destinations": destinations,
                "words": t["words"],
                "width": "Word32",
            }
            if result and mergeable(result[-1], item):
                result[-1]["words"] += item["words"]
            else:
                result.append(item)
    # The snapshot interface has a bounded transfer size even when a packed
    # run is longer. Stream scheduling may subdivide these further.
    bounded = []
    for t in result:
        for offset in range(0, t["words"], MAX_TRANSFER_WORDS):
            bounded.append(
                dict(
                    t,
                    words=min(MAX_TRANSFER_WORDS, t["words"] - offset),
                    source_addresses=[a + 4 * offset for a in t["source_addresses"]],
                    destinations=[
                        dict(d, address=d["address"] + 4 * offset)
                        for d in t["destinations"]
                    ],
                )
            )
    result = bounded
    source_peak = max(src_cursor.values(), default=0) if pack else 0
    destination_peak = max(dst_cursor.values(), default=0) if unpack else 0
    copy_bytes = {
        name: max((sum(c[2] for c in copies) for copies in tiles.values()), default=0)
        for name, tiles in local.items()
    }
    metrics = {
        "source_staging_max_bytes": source_peak,
        "destination_staging_max_bytes": destination_peak,
        "combined_staging_max_bytes": max(
            (
                src_cursor[t] * pack + dst_cursor[t] * unpack
                for t in src_cursor.keys() | dst_cursor.keys()
            ),
            default=0,
        ),
        "source_copy_max_bytes": copy_bytes["source"],
        "destination_copy_max_bytes": copy_bytes["destination"],
        "local_copy_runs": sum(
            len(c) for tiles in local.values() for c in tiles.values()
        ),
        "affine_copy_loops": sum(
            copy_loops(c) for tiles in local.values() for c in tiles.values()
        ),
        # Eight bytes per cycle is an optimistic copy throughput; separate
        # source/destination compute phases have separate critical-path maxima.
        "copy_cycles_floor": sum((b + 7) // 8 for b in copy_bytes.values()),
        "replay_address_ranges_valid": source_peak <= 0x40000
        and destination_peak <= (0x60000 if pack else 0x40000),
    }
    if include_copies:
        metrics["copies"] = local
    return (
        phase if mode == "original" else {"phase": phase["phase"], "transfers": result}
    ), metrics


def geometry(phase):
    endpoints = defaultdict(int)
    outgoing, incoming = defaultdict(int), defaultdict(int)
    for t in phase["transfers"]:
        endpoints[t["source"]] += 1
        outgoing[t["source"]] += 4 * t["words"]
        for d in t["destinations"]:
            endpoints[d["tile"]] += 1
            incoming[d["tile"]] += 4 * t["words"]
    return {
        "transfers": len(phase["transfers"]),
        "receive_fragments": sum(len(t["destinations"]) for t in phase["transfers"]),
        "maximum_endpoint_fragments": max(endpoints.values(), default=0),
        "transmitted_bytes": sum(outgoing.values()),
        "received_bytes": sum(incoming.values()),
        "maximum_outgoing_bytes": max(outgoing.values(), default=0),
        "maximum_incoming_bytes": max(incoming.values(), default=0),
    }


def replay_schedules(rows, output, scheduler, jobs):
    """Encode synthetic phases with the ordinary B1024 production scheduler."""

    def run(row):
        path = output / f"phase-{row['phase']}-{row['mode']}.json"
        if not path.exists():
            return
        result = subprocess.run(
            [
                str(scheduler.resolve()),
                str(path),
                "--stream-words",
                "1024",
                "--balance-streams",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=dict(os.environ, RAYON_NUM_THREADS="4"),
        )
        path.with_suffix(".bench.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Scheduler failed: {path.with_suffix('.bench.log')}")
        for line in result.stdout.splitlines():
            fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
            if "maximumRowWords" in fields:
                assert fields["invariants"] == "PASS"
                row["scheduled_row_max_bytes"] = 4 * int(fields["maximumRowWords"])
                row["scheduled_horizon_cycles"] = int(fields["horizonCycles"])
                return
        raise RuntimeError(f"Missing scheduler result: {path}")

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(run, rows))


def render(rows, output):
    cells = []
    for row in rows:
        label = html.escape(row["label"])
        if row.get("read_write_overlap"):
            label = "READ/WRITE DEPENDENCIES: staging results invalid. " + label
        cells.append(
            f"<tr><td>{row['phase']}</td><td>{row['mode']}</td>"
            f"<td>{row['transfers']:,}</td><td>{row['maximum_endpoint_fragments']:,}</td>"
            f"<td>{row['combined_staging_max_bytes'] / 1024:,.1f}</td>"
            f"<td>{row['copy_cycles_floor']:,}</td><td>{row['affine_copy_loops']:,}</td>"
            f"<td>{row.get('scheduled_row_max_bytes', '—')}</td>"
            f"<td>{row.get('scheduled_horizon_cycles', '—')}</td>"
            f'<td title="{label}">{label}</td></tr>'
        )
    (output / "index.html").write_text(
        """<!doctype html><meta charset="utf-8">
<title>Exchange packing opportunities</title><style>
body{font:17px Georgia,serif;color:#111;background:#fff;margin:2em auto;max-width:1500px;padding:0 1em}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:.5em;text-align:right;border-bottom:1px solid #bbb}
th{position:sticky;top:0;background:white}td:last-child{text-align:left;max-width:400px}a{color:inherit}
</style><h1>Exchange packing opportunities</h1>
<p>Whole-phase staging with unchanged recipient sets and payload bytes. Synthetic addresses;
no placement or hardware validation. Copy cycles are optimistic throughput floors,
excluding barriers and setup. Scheduler results use ordinary B1024 transfers and synthetic placement;
rows are per-phase, before sharing. Affine loops are representability estimates, not generated code.</p>
<p><a href="analysis.json">JSON results</a></p><table><thead><tr><th>Phase</th><th>Staging</th>
<th>Transfers</th><th>Max endpoint fragments</th><th>Max scratch KiB/tile</th>
<th>Copy cycle floor</th><th>Affine loops, all tiles</th><th>Max row bytes</th><th>Exchange cycles (model)</th><th>Provenance</th></tr></thead><tbody>"""
        + "\n".join(cells)
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--phase", type=int, action="append")
    parser.add_argument(
        "--scheduler", type=Path, help="Optional ipu-exchange-schedule-bench executable"
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--replay-only", action="store_true", help="Replay previously exported fixtures"
    )
    args = parser.parse_args()
    if args.replay_only:
        if not args.scheduler:
            parser.error("--replay-only requires --scheduler")
        rows = json.loads((args.output / "analysis.json").read_text())
        replay_schedules(rows, args.output, args.scheduler, args.jobs)
        (args.output / "analysis.json").write_text(json.dumps(rows, indent=2))
        render([r for r in rows if "scheduled_row_max_bytes" in r], args.output)
        return
    snapshot = json.loads(args.snapshot.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        snapshot["phases"],
        key=lambda p: geometry(p)["maximum_endpoint_fragments"],
        reverse=True,
    )
    selected = set(
        args.phase if args.phase else [p["phase"] for p in ranked[: args.top]]
    )
    rows = []
    for phase in ranked:
        if args.phase and phase["phase"] not in selected:
            continue
        # By default analyze every phase; only export selected replay fixtures.
        original = geometry(phase)
        dependent = has_read_write_overlap(phase)
        for mode in ("original", "coalesced", "source", "destination", "both"):
            if dependent and mode != "original":
                continue
            transformed, extra = transform(phase, mode)
            row = {
                "phase": phase["phase"],
                "mode": mode,
                "read_write_overlap": dependent,
                "label": snapshot.get("phase_labels", {}).get(str(phase["phase"]), ""),
                **geometry(transformed),
                **extra,
            }
            assert row["transmitted_bytes"] == original["transmitted_bytes"]
            assert row["received_bytes"] == original["received_bytes"]
            rows.append(row)
            if phase["phase"] in selected and row["replay_address_ranges_valid"]:
                replay = dict(snapshot, phases=[transformed])
                (args.output / f"phase-{phase['phase']}-{mode}.json").write_text(
                    json.dumps(replay, separators=(",", ":"))
                )
    if args.scheduler:
        replay_schedules(rows, args.output, args.scheduler, args.jobs)
    (args.output / "analysis.json").write_text(json.dumps(rows, indent=2))
    render([r for r in rows if r["phase"] in selected], args.output)
    print(
        json.dumps(
            {
                "selected_phases": sorted(selected),
                "analyzed_phases": len(ranked) if not args.phase else len(selected),
            }
        )
    )


if __name__ == "__main__":
    main()
