#!/usr/bin/env python3
"""Controlled gather/pack/multicast experiment on independent activation transfers.

The selected phases must contain independent non-iterated activation streams;
iterated sources are treated as weights. Keeps final destinations and bytes fixed.
Relay scratch is hypothetical; this is
scheduler replay, not a claim that the full model can place the added buffers.
"""

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "packing", Path(__file__).with_name("exchange-packing.py")
)
packing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packing)


def affine_tasks(copies):
    copies = sorted(copies, key=lambda c: c[1])
    tasks = []
    i = 0
    while i < len(copies):
        source, destination, size = copies[i]
        rows, ss, ds = 1, size, size
        if i + 1 < len(copies) and copies[i + 1][2] == size:
            ss = copies[i + 1][0] - source
            ds = copies[i + 1][1] - destination
            if ss >= 0 and ds >= 0:
                rows = 2
                while i + rows < len(copies) and copies[i + rows] == (
                    source + rows * ss,
                    destination + rows * ds,
                    size,
                ):
                    rows += 1
        tasks.append(
            {
                "source": source,
                "destination": destination,
                "row_bytes": size,
                "rows": rows,
                "source_stride": ss if rows > 1 else size,
                "destination_stride": ds if rows > 1 else size,
            }
        )
        i += rows
    return tasks


def scratch_base(intervals, size, start, stop):
    for base in range(start, stop - size + 1, 0x1000):
        if all(base + size <= lo or base >= hi for lo, hi in intervals):
            return base
    raise ValueError("no nonaliasing scratch range in fixture")


def gather(phase, tile_count, shards=1, direct_receive=False, context=None):
    routes = defaultdict(list)
    reads, writes = defaultdict(list), defaultdict(list)
    for t in phase["transfers"]:
        assert t["width"] == "Word32" and len(t["source_addresses"]) == 1
        routes[tuple(sorted(d["tile"] for d in t["destinations"]))].append(t)
    for t in (context or phase)["transfers"]:
        for address in t["source_addresses"]:
            reads[t["source"]].append((address, address + t["words"] * 4))
        for d in t["destinations"]:
            writes[d["tile"]].append((d["address"], d["address"] + t["words"] * 4))
    groups = []
    for receivers, transfers in sorted(routes.items()):
        low = min(
            d["address"]
            for t in transfers
            for d in t["destinations"]
            if d["tile"] == receivers[0]
        )
        high = max(
            d["address"] + t["words"] * 4
            for t in transfers
            for d in t["destinations"]
            if d["tile"] == receivers[0]
        )
        chunk = ((high - low + shards - 1) // shards + 31) // 32 * 32
        pieces = defaultdict(list)
        for t in transfers:
            address = next(
                d["address"] for d in t["destinations"] if d["tile"] == receivers[0]
            )
            assert (address - low) // chunk == (
                address - low + t["words"] * 4 - 1
            ) // chunk, "split crosses fragment"
            pieces[(address - low) // chunk].append(t)
        groups.extend((receivers, part) for _, part in sorted(pieces.items()))
    assert len(groups) <= tile_count, "too many independent relays"
    used, gathered, broadcast, cases = set(), [], [], []
    for receivers, transfers in groups:
        sources = {t["source"] for t in transfers}
        # One distinct relay per segment; no source-own or destination-own
        # loopback. Relays may send original data for other segments.
        leader = next(
            t
            for t in range(tile_count - 1, -1, -1)
            if t not in sources and t not in receivers and t not in used
        )
        used.add(leader)
        bases = {
            t: min(
                d["address"]
                for x in transfers
                for d in x["destinations"]
                if d["tile"] == t
            )
            for t in receivers
        }
        cursor = sum(t["words"] * 4 for t in transfers)
        incoming = scratch_base(reads[leader], cursor, 0x50000, 0x80000)
        copies, cursor = [], 0
        items = sorted(transfers, key=lambda t: (t["source"], t["source_addresses"]))
        for t in items:
            offsets = {d["address"] - bases[d["tile"]] for d in t["destinations"]}
            assert len(offsets) == 1, "replicas have different relative layouts"
            offset = offsets.pop()
            size = t["words"] * 4
            copies.append((cursor, offset, size))
            item = dict(
                t, destinations=[{"tile": leader, "address": incoming + cursor}]
            )
            if (
                gathered
                and gathered[-1]["words"] + item["words"] <= packing.MAX_TRANSFER_WORDS
                and packing.mergeable(gathered[-1], item)
            ):
                gathered[-1]["words"] += item["words"]
            else:
                gathered.append(item)
            cursor += size
        intervals = []
        for _, offset, size in sorted(copies, key=lambda c: c[1]):
            assert not intervals or offset >= intervals[-1][1], "overlapping outputs"
            if intervals and offset == intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], offset + size)
            else:
                intervals.append((offset, offset + size))
        end = intervals[-1][1]
        outgoing = scratch_base(writes[leader], end, 0x80000, 0xC0000)
        # Preserve existing operand padding rather than transmitting holes.
        for begin, stop in intervals:
            for offset in range(begin, stop, packing.MAX_TRANSFER_WORDS * 4):
                size = min(packing.MAX_TRANSFER_WORDS * 4, stop - offset)
                broadcast.append(
                    {
                        "source": leader,
                        "source_addresses": [outgoing + offset],
                        "destinations": [
                            {"tile": t, "address": bases[t] + offset} for t in receivers
                        ],
                        "words": size // 4,
                        "width": "Word32",
                    }
                )
        tasks = affine_tasks(copies)
        if direct_receive:
            # Receive each fragment at its final packed offset. This trades
            # larger gather rows for eliminating the packing pass entirely.
            outgoing = scratch_base(
                reads[leader] + writes[leader], end, 0x80000, 0xC0000
            )
            gathered = [t for t in gathered if t["destinations"][0]["tile"] != leader]
            for t in items:
                offset = next(
                    d["address"] - bases[d["tile"]] for d in t["destinations"]
                )
                gathered.append(
                    dict(
                        t, destinations=[{"tile": leader, "address": outgoing + offset}]
                    )
                )
            for t in broadcast:
                if t["source"] == leader:
                    offset = (
                        t["destinations"][0]["address"]
                        - bases[t["destinations"][0]["tile"]]
                    )
                    t["source_addresses"] = [outgoing + offset]
            incoming = outgoing
            tasks = []
        cases.append(
            {
                "phase": phase["phase"],
                "tile": leader,
                "bytes": end,
                "input_bytes": cursor,
                "input_address": incoming,
                "output_address": outgoing,
                "tasks": tasks,
            }
        )
    first, last = (
        {"phase": phase["phase"], "transfers": gathered},
        {"phase": phase["phase"], "transfers": broadcast},
    )
    assert not packing.has_read_write_overlap(first)
    assert not packing.has_read_write_overlap(last)
    return first, last, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", type=int, action="append", required=True)
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--direct-receive", action="store_true")
    parser.add_argument(
        "--include-repeated",
        action="store_true",
        help="Keep the captured phase's Repeat-weight traffic alongside the activation gather",
    )
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")
    snapshot = json.loads(args.snapshot.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = dict(
        snapshot, phases=[p for p in snapshot["phases"] if p["phase"] in args.phase]
    )
    if not args.skip_original:
        (args.output / "selected.json").write_text(json.dumps(snapshot))
    rows, cases = [], []
    for p in snapshot["phases"]:
        if p["phase"] not in args.phase:
            continue
        activation = {
            "phase": p["phase"],
            "transfers": [t for t in p["transfers"] if len(t["source_addresses"]) == 1],
        }
        assert not packing.has_read_write_overlap(activation), (
            "activation phase is dependent"
        )
        first, last, local = gather(
            activation,
            snapshot["tile_count"],
            args.shards,
            args.direct_receive,
            p if args.include_repeated else None,
        )
        cases += local
        if args.include_repeated:
            first["transfers"] += [
                t for t in p["transfers"] if len(t["source_addresses"]) > 1
            ]
        modes = [
            ("original", p if args.include_repeated else activation),
            ("gather", first),
            ("broadcast", last),
        ]
        if args.direct_receive:
            # Receive-to-send dependencies are explicit in this ordered list.
            # Production scheduling can keep forwarding in one exchange.
            modes.append(
                (
                    "forward",
                    {
                        "phase": p["phase"],
                        "transfers": first["transfers"] + last["transfers"],
                    },
                )
            )
        for mode, phase in modes:
            if mode == "original" and args.skip_original:
                continue
            path = args.output / f"phase-{p['phase']}-{mode}.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": snapshot["schema_version"],
                        "tile_count": snapshot["tile_count"],
                        "phases": [phase],
                    }
                )
            )
            rows.append(dict(phase=p["phase"], mode=mode, **packing.geometry(phase)))
    (args.output / "copies.json").write_text(json.dumps(cases))
    packing.replay_schedules(rows, args.output, args.scheduler, 6)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2))
    for p in args.phase:
        local = [c for c in cases if c["phase"] == p]
        print(
            "phase",
            p,
            "relays",
            len(local),
            "max_copy_bytes",
            max(c["bytes"] for c in local),
            "max_calls",
            max(len(c["tasks"]) for c in local),
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
