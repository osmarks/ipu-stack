# Local packing versus fragmented exchanges — 12 September 2026

## Follow-up: in-place feasibility and a missing control

**The first staging experiment missed receive-then-forward dependencies within a
phase. Its scheduler checks validate the transformed graph, not equivalence to
the original graph. Counterfactual staging timings for such phases are invalid.**
The tool now detects read/write overlap across every captured Repeat source
address and refuses to export transformed fixtures for those phases. The selected
HTML/JSON reports flag affected results and remove their scheduler timings. The
affected phases are PE 58 and SigLIP 16, 50 and 55. The
original tables below preserve the investigation history; affected staging rows
must not be used as performance predictions.

Receiver staging need not require a second full-sized allocation. In the SigLIP
weight phase (55), 1,463 of 1,472 receivers have contiguous, nonoverlapping final
write coverage. The busiest receiver has exactly 27,648 bytes of both payload and
final coverage. PE phase 62 has contiguous coverage on 1,200 of 1,440 receivers;
its busiest receiver has exactly 56,448 bytes of both. Provided the allocator
reserves the destination region through receive and rearrangement, packed data
can occupy that region and be permuted in place afterward. Gaps require individual
allocation boundaries/liveness checks; an address span alone does not establish
that intervening memory is available.

More surprisingly, 1,463 SigLIP weight receivers have an **identity** relative
address mapping after the proposed packing: no permutation would be needed if
the packed receive base were their final destination base. The busiest tile (8)
has 6,912 identity 32-bit words. Thus much of the supposed rearrangement was a
missed coalescing opportunity.

The phase contains 302,674 independent weight transfers followed by 368 forwarding
transfers. Read/write interval checks across all Repeat addresses find overlap
only in that forwarding tail. Sorting/coalescing just the independent prefix,
while retaining the tail afterward and every original address, gives:

| | Original | Coalesced independent prefix |
|---|---:|---:|
| Transfers | 303,042 | 4,326 |
| Maximum row bytes/tile | 7,936 | 1,324 |
| Modelled exchange cycles | 14,672 | 14,171 |
| Added copying or staging | 0 | 0 |

Scheduler invariants pass. This is still an offline fixture, not a production
coalescer change or hardware validation. Blindly sorting the whole phase instead
changes dependency direction and is invalid, even though scheduler invariants
pass. `phase-55-coalesced-prefix.*` is the dependency-preserving control;
`phase-55-reordered.*` is the rejected whole-phase control.

For actual permutations, cycle rotation needs one temporary element; a visited
bitmap or a structured permutation algorithm determines which cycles to traverse.
It need not require a full-sized scratch tensor, but can lose the efficient
streaming behavior of an out-of-place copy. A per-word descriptor table would
undermine the memory saving. Source-side in-place packing additionally requires
that no later consumer needs the original layout, particularly resident weights.


`scripts/exchange-packing.py` analyzes captured exchange phases without building
another executable model. It compares source packing, receiver staging, and both.
Every alternative preserves the source tile, multicast recipient sets and payload
bytes. It then coalesces contiguous transport runs and replays selected phases
through the existing B1024 balanced-stream scheduler and row encoder.

The captures are batch-two capacity baselines: 24-layer PE and 27-layer SigLIP.
They use provisional tensor placement, before final package support placement.
Phase numbers and selected layouts need not match earlier executable profiles.
The replay uses ordinary 32-bit transfers for every alternative, including the
original; it does not evaluate production paired-mode selection.

## PE results

Maximum **per-tile, per-phase** row bytes, before cross-phase row sharing:

| Phase / representative movement | Original transfers → receiver staging | Row bytes, original → staging | Exchange cycles, original → staging | Added ideal copy cycles | Staging KiB/tile |
|---|---:|---:|---:|---:|---:|
| 62: MLP down input, FP8 AMP layout redistribution | 147,712 → 8,655 | 16,776 → 976 | 15,937 → 14,376 | 7,056 | 55.1 |
| 58: MLP up input, FP8 AMP layout redistribution | 45,375 → 5,018 | 8,408 → 840 | 27,446 → 22,524 | 7,552 | 59.0 |
| 84: MAP input, FP16 row-major to heads | 82,504 → 82,399 | 6,008 → 5,936 | 9,256 → 6,220 | 2,816 | 22.0 |

The first two are predominantly **destination fragmentation**, rather than a need
to send that many independent payloads. Packing only the source does nothing for
phase 62. For phase 84, even packing both ends leaves 80,479 transfers: most of its
fragmentation is in the distinct source/recipient routes. A direct staging copy
cannot coalesce those routes without changing ownership or adding a forwarding
exchange.

The large reductions in row storage do not establish speedups. In phase 62 the
added copy throughput floor exceeds the exchange saving by 5,495 cycles. Receiver
staging in phase 58 similarly adds at least 2,630 cycles relative to the original
exchange, before copy setup and barriers, under this scheduling model. Changes in
addresses and ordering also affect scheduler timing even when fragment counts
hardly change; phase 84 illustrates this.

## SigLIP results

| Phase / representative movement | Alternative | Transfers, original → alternative | Max row bytes, original → alternative | Exchange cycles, original → alternative | Added ideal copy cycles | Staging KiB/tile |
|---|---|---:|---:|---:|---:|---:|
| 50: MLP up FP8 input | Receiver | 62,482 → 5,603 | 17,908 → 1,048 | 33,758 → 31,049 | 11,712 | 91.5 |
| 55: MLP down FP8 weights, transposed AMP → block-major | Receiver | 303,042 → 4,326 | 7,936 → 736 | 14,672 → 12,908 | 3,456 | 27.0 |
| 16: fused QKV FP8 input | Receiver | 54,502 → 4,930 | 12,056 → 952 | 29,826 → 21,737 | 8,784 | 68.6 |
| 54: MLP down FP8 input | Receiver | 196,830 → 17,496 | 11,348 → 1,520 | 10,220 → 9,238 | 4,416 | 34.5 |
| 56: MLP down FP16 partial/result movement | Source | 586,368 → 32,576 | 5,168 → 716 | 16,724 → 15,103 | 6,480 | 50.6 |
| 70: MAP row-major to heads | Both | 139,808 → 129,600 | 8,524 → 7,288 | 11,465 → 9,427 | 5,788 | 44.9 |

The largest row shrinks by 94%, but its full-phase buffer is too large to assume
this helps the existing memory failure. The weight transformation is a better
initial implementation target: 91% row reduction with 27 KiB of staging and
regular local copies (1,609 affine groups across all 1,472 receivers; 1,463
receivers need one group, and the maximum is 18). It still adds
at least 1,692 modelled cycles once the ideal copy work is included. Packing both
ends cuts that row further to 316 bytes but needs 57.5 KiB of scratch and 7,360
ideal copy cycles; that is a poor incremental tradeoff.

The 586,368-transfer phase has the opposite problem to the activation input
phases: source packing helps, receiver staging does not. A blanket policy of
staging all receivers would miss it. MAP again mainly has genuinely distinct
routes and benefits little in row storage.

One interesting timing result is QKV **source-only** packing: it leaves 54,502
transfers, reduces exchange time to 22,174 cycles and adds a 4,128-cycle copy
floor. The row only shrinks from 12,056 to 11,764 bytes. This suggests an
address/order-sensitive scheduling opportunity, but does not establish that a
packing kernel is needed to obtain it. The synthetic placement changes both
addresses and ordering, and a real allocator has not been tested here.

## Initial prioritization (superseded for dependent phases by the follow-up)

1. A bounded, regular receiver-staging alternative for weight-format conversions,
   represented in mid planning with its scratch and local-copy costs. This is the
   clearest capacity tradeoff, rather than a claimed throughput win. For fixed
   resident weights, also compare storing/loading them in the consumer format;
   that could avoid the runtime conversion entirely, subject to replication.
2. Selective staging of the largest fragmented activation transfers if the scratch
   can fit. Full-phase staging often needs more temporary memory than the exchange
   rows it eliminates. Chunked staging or writing the transport format directly
   from the producer would need a separate implementation and measurement.
3. Do not apply the same transformation to the MAP redistribution: its recipient
   structure sets a much higher floor. Changing ownership or transport topology
   would be necessary for a large reduction there.

These results establish that much of the exchange **instruction count** is
avoidable, but not the payload traffic. They do not establish that batch-two
SigLIP fits: that needs complete placement with the selected alternative, including
copy code/descriptors and shared exchange rows. No new planner policy or device
kernel has been introduced by this analysis.

## Method and limits

- Full-phase staging needs fresh scratch. The report gives the maximum combined
  source and destination scratch **on the same tile**, not the sum of independent
  maxima. These are not allocations proven to fit alongside the resident model.
- Copy cycles assume eight bytes per cycle and add the separate source/destination
  critical-path maxima. These are ideal throughput floors, not kernel estimates.
  Kernel setup, barriers, bank conflicts and instruction limitations are excluded.
- The report counts contiguous copy runs and greedy affine-loop groups. This is
  useful for spotting whether exchange-row savings might become large copy
  descriptor tables. It is not emitted copy code; groups are counted over all tiles.
- Source contiguity is checked across every captured Repeat iteration. Source
  packing assumes an appropriate gather runs for each iteration; its first-iteration
  copy descriptors are diagnostic, not a generated Repeat implementation.
- Synthetic address ranges reserve room for staging solely for scheduler replay.
  No payloads run on hardware. Scheduler invariants are checked on every replay.
- Row savings cannot simply be subtracted from the complete shared exchange table:
  different phases peak on different tiles and may share rows.
- Whole-phase staging is a deliberately simple experiment. Chunking could lower
  scratch, but would add ordering constraints, launches or exchanges. This tool
  does not assume that those costs disappear.

## Reproduction

Capture with the existing `ipu-e2e-test --capture-exchange-schedule PATH`
entry point and the desired model/planning flags. Captures now carry optional
phase provenance; older snapshots remain readable. The capture entry point now
uses the same memoized operator cost model as ordinary baseline construction.

The captures here used `--workload siglip-vit-benchmark --vit-batch 2 --fuse-qkv
--fp8-scale=-4 --capacity-baseline --optimization-steps 0 --no-profile`, with
`RAYON_NUM_THREADS=12`. PE additionally used `--vit-model pe-core-l14-capacity
--vit-layers 24`; SigLIP used `--vit-layers 27
--exchange-transfer-limit-per-tile 17000`. The latter override only allowed the
diagnostic capture; it does not make the resulting executable fit. Scheduling
flags below apply to offline replay, independently of baseline selection.

```sh
python3 scripts/exchange-packing.py capture.json analysis --top 4 \
  --scheduler target/release/ipu-exchange-schedule-bench --jobs 4
# Select known expensive phases instead of ranking by endpoint fragments:
python3 scripts/exchange-packing.py capture.json analysis --phase 62 --phase 58
# Replay existing exports without loading and transforming the capture again:
python3 scripts/exchange-packing.py capture.json analysis --replay-only \
  --scheduler target/release/ipu-exchange-schedule-bench
```

`analysis/index.html` renders the metrics; `analysis.json` contains numeric results.
Each exported counterfactual has its scheduler log beside it. By default every
phase is analyzed but only the highest-ranked phases are exported. Explicit
`--phase` restricts both analysis and export. Capturing and transforming these
large JSON snapshots takes minutes and several GiB on the host; this is an offline
diagnostic, not an addition to candidate planning.

Artifacts: `artifacts/exchange-packing-20260912/`, with PE results in
`pe-selected/` and SigLIP results in `siglip-worst/`. The earlier all-phase scans
in `pe/` and `siglip/` were exploratory; use the selected reports for encoded row
comparisons (they also enforce the exchange snapshot's transfer-length limit).

Validation: eight Python tests cover payload reconstruction through gather/transport/
scatter, multicast recipients, Repeat contiguity, transfer-length limits and copy
loop grouping. The Rust randomized capture replay test passes. All 36 selected scheduler replays
(three PE and six SigLIP phases, four modes each) pass scheduler invariants.
