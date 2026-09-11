# Exchange scheduling frontiers

The report is `artifacts/exchange-frontier-20260911/combined/index.html`, with
standalone SVG/PDF/PNG plots and `results.csv` / `frontier.csv`. It compares
scheduled cycles after the barrier against maximum per-tile row bytes and,
separately, total row bytes. Plot axes focus on the nondominated frontier; the
CSV retains all dominated points, start-time spread and scheduling/codegen time.

## Workload and method

The snapshot is from the address-resolved, resident 27-layer batch-1 ViT baseline
with FlashAttention, pairwise operand placement and compact exchanges:
`artifacts/baseline-local-planner/pairwise-resident-full27/transfers.json`.
Phase IDs refer to that capture, not the latest optimized materialized-attention
profile. Selected phases are 0, 3, 7, 9, 11, 12, 14, 16, 17, 32, 42, 47 and 49.
They include wide multicast, small fan-out, mixed broadcasts, fine-grained
unicast and large rearrangements. `phase-types.json` records their characteristics.

The policies are automatic, combined, directional, remaining-combined and
remaining-directional, plus ordinary and balanced streams at 64, 128, 256, 512,
1024, 4096 and 16384 words per chunk. These are direct ordering experiments;
balanced points are not filtered through production's incumbent row-budget cap.

Widths were selected once using the latency-oriented production policy and then
held fixed across orderings. Phase 9 selected paired transfers; all other selected
phases remained ordinary. To avoid hiding the interaction between pairing and
compact ordering, all 19 configurations were also measured on ordinary phase 9.
There are **266 validated measurements**, with exact addresses, transfer contents
and all captured Repeat source addresses retained.

`scripts/exchange-frontier.py` runs each policy once with one Rayon worker.
Scheduling/codegen milliseconds are single-run elapsed times, separate from
snapshot parsing and validation. They are indicative compiler timings, not
hardware cycle measurements or process CPU accounting. Manifests contain commands.

## Useful tradeoffs

All storage below is maximum row bytes per tile, before cross-phase sharing and
Repeat-patch support. Isolated phase footprints cannot simply be added to predict
full-model memory use.

| Phase | Configuration | Max row bytes | Scheduled cycles |
| --- | --- | ---: | ---: |
| 3 | streams 256 | 284 | 6,258 |
| 3 | automatic | 348 | 3,762 |
| 3 | balanced 256 | 360 | 3,661 |
| 9 | ordinary streams 256 | 1,148 | 17,780 |
| 9 | ordinary balanced 256 | 1,660 | 16,505 |
| 9 | paired remaining-directional | 3,400 | 14,757 |
| 12 | streams 512 | 632 | 15,060 |
| 12 | balanced 512 | 696 | 11,470 |
| 42 | streams 256 | 156 | 18,402 |
| 42 | automatic | 188 | 10,612 |
| 49 | streams 256 | 1,624 | 27,439 |
| 49 | balanced 256 | 1,796 | 14,301 |
| 49 | automatic | 2,292 | 12,609 |

Phase 42 trades only 32 bytes on the largest row for 7,790 cycles. Phase 49 trades
172 bytes for 13,138 cycles, with another 496 bytes buying a further 1,692 cycles.
These are substantial opportunities that the current strict no-row-growth
comparison rejects.

Phase 47 has an improvement in all three metrics: balanced 1024 gives 6,170
cycles, 6,076 maximum row bytes and 2,468,700 total bytes. Streams 256 gives 9,424,
6,316 and 2,506,432 respectively; automatic gives 10,468, 14,868 and 5,098,180.
Its scheduling/codegen time is also lower: about 6.76 s versus 20.35 s for automatic.

Phase 16 illustrates why the two storage metrics differ. Balanced 128 reduces
cycles from 11,164 to 8,023 and maximum row bytes from 4,176 to 4,144, but increases
total bytes from 1,360,840 to 1,372,164. Production's current total-row cap rejects
that small aggregate increase even though the worst row gets smaller.

Pairing is not universally the best storage/performance choice. For phase 9,
ordinary streams 256 save 66% of maximum row storage against the fastest paired
point, at a 20% cycle increase. Tiny paired stream chunks can be dramatically
worse in both dimensions; the full data includes schedules exceeding 100,000
cycles. Conversely, phases 11 and 32 are identical across all tested policies,
and phase 17 buys only 40 cycles by increasing maximum row storage from 212 to
264 bytes. There is no uniformly best chunk size or queue priority.

## Hardware checks and replay initialization

Seven distinct replay cases passed, each checking 8,192 systematically sampled
words: paired phase 9 automatic; ordinary phase 9 balanced 128; phase 16 balanced
128; phase 47 balanced 1024; phase 42 automatic; and phase 49 streams 256 and
balanced 256. Commands and outcomes are in `hardware/results.json`. These are
first-iteration payload checks with the full captured Repeat constraints used
in scheduling, not direct cycle measurements of every frontier point.

Six fixtures initially failed before reaching hardware because their initial
scratch data overlapped the host-exchange aperture. The low-level diagnostic
packager now stages these initial values in independently allocated per-tile
region-1 holes and prepends a supervisor copy after the host handshake. Captured
exchange addresses and rows are unchanged. Initialization finishes before the
exchange barrier, and generated/linked code and host metadata still exclude all
reserved data. A regression covers fragments crossing either aperture boundary,
unaligned byte fragments and multiple fragments on one tile. All eleven package
tests and the workspace all-targets check passed.

The one fixture that already passed was not rerun. The other six passed after
fixing fixture initialization. This diagnostic change does not alter production
model packaging or the measured 27-layer model's plan.
