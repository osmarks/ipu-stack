# Optimized materialized ViT exchange frontiers

Report: `artifacts/exchange-frontier-materialized-20260911/combined/index.html`.
The report covers **all 80 distinct exchange phases**, with 1,539 validated
measurements: 19 order/chunk configurations per phase, plus 19 paired-width
alternatives for phase 43. The pages contain standalone SVG/PDF/PNG plots, and
`results.csv` / `frontier.csv` retain all points. The black ring marks a fresh
replay of the original S256/B1024 policy. It is not necessarily the exact cached
recipe used in the executable.

## Capture and interpretation

This is a fresh build of the resident 27-layer, batch-1 ViT with FP8 weights
(scale -4), fused QKV, automatic attention selection and eight local optimization
steps. All seven accepted improvements and their exact placed cycle estimates
match the earlier optimized materialized-attention run (9.625460 ms on hardware).
The capture contains 2,080,513 transfers and preserves all Repeat source
addresses. It packages one host invocation rather than the earlier validation's
three. The unchanged production scheduler selected Word32 transfers throughout.
The independent latency-oriented width comparison selected Paired64 only for
phase 43; this alternative is plotted with triangles and a `p` suffix.

Phases 7–40 are inside the transformer layer and execute 27 times; the other
phases run once. These labels come from the matched run's profiler Repeat epochs.
Schedules are evaluated once with all captured address variants, not separately
for each layer. This capture's phase IDs differ from the old FlashAttention
frontier report.

Cycles are encoded schedule horizons after the barrier, not full-model hardware
measurements. Rows are isolated encoded storage before cross-phase sharing and
Repeat patch support. Neither maximum row changes nor isolated row totals can
be added to predict exact complete-package memory usage. Changes can also affect
row sharing and patch work, so summed horizon savings are not a runtime guarantee.
Eight single-worker configurations run concurrently; their compiler elapsed
times include contention and should not be compared as isolated CPU timings.

## Useful points in the repeated layer

All row sizes below are the largest row in that phase, in bytes per tile. The
reference is the original S256/B1024 policy replay.

| Phase | Candidate | Before cycles | After cycles | Before row bytes | After row bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 18 | S64 | 19,194 | 17,162 | 1,280 | 1,024 |
| 21 | S64 | 5,906 | 5,581 | 1,336 | 1,228 |
| 18 | B256 | 19,194 | 12,307 | 1,280 | 1,288 |
| 31 | automatic | 18,402 | 11,968 | 164 | 196 |
| 38 | B256 / B1024 | 27,456 | 14,529 | 1,596 | 1,788 |
| 23 | remaining-combined | 7,526 | 4,445 | 348 | 428 |
| 17 | B256 | 9,824 | 7,374 | 1,140 | 1,128 |
| 12 | B1024 | 4,487 | 2,258 | 1,056 | 1,240 |

Phases 17–18 belong to the attention operation; phase 31 prepares the MLP
upprojection, and phase 38 prepares its downprojection. These operation mappings
come from the hardware profile, not inferred transfer sizes.

The first two reduce both maximum and total row storage. Phase 17 instead
increases the total from 708,796 to 800,444 bytes, despite reducing the maximum;
the old aggregate cap rejects it. Phase 18's B256 increases total storage from
1,080,596 to 1,128,420 bytes; phase 31's automatic schedule increases it from
149,388 to 197,136 bytes. Small maximum-row changes do not mean every tile's row
changes by that amount.

Across all measured points, requiring both storage metrics not to grow gives
63,943 fewer scheduled cycles when weighted by phase execution count. As an
illustrative comparison window, allowing each maximum row to grow by up to
256 bytes gives 1,166,453 fewer weighted scheduled cycles (0.778 ms at 1.5 GHz).
That window is **not a new production policy or a demonstrated package fit**.
`comparisons.csv` records both calculations; `analyze.py` reproduces the combined
report and comparisons.

## Production change and validation

Compact selection now compares S(W/4), S(W), and B(4W), using the original S(W)
maximum and total row sizes as hard caps. It minimizes horizon, then maximum
row size, then total row size, accepting equal-cycle storage reductions. At the
current setting this adds S64 to S256/B1024. It adds one bounded alternative,
not the full offline sweep to every compiler candidate. Recipe caching and
ordinary/paired selection remain in use.

Identical candidate transfer orders are discarded before physical scheduling and
row encoding. This does not change the selected schedules. The baseline build
reached validation in 82.671 seconds with order deduplication versus 91.425 seconds
without it, at the same 20,163,045-cycle estimate and 40,120-byte exchange table.
Both used eight Rayon threads; these were single observations with other work
running, not an isolated compiler-scaling experiment.

The exchange test suite passed: 37 tests, with two hardware tests ignored.
Six hardware replays passed, checking 8,192 sampled words each: phase 18 S64,
phase 21 S64, phase 18 B256, phase 31 automatic, phase 38 B1024, and phase 43
paired automatic. They check first-iteration payloads, with all captured Repeat
addresses retained in software timing validation. Commands are in
`hardware/results.json`. These are not per-point hardware cycle measurements.

The complete resident 27-layer model passed all three inference calls against
FP32 reference, each with cosine similarity 0.994168165 and maximum absolute
error 0.424608. The renderer-cropped span is **14,374,044 cycles / 9.582696 ms**,
versus 14,438,190 cycles / 9.625460 ms previously: **64,146 cycles (0.444%) saved**.
The final exchange table shrank from **37,744 to 37,384 bytes per tile**. The
selected layout sequence is unchanged; its placed estimate improved from
14,930,507 to 14,866,868 cycles. Runtime and full logs are in
`small-streams-full27/model.html`, `model.ipuprof`, and `run.log`.

There is a compiler cost: the full run took **1,325.535 seconds (22m 6s)** to
plan, versus **1,013.903 seconds (16m 54s)** for the original-policy capture with
the same eight-thread pool. The full run started before duplicate-order
elimination was added; that optimization preserves its selected schedules.
The 82.671-versus-91.425-second baseline comparison above measures the later
optimization, but a complete optimized planning run with it has not been timed.
The full-run comparison also packages three host calls instead of one and uses
hardware validation rather than snapshot export; planning times exclude those
later execution/export steps.

The larger storage trades in the table have passed representative payload
replays, but have not been selected and placed together in a complete model.
They remain excluded by the production per-phase row limits.
