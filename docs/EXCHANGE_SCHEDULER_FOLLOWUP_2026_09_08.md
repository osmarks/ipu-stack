# Exchange scheduler follow-up

Baseline: `f7052f8` (the implementation benchmarked in
[the first report](EXCHANGE_SCHEDULER_BENCHMARKS_2026_09_08.md)).
Artifacts, saved benchmark binaries and logs: `artifacts/exchange-followup/`.

## Implemented changes

- Index SRAM send/receive hazards by memory element. Intervals are ordered by
  start and store prefix maximum ends, allowing exact overlap queries without
  scanning an endpoint's complete history. Out-of-order and nested intervals
  retain the original behavior; adjacent intervals are not coalesced.
- Decode each primitive transfer once. `MulticastPlan::prepare` borrows the
  patched rows, so they cannot change behind cached timing. Offset search,
  timing queries and insertion all use that preparation. Existing primitive
  decoders and the full-row encoding oracle remain authoritative.
- Borrow event slices lying within one shared chunk instead of allocating
  temporary vectors. Cross-chunk slices still allocate.
- Reduce ready-queue key size and replace a consumed group head directly with
  its successor instead of popping and pushing the global heap separately.
- Refresh the whole ready heap sooner when many keys have become stale. Both
  readiness and pressure remain monotone priority bounds; this changes compiler
  effort, not the selected order. The comparison against eager selection now
  includes larger endpoint sets as well as the earlier small cases.

The first four changes are in `1f3e68e`; the heap refresh change is `fc89fcc`. None changes exchange instructions,
required timing gaps, transfer grouping, or table-size policy. Whole-phase
strict retries and broader physical-schedule reuse remain separate work.

## Profile and measurements

The 391,310-transfer B2 phase's baseline profile attributed 40.92% of samples to
`BinaryHeap<ReadyTransfer>::sift_down_range`, 15.31% to the greedy scheduler body,
and 5.03% to `earliest_transfer_offset_impl`. SRAM queries and primitive decoding
were not the dominant remaining cost. Profile: `before.perf.data`, 499 Hz,
4,088 samples. Its instrumented wall time is not used as a benchmark baseline.

Single-thread replay samples, with individual processes pinned to separate CPU
cores and independent runs allowed to overlap:

| Change | B2 phase 31, 391,310 transfers | B4 phase 33, 1,172,736 transfers |
|---|---:|---:|
| Baseline | 44.56 s | 74.96 s |
| Indexed hazards | 45.29 s | 71.56 s |
| Also prepared timing | 42.08 s | 69.97 s |
| Also borrowed slices and smaller heap operations | 42.69 s | 72.78 s |
| Also earlier whole-heap refresh | 37.40 s | 62.28 s |

These are CPU wall times, excluding JSON parsing and post-schedule validation.
The small differences between intermediate variants are not robust evidence of
individual speedups. MLP and attention are useful correctness fixtures here;
their timing samples vary enough that no speedup is claimed for them.

A further comparison ran both versions sequentially on CPU 12, with two
iterations per process. Baseline schedule times were 31.762–40.220 seconds;
with all changes they were 27.270–28.750 seconds. The faster sample improves
by 14.1%; the two-sample means improve by 22.2%. This is still a small CPU
sample, not a confidence interval. Logs: `controlled-{before,refresh}.log`.

## Validation

- Full codegen/exchange suite: 165 + 50 tests passed, six manual/ignored tests;
  the codegen doctest also passed. Clippy passed for codegen, exchange and tests.
- `check-results.py` compares both final follow-up variants against the previous
  implementation on 27 phases: all 4 MLP phases, 21 attention phases, and the two
  large ViT phases. Every horizon, total/max row size and row fingerprint agrees.
- Small B2 FP8 ViT packages from both `1f3e68e` and `fc89fcc` have identical
  tile images, bindings and profile metadata to the previously hardware-validated
  accepted package. The final comparison is in `package-comparison.log`.
  No duplicate deterministic hardware run was needed.
