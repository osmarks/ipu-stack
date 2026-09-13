# Exchange costing and GeLU ownership, 2026-09-13

## Model change

Exchange estimates now separate payload bandwidth (4 bytes/cycle) from
supervisor control issue work, taking their maximum rather than charging 160
cycles for every logical fragment. Low costing counts send controls and receive
source/neutral controls, plus pointer resets inferred from relative allocation
identities and offsets. Consecutive receive addresses retain the pointer;
strided rows count resets arithmetically using the existing compressed geometry.
Multicast payload is counted once at the sender. Ordinary transmitters have
independent lanes; they no longer pessimistically share a lane with their neighbor.
TX/RX controls on the same tile share supervisor issue capacity.

The mid estimate lacks concrete receive addresses and uses three receive
controls per estimated fragment. It retains the existing footprint calculation;
changing cycle pricing does not remove row-storage screening. The local
pack-versus-word-exchange decision uses the same payload/control price. This
changes some concrete copy realizations even when a saved mid recipe is fixed.
Low phase transport comparison also no longer walks geometry twice to account
for the incorrect shared-TX assumption.

This is a resource-work estimate, **not a schedule or a guaranteed bound**.
Transfer order can change pointer continuation, paired mode depends on placement,
and route latency, bank conflicts and dependency chains are not predicted.
There is no new scheduling search, retry layer, or global correction multiplier.

## Calibration

Artifacts are under `artifacts/exchange-cost-20260913/`: benchmark scripts and
JSON, build logs, `calibration.csv`, `calibrate.py`, and `calibration.svg`.

48 logged phases from the saved BS1 recipe and 47 from the GeLU alternative
were matched to their final scheduled horizons. Many phases are identical
between these builds; these are not 95 independent workloads. Scheduled values
below include the model's 600-cycle phase overhead, not hardware measurements.
The comparison with the old fragment penalty uses the same new geometry and
independent TX lanes, isolating the penalty rather than reproducing the entire
old compiler.

| Exchange | New estimate | Scheduled | Old fragment penalty |
|---|---:|---:|---:|
| Bias preparation | 2,752 | 3,331 | 39,040 |
| Original GeLU → down projection | 10,360 | 10,769 | 21,440 |
| Bias-owned GeLU → down projection | 10,360 | 10,765 | 39,040 |

The two down-projection inputs both have a maximum 39,040-byte endpoint payload,
but 134 versus 244 fragments. Their schedules differ by four cycles, not the
17,600 cycles implied by the old fragment charge.

For the saved recipe, median estimate/scheduled is 0.827; median multiplicative
error improves from 1.70× to 1.21×. The new estimates range from 0.40× to 0.98×
actual scheduled duration. In particular, the original GeLU redistribution is
still underestimated (1,676 versus 4,178 cycles). Conflict/ordering prediction
remains an opportunity; these numbers do not justify hard feasibility bounds.

Release/native-ISA runs with 24 Rayon threads:

| Recipe | Mid costing | Low expansion including cost | Warm low re-cost | Footprint |
|---|---:|---:|---:|---:|
| BS1 original | 18.4 ms | 8.54 s | 1.30 s | 1.64 s |
| BS1 bias-owned GeLU | 17.1 ms | 8.42 s | 1.29 s | 1.60 s |
| BS2 saved | 18.5 ms | 11.78 s | 1.71 s | 2.25 s |

Prior warm low re-cost was about 1.2 s / 1.6 s for BS1 / BS2. Some tests ran
concurrently with these measurements and copy realizations changed, so this is
an iteration-cost check, not a clean microbenchmark of estimator overhead.

## GeLU selection and hardware

The saved extended BS1 recipe puts bias output on 729 row groups × 2 channel
groups (1,458 tiles), then redistributes to 1,472 flat GeLU owners. Opening
boundary 360 and reselecting GeLU retains bias ownership and removes that
redistribution. Existing proposal generation already produces this alternative.

Previously the coarse cost rejected it by 281,637 cycles. Now it predicts a
67,095-cycle improvement (13,118,410 → 13,051,315); unscheduled low costing
predicts 77,166 cycles (10,456,164 → 10,378,998). The local-search trace confirms
that this proposal is unvisited and passes the coarse gate. A limited shortlist
can still prioritize other proposals ahead of it. The eight-step resumed search
finished at 71 attempts in 396 seconds and accepted a different layout change
(operation 12, boundary 350), reducing its placed estimate from 11,581,131 to
11,559,423 cycles. It did not select the GeLU ownership change within that budget.
The GeLU hardware experiment below loads that alternative explicitly; it is not
a claim that the short search found it. The search result was compiled but was
not separately hardware-tested.

Both recipes were built with the new compiler and run once on hardware:
27 layers, BS1, fused QKV, FP8 weights with scale -4, B1024 exchange ordering,
FP32 reference, profiling of the first repeated layer. Profile times use the
renderer's cropped origin. This isolates the ownership change from the earlier
reduction bypass and from the new copy realization choices.

| Measurement | Original GeLU ownership | Bias-owned GeLU |
|---|---:|---:|
| Cropped full-model cycles | 11,337,126 | 11,216,910 |
| Cropped full-model time | 7.558084 ms | 7.477940 ms |
| GeLU phase span | 14,328 | 11,928 |
| Mean tile kernel cycles | 11,486.5 | 11,388 |
| Maximum tile kernel cycles | 12,360 | 11,388 |
| Kernel estimated useful utilization | 55.7% | 56.7% |
| Device estimated useful utilization during GeLU | 44.6% | 53.6% |
| FP32-reference cosine | 0.994264800 | 0.994264800 |

This is a **1.06% full-model improvement**. The new GeLU kernel time is uniform
across all 1,458 participating tiles; the remaining 540-cycle phase spread is
start skew. Its estimate is 11,382 cycles, six below measurement. The old flat
ownership includes irregular work lengths and slower scalar-pair tails. The
arithmetic kernel itself is unchanged.

Rendered profiles: `base-profile/model.html` and `gelu/model.html` in the artifact
directory; corresponding packages, raw profiles, reference data and exact memory
profiles are alongside them. The earlier historical extended-search profile was
11,499,450 cycles, but it also predates the reduction bypass and this cost-model
change, so its 2.46% difference must not all be attributed to GeLU ownership.

Bias–GeLU fusion is considered with the compatible ownership, but still rejected:
14,034 estimated cycles separately versus 15,726 fused. `GELU_WITH_BIAS` loads
and adds the bias twice: once before the clamped polynomial and again to recover
unclamped x+b for the final multiplication. It also reloads polynomial constants
and performs per-row setup. Fusion avoids an add pass but increases inner-loop
work. This rejection has not been independently benchmarked with a forced fused
MLP kernel. Improving that kernel is a distinct task from making ownership
compatible. GeLU itself now occupies about 2.7% of total execution cycles, so
its remaining non-arithmetic overhead is not a large whole-model bottleneck.

## Validation

The full codegen test run passed 281 tests and found one obsolete cost assertion:
it expected 256 fragmented bytes to cost more than an 8-KiB contiguous transfer.
That test now compares equal 8-KiB payloads. All 23 estimator tests then passed,
including the corrected case and a new payload/control, pointer-continuation,
and independent-neighbor test. Existing randomized geometry checks compare
compressed receive-control counts against enumerated byte spans. Both full-model
hardware/reference checks passed. No numerical kernels changed.
