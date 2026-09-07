# Estimated useful work in execution profiles

Newly built profiled packages carry useful issue-cycle equivalents for each
instrumented kernel sample. The HTML renderer shows a kernel table and timeline
tooltips; `profile-query` includes the estimates in text and JSON. Existing
profiles remain readable, but have no estimates: kernel names alone do not
identify the unpadded work.

For a sample with measured duration D, useful work U and physical work P:

- Useful efficiency is U/D. For GEMM this is conventional kernel MFU.
- Work-cycle fraction is P/D. It excludes setup, addressing, coefficient loads,
  worker imbalance and other overhead from its numerator.
- Lane occupancy is U/P: their product is U/D.
- Device useful utilization divides U by the kernel's interval union times the
  number of profiled tiles. This additionally exposes idle tiles and imbalance.
  Partial estimate coverage makes it a lower bound.

Aggregation sums cycles before division, never averages percentages. Consecutive
calls can have different geometries; their work is summed individually. Samples
cut by the initial-sync crop are excluded from estimates because we cannot locate
the useful instructions within the cut. Coverage reports the measured tile cycles
with estimates, so unsupported work is not silently treated as zero.

## Bases

| Kernel | Useful issue-cycle equivalent |
| --- | --- |
| GEMM | Logical FLOPs / 128 (FP16), 256 (FP8), or 32 (FP32), per IPU21 tile-cycle |
| FP16 GeLU | 12 arithmetic instructions per four values: ten v4 operations, two v2 tanhs |
| Add / sum reduction | Adds / four FP16 or two FP32 lanes |
| FP16 ↔ FP32 conversion | One conversion instruction per two elements |
| Rearrangement | Logical bytes / 4 bytes per cycle: a dense 64-bit load and store take two issue slots |
| Attention softmax | Eight arithmetic/conversion instructions per key pair, plus five per row |
| Padding-only initialization | Zero useful work |

These definitions are intentionally separate from full kernel latency estimates.
For example, coefficient-loading overhead belongs in the GEMM denominator, not
its useful-work numerator. Arithmetic that coissues with memory movement counts
once. The non-GEMM rates describe the current arithmetic algorithm, not a universal
minimum for computing that mathematical function. A better approximation can
reduce runtime without increasing this percentage.

Known shard padding is excluded. Attention's GEMMs additionally retain valid
contraction and output-column bounds because scratch tensor shapes themselves
include padding. Tests check total FLOPs against the original mathematical
attention for streaming and materialized strategies, including incomplete key
blocks. Rearrangement estimates use the declared logical tensor domain; zeros
explicitly represented inside that domain cannot currently be recognized as
semantically unnecessary.

Raw local byte copies lack logical padding information, and aggregate Repeat
samples contain both computation and exchange: these currently report unavailable.
The merge kernel and monolithic attention kernels also remain unmodelled. No
estimate is fabricated for them. Estimates above 100% are displayed unchanged,
which makes rate/geometry mistakes visible instead of hiding them by clipping.

The model is in `crates/ipu-codegen/src/package/profile_work.rs`. Adding a kernel
requires a useful-work numerator and an explicit basis; it does not require
running code generation during planning or changing the exchange scheduler.

## Hardware check (2026-09-07)

A fresh 1,472-tile SigLIP streaming-attention profile passed all 839,808 output
checks and took 355,320 cycles after the renderer's initial-entry crop. Selected
cycle-weighted kernel estimates:

| Kernel group | Useful efficiency | Lane occupancy |
| --- | ---: | ---: |
| Large-row projection GEMM, K384/C16 | 72.4% | 99.0% |
| Small-row attention QK, K80/C64 | 13.0% | 85.4% |
| Small-row attention PV, K64/C80 | 12.9% | 85.4% |
| Full-block softmax | 51.8% | 100.0% |
| AMP unpack, full rows | 8.5% | 100.0% |
| AMP unpack, padded row tail | 4.2% | 56.2% |

The attention GEMMs' low efficiency is largely outside AMP arithmetic, rather
than simply wasted lanes. This makes the distinction that active-tile occupancy
alone concealed. The copy figures use the dense-copy baseline, not AMP MFU.

Artifacts: `artifacts/useful-work/attention/{execution.ipuprofile,profile.html,summary.json,kernel-work.png}`.
Headless Chromium rendered all 22 groups with values matching `profile-query`;
kernel selection now focuses the timeline while retaining the full comparison
table. Clicking the selected kernel again clears focus without resetting zoom. The codegen test
suite passed (131 tests plus the new attention FLOP regression; one pre-existing
ignored test), all 11 profile tests passed, and Clippy passed for codegen,
profile, and CLI.

The batch-1 MLP also passed hardware validation (maximum absolute error 0.001221)
and took 175,578 cropped cycles. Its GEMM groups show 84.6–89.3% useful efficiency,
GeLU 54.7%, and reductions 62.4%. Their useful lane occupancy is 99.2–100%.
Artifacts are under `artifacts/useful-work/mlp/`. Both completed hardware workloads
were measured once. An initial MLP build used the benchmark's batch-4 default and
was stopped during CPU exchange-placement work; its log is retained under
`artifacts/useful-work/mlp-b4-build/`.

Reproduce the MLP case with `ipu-trivial-test c600-init.ipucfg --workload
siglip-mlp-benchmark --mlp-batch 1 --package model.ipuexe --profile-output
execution.ipuprofile`; the attention case uses `--workload
siglip-attention-benchmark --attention-strategy flash` instead.


Materialized attention passed the same 839,808-element hardware check (maximum
error 0.000930), taking 320,256 cropped cycles versus streaming's 355,320. The
cycle-weighted QK and PV kernel groups achieved 11.9% and 12.1% MFU, respectively,
with 85.4% useful lane occupancy. Softmax achieved 57.4% useful efficiency;
projection and unpacking figures matched the streaming case. The GEMMs still
assign only seven or eight query rows to each tile, so full score materialization
does not resolve the short-row arithmetic efficiency problem.

Artifacts are under `artifacts/useful-work/materialized/`. All three HTML profiles
were regenerated with the selection fix. Chromium checks verified that selecting,
switching, and deselecting kernels preserves all 21 materialized-attention table
rows and the current zoom.
