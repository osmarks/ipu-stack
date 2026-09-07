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
