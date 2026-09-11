# Attention grids, residual statistics, and bounded follow-up experiments

Baseline: `1ec0e5c`, 27-layer batch-1 ViT at 13,129,692 cropped cycles
(8.753128 ms). FP8 weights, FP16 attention arithmetic, fused QKV, B1024.
No more aggressive attention quantization is introduced here.

## Intermediate product partition counts

The existing independent attention product search used query partitions
`1,2,4,8,16` and PV inner partitions `1,2,4,8`. It now enumerates all integers
within those same upper limits. This retains every previous candidate and uses
the same complete mid implementations and cost evaluation.

A controlled projected-attention benchmark uses batch 1, materialized attention,
fused QKV, FP8 scale -4, four local optimization steps, and B1024. Both binaries
use the same frozen device sources. Each executes once on hardware and passes
reference validation (maximum absolute error 0.000061).

| Metric | Old enumeration | Intermediate counts |
|---|---:|---:|
| Cropped execution cycles | 221,250 | 216,444 |
| Cropped execution time | 147.500 us | 144.296 us |
| Maximum QK kernel cycles | 13,938 | 12,798 |
| Maximum PV kernel cycles | 16,620 | 15,228 |
| Active PV tiles | 1,024 | 1,440 |
| Package planning wall time | 135.940 s | 135.607 s |

The 2.17% whole-benchmark improvement includes exchange/layout changes, not
just greater PV occupancy. PV now uses 48/49 query rows and inner width 128,
rather than 91/92 rows and inner width 96. QK uses 81-row blocks. Build times
were measured concurrently and are not a precision compiler benchmark.

Artifacts: `artifacts/attention-grids-20260911/{baseline,dense}/`, with the old
binary, source snapshot, logs, profiles and queries retained in the parent.

## Feature-partitioned residual statistics

The original apply kernel combined `(mean, centered squared-deviation sum)`
using its own output width as each statistics partition's width. That was valid
for its original equally partitioned apply, but wrong when complete output rows
consume statistics from smaller producer partitions.

Moments now store `(mean, variance)`. For equal-size partitions, the merged mean
is the mean of partial means, and the merged variance is the mean of
`partial_variance + (partial_mean - merged_mean)^2`. The apply width is no longer
part of this calculation. This avoids another ABI argument or temporary type.

The residual fusion can retain a partition dimension in its FP32 statistics,
collect those small vectors on the normalization owners, and apply them to the
existing redistributed residual. Residual lifetimes/outputs remain explicit.
Only equally sized, unpadded feature partitions are eligible; uneven partitions
would need counts or weights. The change does not fuse across Repeat iterations.

690 direct hardware cases pass, including multi-part statistics applied to a
complete row, constant/large-mean inputs, offset/in-place outputs, FP8 variants,
and guards. The mid regression also expands, validates, places and materializes
a two-feature-partition residual followed by whole-row apply. Both the live
residual and the statistics keep their own allocations. All 235 codegen tests
pass (four ignored).

Direct width-1152 kernel changes from storing normalized variance:
moments 3,366 -> 3,384 cycles, fused add/moments 4,050 -> 4,068, apply with one
part 4,686 -> 4,644. This is primarily a representation/capability correction;
it does not itself establish a large end-to-end performance benefit.

The canonical ViT fusion trace reaches the second transformer layernorm and
prices the separate add/LN at 8,211 cycles versus 8,632 for fused moments,
collection and apply. It correctly retains the separate path. The faster
layernorm from the preceding change makes this tradeoff less favorable. This
trace is in `artifacts/attention-residual-20260911/trace.log`; the debug event
is retained for future rejected-fusion investigations. Eligibility uses the
existing resolved-layout partition extents, including grouped padding, rather
than a separate approximation of tiling rules.

## Full 27-layer hardware result

| Build | Cropped cycles | Time | FP32-reference cosine |
|---|---:|---:|---:|
| Previous kernel upgrades | 13,129,692 | 8.753128 ms | 0.994130268 |
| Broader attention grids and statistics support | 12,836,406 | 8.557604 ms | 0.994102280 |

All three resident invocations pass, including the >0.99 cosine requirement.
Runtime improves 2.23%. The profiled layer selects the new QK/PV grids; maximum
QK and PV calls take 12,804 and 15,234 cycles respectively. Softmax remains
23,706 cycles, and both transformer layernorms remain 7,146. Partial residual
statistics fusion is not selected, so it is not credited with this improvement.

The same eight local optimization steps now reach additional changes around
the projection views/preparation. This is a complete-plan comparison, not an
isolated estimate from GEMM kernel timings. Planning took 944.736 seconds.
The package, profile, rendered HTML, screenshot and queries are under
`artifacts/attention-residual-20260911/full27/`; kernel sources are frozen in
its sibling `source/`. The later eligibility checks do not change this model's
uniform, unpadded two-feature-partition residual geometry.

## QK row maxima: measured optimistic bound

An isolated softmax source removes all score reads and maximum arithmetic,
substituting zero for the known maximum of constant-zero test inputs. The
original and modified kernels both pass the existing probability checker.
This is a timing bound, **not** a general softmax implementation.

| Rows / valid keys / padded keys | Original | Free maximum | Saving |
|---|---:|---:|---:|
| 7 / 729 / 768, split workers | 20,226 | 16,578 | 3,648 |
| 8 / 729 / 768, split workers | 20,244 | 16,596 | 3,648 |
| 7 or 8 / 729 / 768, whole rows | 25,350 | 20,166 | 5,184 |

Production costs select split workers for these shapes. Removing that scan saves
about 0.75% of a 486k-cycle transformer layer before accounting for QK epilogue
work and partial-max collection. The experiment retains launch/local-max control
work; a fully integrated implementation could also remove some of that setup.
It would need maxima per query over QK's key partitions, then a max reduction
alongside the score redistribution. The score GEMMs have no such extra output
or epilogue currently. This is not a promising several-percent optimization on
its own, so no production kernel or mid operation was added.

Artifacts: `artifacts/row-max-20260911/{check,split-check,source}/`.

## Mixed paired-transfer selection

A bounded offline experiment uses the resident materialized-ViT capture in
`artifacts/exchange-frontier-materialized-20260911/transfers.json`, retaining
all Repeat addresses. The six largest payload phases with legal paired
alternatives are replayed with B1024. The subsets are:

- ordinary and all eligible paired transfers;
- pair only when the source's companion has no sends;
- pair only the heavier sender of each physical pair (lower tile breaks ties);
- pair when destination receive pressure exceeds twice the companion's send load.

Duplicate subsets are omitted. All 19 schedules pass the production timing,
encoding, dependency and SRAM-hazard validators. No mixed subset improves on
the ordinary schedule. Most eligible transfers in the large phases are only
32 words; reduced payload bounds do not translate into shorter encoded rows.

| Capture phase | Ordinary cycles | Dominant-sender subset | All paired |
|---|---:|---:|---:|
| 0 | 6,634 | 16,017 | 18,866 |
| 9 | 16,784 | 26,102 | 28,841 |
| 21 | 5,510 | 10,800 | 11,410 |
| 43 | 8,266 | 8,975 | 8,131 |
| 49 | 162 | 348 | 268 |
| 74 | 2,258 | same as all paired | 2,382 |

The all-paired phase-43 exception saves only 135 cycles and raises maximum row
storage from 748 to 1,888 bytes per tile. These are scheduling results on an
older fixed capture, not latest-model hardware timings. Since no new subset
wins, none is integrated and no additional hardware replay is needed to choose
between them. This is not an exhaustive search over subsets.

Artifacts and reproducer: `artifacts/mixed-pairing-20260911/`.
