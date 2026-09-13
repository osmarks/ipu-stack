# Main GEMM effective utilization

The main F16/F8 kernels already overlap AMP arithmetic with paced loads/stores
throughout their steady-state row loops. For the projections and MLPs, nearly
all arithmetic lanes carry real values. Most remaining loss is repeated panel
setup, coefficient loading, and AMP pipeline fill/drain, rather than padding or
an obvious unpaired load/store in the inner loop.

The runtime profiler's useful utilization is logical FLOPs divided by peak FLOPs
per cycle and measured kernel cycles. It is not an instruction counter. Its
separate lane-occupancy estimate excludes physical padding. Per-symbol device
utilization is also affected by splitting one GEMM across row/column
specializations, so it should not be interpreted as whole-phase occupancy.

## Implemented changes

`device/gemm_f16_amp.S` now:

- Retains the panel byte stride in worker m4 after row partitioning, avoiding a
  frame load on every subsequent inner/output panel in the plain-output path.
- Issues accumulator clearing alongside the first output-partial load.
- Selects retained entry labels after the empty-worker test when the physical row
  count proves that all six workers have work. Small-row and packed-output cases
  retain their checks.

These changes reuse the same worker bodies and frame. They add no scratch space
or kernel variants. Constant-time GEMM costing already tracks the resulting
measurements within 2% for the tested plain-output geometries; calibration tests
now record current device measurements rather than the older kernels.

## Matched 27-layer hardware results

Same saved BS1/BS2 plans as the base-aware exchange results, B1024 schedules,
FP8 projections/MLP/PV, and FP16 QK. Runtime uses the viewer's initial-sync crop.

| Batch | Before | After | Improvement |
|---|---:|---:|---:|
| 1 | 10,446,450 cycles / 6.964300 ms | 10,371,582 / 6.914388 ms | 0.72% |
| 2 | 17,176,698 cycles / 11.451132 ms | 16,985,046 / 11.323364 ms | 1.12% |

Representative BS1 per-layer kernel specializations, using local R/K/C extents:

| Operation | R/K/C | Kernel cycles before → after | Useful utilization before → after |
|---|---|---:|---:|
| Fused QKV projection, FP8 | 244/128/64 | 18,822 → 18,576 | 83.0% → 84.1% |
| MLP up-projection, FP8 | 82/192/160 | 30,510 → 29,508 | 64.5% → 66.7% |
| MLP down-projection, FP8 | 122/160/128 | 26,886 → 26,232 | 72.3% → 74.1% |
| Attention output projection, FP8 | 146/96/48 | 7,248 → 7,122 | 72.4% → 73.7% |
| QK, FP16 | 81/80/80 | 12,804 → 12,402 | 56.9% → 58.8% |
| PV, FP8 | 57/128/80 | 8,436 → 8,124 | 46.2% → 48.0% |

This excludes the one-time image projection and MAP head. Other tiles use smaller
row or column specializations. QK and PV have additional padding losses: the
listed cases have approximately 90% and 85% lane occupancy. Projection/MLP
occupancy is approximately 99–100%.

Both resident inference calls pass at each batch size. Embeddings are byte-for-byte
identical to the pre-change packages. Minimum cosine similarity against FP32
remains 0.994352454 at BS1 and 0.994151272 at BS2. Separate native-output and
packed-output GEMM checks also pass; the packed R128/K64/C64 kernel takes 19,872
cycles. All 22 kernel-related codegen tests pass.

## Where a larger improvement could come from

The coefficient store exposes 64 64-bit CWEI entries: one complete FP8 K32/C16
panel, or F16 K16/C16 panel. The current interleaved feed uses 32 `ld128putcs`
instructions per panel. There is no second complete coefficient bank available
to this kernel for ordinary double buffering. The coefficients are shared by all
six workers, so overwriting them while workers still consume them requires a
new, carefully coordinated execution scheme.

For the MLP up-projection example, each tile executes 60 coefficient panels.
Its physical AMP work is 19,680 cycle-equivalents; the measured kernel takes
29,508 cycles. The approximately 9,800 remaining cycles include worker setup,
pipeline seeding/draining, coefficient feed, and supervisor coordination. The
steady-state AMP loop itself is already full.

The most promising larger knob is amortizing this cost over more rows. For
example, a 9-row-partition / 6-K-partition geometry could become
3-row-partition / 18-K-partition, retaining the same column partitioning and
number of compute tiles. Local geometry changes roughly from R82/K192/C160 to
R244/K64/C160: 60 panels become 20. The simple kernel model suggests roughly a
22% GEMM-only reduction in cycles.

That is an untested layout proposal, not a predicted model speedup. Each local
FP16 partial output grows from about 25.6 KiB to 76.3 KiB, and three times as many
K partials must be reduced. The smaller weight staging buffers offset some of
the local memory increase. Placement, complete reduction cost, and exchange
traffic can erase the kernel gain. Existing cost models can screen this tradeoff;
it should be evaluated as a complete GEMM/reduction region, particularly near the
27-layer memory limit.

Retaining accumulators across K panels is also not a free way to remove the
partial reloads. The current schedule loads a coefficient panel once and applies
it to all local rows. Keeping only a small set of output rows live in accumulators
would require revisiting/reloading the coefficients for additional row groups,
unless the coefficient/worker execution scheme is redesigned. I would evaluate
row-heavy plans and complete reduction fusion before attempting that redesign.

## Artifacts

- `artifacts/gemm-utilization-20260913/bs1-final/`: final BS1 package, build log,
  saved resident tensors, `model.capnp`, incremental `model.html`, and query JSON.
- `artifacts/gemm-utilization-20260913/bs2/`: equivalent BS2 artifacts.
- `artifacts/gemm-utilization-20260913/bs1/`: intermediate stride/clear optimization
  before the nonempty entry labels (10,398,354 cycles).
- `artifacts/gemm-utilization-20260913/check-kernels.sh`: additional hardware checks.
- Before profiles: `artifacts/exchange-base-aware-20260913/{final,bs2}/`.
