# Fused FP8 PV preparation

Materialized FP8 PV now obtains probabilities directly from softmax in FP8
AMP-left order. The softmax arithmetic and FP32 maximum/denominator remain
unchanged. Full panels convert eight probabilities at once in registers;
only the masked final panel uses a bounded 32-byte FP16 temporary per row.
The packed buffer reserves 64 extra bytes per row for persistent statistics,
segmented reduction scratch and that temporary. Merge specializes the byte
offset to the FP32 statistics separately from its output precision.

V preparation packs unreplicated 64-row native panels across its available
tiles, then casts them before query replication. The existing cast and layout
conversion kernels do the work. The distributed-product builder preserves
already-quantized operand precision instead of forcing an FP16 intermediate.
FP16 PV retains its previous preparation. This remains selected through the
experimental `--attention-pv-fp8-scale` option; the default is FP16 PV.

Candidate generation now excludes K grids whose rounded final shard is empty.
For 729 keys, seven FP16 partitions use 112 elements each; switching that same
grid to FP8 rounds each partition to 128, putting the seventh beyond the data.
The fixed-layout experiment therefore uses six FP8 K partitions, retaining
13 query partitions. Other operator layouts are loaded from the same saved
BS1 recipe. Only encoder PV changes precision; MAP remains FP16.

## Full 27-layer resident hardware comparison

FP8 weights at scale -4, fused QKV, B1024 exchanges, first repeated layer
instrumented. Times use the renderer's normal crop. No extra layout-search
steps; the scripts and edited experimental checkpoint are saved with artifacts.

| Variant | Cropped cycles | Time | FP32-reference cosine |
|---|---:|---:|---:|
| FP16 PV, producer-output changes and seed bypass | 10,996,980 | 7.331320 ms | 0.993916310 |
| FP8 probabilities, late V cast | 11,081,280 | 7.387520 ms | 0.994352454 |
| FP8 probabilities, distributed packing + early V cast | 10,666,374 | 7.110916 ms | 0.994352454 |

The retained FP8 implementation is 3.01% faster than the matched FP16 control;
it is 3.05% faster than the 11,001,942-cycle model before this work. Both
resident inferences pass in every full-model variant. These are randomized
model/input checks; a slightly higher cosine is not evidence that FP8 improves
accuracy.

The current pass still does not select FP8 BiasGeLU in this saved layout; its
kernel and capability are implemented, but the alternative preparation does
not win the estimate. The FP8 PV improvement should not be attributed to it.

## Complete attention preparation, not just GEMM

Selected phase spans below belong to the encoder attention operation. Spans
can overlap when different kernels share a phase, so do not add these rows.

| Component | FP16 PV | FP8, late V cast | FP8, early V cast |
|---|---:|---:|---:|
| Exchange phase spans | 81,462 | 81,456 | 66,558 |
| Softmax phase | 25,602 | 28,506 | 28,500 |
| V cast phase spans | absent | 10,542 | 5,220 |
| Long PV GEMM tile kernel | 14,364 | 8,436 | 8,436 |

The QK kernels do not change, but some phases containing them shorten because
other preparation no longer holds them up. Maximum per-tile exchange table
storage is 39,100 / 38,836 / 38,652 bytes respectively. The win is chiefly
execution time, not a large memory-capacity improvement.

## Validation and artifacts

`softmax_check --fp8-output` checks packed probability coordinates, numerical
probabilities, FP32 state and output canaries against the original FP16 source.
Tests cover 1/2/6/7 rows, keys 1/2/15/16/17/31/64/65/729/768, whole and segmented
schedules, and random, constant and extreme finite scores: 124 cases in total.
The arithmetic denominator is intentionally not recomputed from quantized
probabilities. Its comparison tolerates the separately measured quantization.

The codegen suite passed 283 tests with five ignored and one assertion expecting
a seed copy that the new reducer removes. The updated test verifies the direct
reducer's Repeat pointer instead and passes. Attention contract/recipe tests,
FP8 attention expansion (including unreplicated V casts), and eight kernel-cost
tests pass after the changes. Source/consumer output-fusion tests also pass.

Artifacts: `artifacts/producer-fp8-20260913/`. `seed-full/` is the FP16 control,
`pv-full/` preserves the slower probability-only experiment, and `pv-early-full/`
contains the retained implementation's package, rendered `model.html`, raw
profile, exact memory profile and resident inputs/output. Corresponding `.sh`
and `.log` files record invocation and validation. `pv-state.json` records the
encoder PV precision/grid change.
