# Layernorm and GeLU kernels, 2026-09-11

## Retained changes

`elementwise_vector.hpp::normApply` handles four aligned FP16 elements per
worker iteration. Statistics, centering and normalization remain FP32; the
last gamma/beta multiply/add uses FP16 vector instructions. Irregular widths
and four-byte-offset pointers keep the pair loop. Distributed layernorm apply
and add-layernorm share the implementation. FP8-output layernorm is unchanged.
Cost estimates and useful-work accounting distinguish these paths.

`gelu_f16.S` and `gelu_f8.S` load each duplicated clamp bound with one `ld64`
instead of two `ld32`s. The arithmetic and its rounding are unchanged. The
supervisor frame grows to 64 bytes; bounds occupy aligned pairs at offsets
40 and 48. The vector body saves eight instructions per sixteen values.
The scalar-pair tail is unchanged. Bias-GeLU shares these macros.

## Hardware measurements

Cycles include the direct kernel call, with six workers.

| Kernel / shape | Before | After |
|---|---:|---:|
| Layernorm, 1 × 1152 | 9378 | 7140 |
| Add-layernorm, 1 × 1152 | 11598 | 8784 |
| Layernorm apply, 1 × 1152 | 6696 | 4686 |
| Layernorm, 3 × 1152 | 27876 | 21162 |
| FP16 GeLU, 1408 values | 7980 | 7260 |
| FP16 GeLU, 2208 values | 12060 | 10956 |
| FP8-output GeLU, 1 × 1152 | 7068 | 6498 |
| FP8-output GeLU, 1 × 2152 | 13656 | 12510 |

Small GeLU tails do not benefit; 94 values regress by six cycles because the
changed vector body changes which worker finishes last.

## Full-model validation

Same batch-1 27-layer fused-QKV FP8-weight ViT, scale -4, eight local optimization
steps and B1024 exchange ordering. Times use the runtime renderer's cropped
range. Three resident invocations check parameters and repeat state survive;
they are not statistical timing repetitions.

| Build | Cropped cycles | Time | FP32-reference cosine |
|---|---:|---:|---:|
| Prior baseline | 13,281,234 | 8.854156 ms | 0.994168165 |
| Layernorm only | 13,158,042 | 8.772028 ms | 0.994130268 |
| Layernorm + GeLU | 13,129,692 | 8.753128 ms | 0.994130268 |

Layernorm-only passes all three invocations. The two transformer layernorms
fall from 9384 to 7146 cycles in the actual profile. The overall improvement is
0.93%; exchange boundaries and the MLP GeLU are unchanged in that comparison.

Artifacts: `artifacts/norm-quads-20260911/` (direct checks and full-model rendered
profile); `artifacts/gelu-broadcast-20260911/packed-check/` (actual FP8 experiment);
`artifacts/gelu-broadcast-20260911/fixed-equivalence/` (FP16 timings and exact
comparison). The directory name predates the final paired-load implementation:
min/max instructions do not support broadcast operands.

The combined build passes all three resident invocations, retains the same
local-search choices, and improves total runtime by 1.14%. Its profiled GeLU
step falls from 13,410 to 12,360 cycles; layernorm stays at 7,146. The additional
28,350-cycle whole-model saving is close to the cost model's 28,512-cycle
prediction. Repeat patching is unchanged: 2,052 arithmetic words across 1,274
calls (maximum 18), and 1,828 row-sharing words across 1,466 calls (maximum 2)
in the profiled layer. This gain comes from the kernels, not fewer barriers.

The final package, profile, rendered HTML, screenshot and queries are in
`artifacts/norm-gelu-20260911/full27/`. The full build uses the frozen
`artifacts/norm-gelu-20260911/source/` source tree. Full planning took 14m35s;
reference preparation and hardware checks took about 27s after loading.

## Validation infrastructure

The codegen library suite passes 235 tests (four ignored), and
`cargo check --workspace --all-targets` passes.

690 elementwise hardware cases cover residual/FP8 variants, in-place output,
constant rows, large means, alignment fallback, tails and guards. The GeLU
comparison passes all 63,488 finite FP16 values bit-for-bit, including additional
in-place and tail cases (204 cases including unchanged reduction controls).

The comparison exposed a diagnostic-package placement bug. Explicit test data
was byte-disjoint from host code but shared its 16 KiB memory element, causing
supervisor memory conflicts. The diagnostic builder now excludes the data's
executable elements before placing code and protects host-code elements before
placing descriptors. The exhaustive test's large data moves to interleaved
memory to leave enough executable space. Timeout diagnostics identify the case,
PC and hardware exception. Both comparison tools honor isolated source trees.
