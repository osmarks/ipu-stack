# Bias–GeLU with MIX and aligned loads, 2026-09-13

The compatible 1,458-tile GeLU ownership from the exchange-cost experiment can
now profitably fuse its bias add. No new mid operation or planner exception is
needed: the existing BiasGelu candidate becomes cheaper with the updated kernel
and instruction-based cost.

## Kernel

`f16v4mix` computes `a*x + b*y`, taking its scalar coefficients from `TAS` and
writing FP32 accumulators. Its ARF destination receives the **previous**
accumulators; `f16v4gacc` reads the new result as FP16. See IPU21 ISA 1.3.1,
§3.7.3.3.24 and §3.7.3.3.15.

For `z = x + bias`, retain unclamped z in `a4:5`, clamp a working copy to [-8,8],
and compute its square and cube. TAS holds half(alpha*beta) and half(alpha),
where alpha is the existing half approximation of sqrt(2/pi) and beta the
existing half approximation of 0.044715. MIX forms `alpha*beta*z^3 + alpha*z`.
This replaces three arithmetic issues with MIX and GACC while freeing the two
coefficient registers. The final multiplication uses retained unclamped z;
it no longer reloads x and bias, re-adds bias, or reloads coefficients each quad.
MIX's mandatory read of old accumulators means each worker must clear AACC at
entry, even though that destination is dead. TAS and accumulators are per-worker;
no general AMP coefficient-memory setup is required.

The changed coefficient product and rounding are intentional. The scalar-pair
fallback retains the earlier polynomial; its coefficients are restored once at
tail entry, not on every pair. Ordinary GeLU and FP8-output GeLU retain their
previous arithmetic.

`ld64` requires an eight-byte-aligned address and even ARF register pair (ISA
§3.7.5.3.9). A runtime check tests input, bias and output addresses, and that each
row contains whole four-half quads. Compatible rows use `ld64step`/`st64step`
with six-worker strides and a hardware repeat loop. Bound loads dual-issue with
bias addition and clamping: fifteen issue groups per quad. Four-byte-offset
views and widths such as 94 retain the existing narrow loop. No stronger
allocation or view-contiguity requirement was added.

## Hardware

Direct-call cycles, including setup. Baseline is the kernel before this change.

| Rows × width | Old fused | MIX, stepped/dual-issued aligned path |
|---|---:|---:|
| 1 × 16 | 1,104 | 540 |
| 1 × 94 (narrow) | 1,704 | 1,458 |
| 1 × 98 (narrow) | 1,320 | 1,170 |
| 1 × 1,408 | 11,016 | 5,760 |
| 1 × 2,152 | 17,124 | 8,550 |
| 1 × 2,208 | 16,680 | 8,730 |
| 3 × 2,152 | 50,988 | 25,206 |
| 3 × 2,152, four-byte-offset pointers | 50,988 | 37,950 |

The old fused estimate of 15,726 cycles understated the measured 17,124.
The new estimate is 8,550 versus 14,034 for separate add + GeLU, so existing
fusion selects it in the compatible recipe. The model assumes normal allocation
alignment; unusual offset views can still execute the slower checked fallback.
Useful-work accounting now counts the MIX arithmetic rather than the old
separate multiplies/add. GACC is treated as data movement.

Same loaded BS1 recipe, 27 layers, fused QKV, FP8 weights scale -4, B1024,
first repeated layer instrumented, renderer-cropped timing:

| Metric | Compatible ownership, separate add/GeLU | New fused kernel |
|---|---:|---:|
| Full-model cycles | 11,216,910 | 11,070,624 |
| Full-model time | 7.477940 ms | 7.380416 ms |
| GeLU / bias-GeLU tile kernel | 11,388 | 8,556 |
| GeLU / bias-GeLU phase span | 11,928 | 9,096 |
| FP32-reference cosine | 0.994264800 | 0.994221269 |

The fused column includes the bias work, which the separate GeLU column does
not. Whole-model improvement is 1.30%, or 2.35% including the earlier ownership
change compared with 11,337,126 cycles. This experiment loads the compatible
ownership explicitly; it does not claim the default short search chose it.

## Validation and artifacts

`ipu-kernel-equivalence` now supports bias rows and four-byte-offset buffers.
Final tests cover 258 bias cases and 206 unchanged GeLU/reduction controls,
including all finite FP16 encodings with nonzero biases, one/three rows, tails,
in-place output and canaries. A poisoned-accumulator fixture verifies explicit
initialization under strict FP exceptions. Maximum bias-GeLU absolute error
against the mathematical reference was 0.002014; unchanged GeLU/reduction
controls remain bit-identical to the saved source. Both full-model resident
inferences pass cosine >0.99 and produce identical logical outputs.

All eight kernel-cost tests and three fusion tests pass. An intermediate full
codegen suite passed 283 tests with one fusion-selection failure while it used
the slower prototype's cost; the final aligned kernel fixes that selection and
the affected fusion tests pass. The full suite was not rerun after that change.

Artifacts: `artifacts/bias-gelu-20260913/`. `reference/` preserves the original
assembly. `dual-check/`, `dual-rows3/`, `offset-check/` and `dual-control/` contain
final direct measurements. `full/` is the initial MIX-only prototype, **not**
the final build. `final/model.html`, `final/model.profile.capnp`, the package,
reference buffers and exact memory profile are the final full-model result.
