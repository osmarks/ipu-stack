# GELU and reduction loop upgrade

The September 6 kernel update (`f817817`) replaces paired F16 GELU arithmetic
with four-lane instructions and pipelines reduction loads with additions.
GELU uses 10 arithmetic instructions per four elements instead of 18; its
two-lane tail remains unchanged. Reduction loads the next partial while adding
the previous one, without reading beyond the input. This does not use `f16v4mix`.

## Hardware measurements

Standalone kernel cycles include worker dispatch and completion:

| Kernel | Elements | Total partials | Before | After |
| --- | ---: | ---: | ---: | ---: |
| GELU | 2208 | — | 13134 | 8718 |
| Reduction | 2208 | 4 | 7734 | 4974 |
| Reduction | 576 | 15 | 6978 | 3090 |

Full MLP timings below use the **profile renderer's default cropped interval**,
not the full profiling counter. Every run passed the numerical check, with
maximum absolute error 0.011719.

| Selection | Before | After |
| --- | ---: | ---: |
| Historical geometry | 188196 | 176706 |
| Down-22 geometry, standard weight input | 186630 | 175140 |
| Automatic selection after cost calibration | 195990 | 178782 |

The fixed historical geometry improves by 6.1%. Automatic selection may change
placement or plans, so its difference is not solely a kernel comparison.
Artifacts and rendered profiles are under
`artifacts/layout-sweep/kernel-upgrade/` (ignored by Git).

## Numerical validation

`ipu-kernel-equivalence` checks 204 cases covering all 63,488 finite F16 input
bit patterns for GELU, partial worker waves, two-lane tails, in-place output,
output canaries, and reductions with 2–28 partials. The numerical GELU reference
is the tanh approximation evaluated in F64 on inputs in [-8, 8]; reduction uses
an F64 sum of F16 inputs sampled in [-0.25, 0.25]. Acceptance tolerance is
`0.01 + 0.002 * abs(reference)`. Maximum observed absolute errors were
0.00201314 and 0.00464249 respectively.

Bitwise agreement with the reference assembly is optional (`--exact`). It also
passed for these changes, but is not a requirement for future numerical
improvements. Exhaustive GELU testing masks floating-point exceptions in the
test wrapper; production exception handling is unchanged. Values outside
[-8, 8] are exercised but are not checked against the F64 numerical reference.

To reproduce, save `device/gelu_f16.S` and `device/reduce_add_f16.S` from
`f817817^` in a reference directory, then run:

```sh
cargo run --release -p ipu-tests --bin ipu-kernel-equivalence -- \
  --sdk "$SDK_PATH" --reference "$REFERENCE_DIRECTORY"
```

The tool also fixes standalone profiled package generation to retain the cycle
sampler symbol, and waits for the runtime completion handshake before collecting
results.

## Costing

The constant-time cost helpers now match these loops. Reduction is
`282 + 6 * ceil(elements / 48) * (11 + 2 * (partials - 1))` cycles. GELU with
length divisible by 16 is `300 + 366 * ceil(elements / 96)`; other even lengths
evaluate six worker workloads, including scalar tails. The tail estimate is
conservative by at most six cycles in the measured cases.

The shortlist regression still requires both historical memory alternatives to
survive operator shortlisting. It no longer requires their exact combination to
survive the complete-program beam: changed kernel throughput can legitimately
favor different grids. Planner beam width is unchanged.
