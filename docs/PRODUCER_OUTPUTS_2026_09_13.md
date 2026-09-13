# Producer output capabilities and FP8 epilogues

`kernel/output.rs` describes the implemented FP8 output orders, operand count,
row-parameter placement, column granularity and complete-row requirements.
`mid/output_fusion.rs` matches/safely rewrites producers and uses ordinary
broadcast placement for their parameters. ABI checks share the capability.
BiasGeLU now has a `bias_gelu_f8` implementation in the existing GeLU FP8 source,
using the MIX polynomial and retaining FP16 arithmetic before final conversion.
It accepts linear or AMP-left FP8 output, including output padding.

The common pass prices execution at the consumer's owners and at the original
producer's owners, including the changed FP8 redistribution. Add/GeLU fusion
runs before output fusion so the two epilogues can compose. Existing other-user,
alias and intervening-write checks apply to both locations.

810 direct elementwise cases pass with the bias variant enabled. For a linear
1x1152 bias-GeLU FP8 output the kernel takes 7152 cycles; 1x2152 takes 13776.
One/three-row, packed output and padding cases are included. Three fusion tests
pass, including relocation of the bias and validation of generated kernel calls.

The consumer-only full 27-layer BS1 run retains FP16 BiasGeLU and is unchanged
at 11,001,942 renderer-cropped cycles. Moving the arithmetic onto the down-GEMM's
owners is expensive; merely providing the epilogue does not make it profitable.
The producer-owned alternative run changes other eligible producers (including
layernorm), still retains FP16 BiasGeLU, and takes 11,016,888 cycles: 0.14% slower.
Both resident inferences pass against FP32, cosine 0.993916310 for that variant.
This is a measured limitation of the estimated choice, not a demonstrated speedup.

Artifacts: `artifacts/producer-fp8-20260913/`. `bias-check/` has direct diagnostics;
`bias-full/` is the consumer-only control; `source-full/` contains the build,
profile, exact memory placement and resident-reference validation for both
execution locations. The saved layout is the earlier compatible BS1 recipe,
with zero extra layout-search steps.
