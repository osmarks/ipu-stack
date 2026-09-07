# Native F143 GEMMs

The retained IPU21 path uses `f8v8hihov4amp`: FP8 F143 operands,
FP16 accumulator inputs/outputs, and stored FP16 GEMM results. Output storage
precision and accumulator precision are separate from operand precision.
Reductions, GELU, and attention computation continue in F16.

`ipu_codegen::f143` provides the historical host codec and power-of-two scale
selection. F143 has exponent bias 8, maximum magnitude 240, unsigned zero,
and NaN encoding 0x80. Negative values that round to zero must encode as 0,
not 0x80. Finite overflow saturates in the host encoder.

The activation-aware reconstruction tool `tools/quantize_siglip_f143.py` was
recovered unchanged from commit `66d71b6`. It supports block-diagonal GPTQ,
sequential calibration, bounded LayerNorm equalization, and bias correction.
It writes reconstructed floating-point SafeTensors, not device-ready bytes.
Its default independently scaled 64x64 weight blocks still require a model
import path carrying those scales into the chosen GEMM blocks; restoring the
tool alone does not provide that integration.

## Compiler selection

Use `OperatorCandidate::fp8_gemm(tile_count, scale_exponent)` in the candidate
catalogue and `Precision::F8F143 { scale_exponent }` for encoded parameters.
The benchmark CLI exposes this as `--fp8-scale -4`. Its current policy uses
one explicit scale for both operands of every selected GEMM; the hardware
product scale is their sum, so this option accepts -16 through 15.
For example, -4 represents values up to magnitude 15. It is a bring-up value,
not calibration for arbitrary model activations.

The compiler represents casts and layout changes explicitly. AMP panel widths
change from 16 elements in F16 to 32 in FP8, so a flat cast of packed F16 bytes
is incorrect. The current path redistributes through row-major storage with
the consumer's ownership before casting, then packs the FP8 operand. The
common quantizer uses vector F16 loads and `f16v8tof8`; it writes FP8, while
GEMMs and reductions still write F16. Packing and unpacking remain a meaningful
performance and temporary-memory cost. Attention's QK/PV products stay F16.

The same cast price is used by conversion insertion, compact mid costing,
and expanded kernel costing. Native FP8 GEMM estimates use the retained
instruction structure with 32-element K groups, including coefficient loads
and launch overhead; they do not assume ideal AMP throughput.

## Hardware validation (2026-09-07)

Gaussian final-output checks passed for a 128x128x64 GEMM (maximum absolute
error 0.025391), a small two-block MLP (0.023132), repeated four-head projected
attention (0.000162), full 16-head fused-QKV projected attention (0.000910),
and the full batch-one 729x1152x4304 MLP (0.025391). These check implementation
and approximate numerics, not whole-model accuracy. Separate quantization of
internal activations is not reproduced bit-for-bit by the host reference.

The initial vector-quantizer profiles, before recalibrating the planner's cast
price, measured 341,844 cropped cycles for MLP and 240,780 for fused projected
attention. Profiles are under `artifacts/fp8/{mlp-b1-vector,attention-b1-vector}`.

The original selected separate-QKV F16 package has exactly 46 resident copies
of each weight matrix (349.3 MiB in total). The tested full FP8 fused-QKV plan
also has 46 copies (174.7 MiB). The historical tested three-block F16 MLP has
approximately one copy per weight matrix. The initial single-block FP8 MLP
instead selects 46 copies for the up-projection and approximately one for the
down-projection. These are persistent bindings, not temporary operand staging.
`package-inspect --bindings` now reports allocated bytes so this is measurable.

The FP8 batch-two, three-block MLP remains rejected by the current planner's
separate memory-class arena estimate: standard 252,000 bytes, interleaved
268,288 bytes, and 49,152 bytes of package support before arena rounding.
This is not an observed hardware OOM or proof that no physical placement fits.

Bring-up also fixed two independent issues: Repeat's physical parameter stride
now incorporates final bank-separation requirements, and a receive control at
exactly a send's start is encoded in the preceding interval (a 28-word multicast
loopback encountered in MLP).

With the shared calibrated cast price, the final selected batch-one MLP runs
in **215,166 cropped cycles**, with maximum absolute error **0.025513**. The
fused projected-attention plan remains at **240,780 cycles**, error **0.000910**.
Rendered profiles are `artifacts/fp8/mlp-b1-final/profile.html` and
`artifacts/fp8/attention-b1-final/profile.html`. Batch-two/three-block MLP still
fails the same arena estimate with the calibrated prices. Weight replication
counts remain as reported above.
