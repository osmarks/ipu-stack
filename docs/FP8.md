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
