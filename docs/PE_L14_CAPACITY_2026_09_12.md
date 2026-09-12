# PE-Core L/14 capacity check — September 12, 2026

Source: `../perception_models`, commit
`3e352cca660658d4b5c90f42a7808b11469e4c66`, specifically
`core/vision_encoder/config.py`, `pe.py` and `rope.py`.
The target is the image-embedding model **PE-Core-L14-336**, not the
448-pixel PE-Lang or PE-Spatial variants.

## Architecture and probe scope

| Property | PE-Core-L14-336 |
|---|---|
| Encoder | 24 layers, width 1024, MLP width 4096 |
| Encoder attention | 16 heads, 64 channels per head |
| Input | 336 x 336, 14 x 14 patches |
| Sequence | 576 image tokens plus one class token = 577 |
| Pool | Learned-query attention, eight heads of width 128, MLP width 4096 |
| Additional bookends | Bias-free patch projection, pre/post encoder layernorm, final 1024 x 1024 projection |
| Position encoding | Learned absolute positions plus 2D RoPE on Q/K |

`--vit-model pe-core-l14-capacity` reuses the ViT graph builder and existing
operators for a capacity probe. The encoder uses one structured Repeat with
24 distinct resident parameter sets. The class slot is a zero input patch;
its learned embedding is folded into the first absolute-position vector.
This is algebraically valid because the patch projection has no bias.
The pool probe is shared numerically across batch entries, as in the existing
SigLIP fixture. For batch one the graph has 317,150,208 parameter elements;
the separate class vector has been folded away.

**This is not a complete PE implementation.** It omits 2D RoPE and retains the
existing layernorm epsilon of 1e-6 instead of PE's 1e-5. Inputs and weights are
randomized. Numerical comparisons therefore validate the probe, not the PE
checkpoint or its task accuracy. RoPE needs an actual implementation before a
working port can be claimed; it also changes layout/materialization choices,
so these results are not a proof of what the fully implemented model can fit.

## Sweep

All cases: 24 layers, FP8 GEMM weights/operands at scale -4 with FP16
accumulation, FP16 attention, fused QKV, B1024 exchange streams, no profiling,
three resident inference calls, and FP32-reference checking when build succeeds.
No weights are streamed between inference calls.

The sweep uses zero local optimization steps to test the canonical plan.
For rejected cases, increasing that budget cannot help: the current optimizer
requires the initial plan to pass full packaging before it searches.
The automatic attention mode uses a single key block for batch one and
64-key blocks for the encoder at larger batches. Explicit materialized
attention was also tested at batches 2, 4 and 8.

| Case | Result | Planning time |
|---|---|---:|
| b1 | PASS | 46.455 s |
| b2 | placement failed: tile 0 has insufficient Ipu21Standard SRAM for 16384 bytes | 46.529 s |
| b3 | placement failed: tile 736 has insufficient Ipu21Standard SRAM for 61440 bytes | 55.912 s |
| b4 | placement failed: tile 0 has insufficient Ipu21Standard SRAM for 3072 bytes | 68.685 s |
| b5 | placement failed: tile 736 has insufficient Ipu21Standard SRAM for 61440 bytes | 95.894 s |
| b6 | exchange transfer count exceeds per-tile limit: 16738 fragments, limit 16384 | 20.533 s |
| b7 | exchange transfer count exceeds per-tile limit: 16915 fragments, limit 16384 | 21.171 s |
| b8 | exchange transfer count exceeds per-tile limit: 18213 fragments, limit 16384 | 21.159 s |
| b2-materialized | placement failed: tile 92 has insufficient Ipu21Standard SRAM for 3072 bytes | 43.864 s |
| b4-materialized | placement failed: tile 184 has insufficient Ipu21Standard SRAM for 3072 bytes | 65.298 s |
| b8-materialized | exchange transfer count exceeds per-tile limit: 19199 fragments, limit 16384 | 23.002 s |

These are concurrent-build wall times, not compiler performance comparisons.
Allocation sizes in the failures are the individual failed requests, not
additional memory required to make the entire graph fit.

Batch one passes all three resident calls at FP32-reference cosine
**0.993937987**, maximum absolute error 0.384227. Its exact parameter
allocations total **318,128,128 bytes (303.4 MiB)**. Taking the union of every
allocation/reservation over execution, never-used address space per tile is
139,524 bytes minimum, 198,620 median, and 491,548 maximum. This includes all
resident weights and support reservations. White space in stacked reuse rows
is not additional memory capacity.

Batch two's lifetime-ordered allocation first fails on a 56,456-byte FP8 MLP
activation, whose largest remaining hole is 49,840 bytes. The size-ordered
retry fails on a 16,384-byte FP16 activation, with a largest hole of 13,312
bytes. These are placement failures despite the smaller parameter budget;
they do not establish hardware impossibility. The materialized alternatives
also fail, so this is not solely the automatic attention choice.

**Outcome:** the current plans do not demonstrate a higher-batch fit. The
smaller model substantially reduces parameter storage, but switching dimensions
alone does not remove the existing baseline-placement problem.

## Artifacts and reproduction

`artifacts/pe-capacity-20260912/summary.json` records the full sweep.
Each case directory has its run log and initial memory estimates. `b1/` also
contains the executable and exact placement map, linked as `b1/memory.html`.
No runtime timing for the probe is presented as PE inference performance.

```sh
source .env
RAYON_NUM_THREADS=16 RUST_LOG=info,ipu_codegen::place=debug \
  target/release/ipu-trivial-test "$IPU_CONFIG" --sdk "$POPLAR_SDK_ENABLED" \
  --runtime-source artifacts/attention-residual-20260911/source/static_runtime.S \
  --workload siglip-vit-benchmark --vit-model pe-core-l14-capacity \
  --vit-layers 24 --vit-batch 2 --fuse-qkv --fp8-scale=-4 \
  --optimization-steps 0 --exchange-stream-words 1024 --no-profile \
  --reference-run --reference-fp32 --reference-inferences 3 \
  --memory-profile-directory artifacts/pe-capacity-20260912/b2/memory \
  --device-lock artifacts/layout-sweep/device.lock \
  --package artifacts/pe-capacity-20260912/b2/model.ipuexe
```

Add `--attention-strategy materialized` for the explicit alternative.
Graph tests cover the PE token count, pool head geometry, extra bookends,
parameter count and zero class patch, plus both existing SigLIP graph tests.
All pass, as does the workspace all-targets check.
