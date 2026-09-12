# Pretrained SigLIP validation — 2026-09-12

This checks the original SigLIP So400m/14 384 vision tower, including all 27 encoder layers and the MAP head. Earlier full-model cosine measurements used randomized parameters and inputs; they do not establish pretrained-model accuracy.

## Checkpoint and input

- [Google checkpoint](https://huggingface.co/google/siglip-so400m-patch14-384), revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`.
- 428,225,600 vision parameters, exported as 342 named tensors after QKV/KV concatenation. The text tower is not instantiated or executed.
- Six photographs from the [original Big Vision SigLIP demo](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb): authors, siglip, caffeine, robosign, fried_fish and cow_beach2. Original URLs and image/tensor SHA256 hashes are recorded in the fixture manifest.
- [Checkpoint preprocessing](https://huggingface.co/google/siglip-so400m-patch14-384/blob/9fdffc58afc957d1a03a25b10dba0329ab15c2a3/preprocessor_config.json): RGB, bicubic resize to 384×384, divide by 255, normalize with mean/std 0.5. Resize is not aspect-preserving letterboxing or a center crop.
- After preprocessing, the IPU consumes the top-left 378×378 pixels as 729 NHWC patches of 14×14×3. The reference consumes 384×384; its unpadded stride-14 convolution ignores the same last six rows/columns. Resizing directly to 378 would be a different input.

The independent reference is Hugging Face's FP32 `SiglipVisionModel`, with the checkpoint's tanh GELU and layernorm epsilon 1e-6. The output is its 1152-component `pooler_output`, before optional L2 normalization. Cosine is unaffected by L2 normalization.

A separate FP32 implementation using the exported tensor files and our fused QKV/KV graph agrees with that reference on all six images: cosine ≥0.99999994 and maximum absolute error ≤4.34e-5. This checks patch order, weight transposes, fused projection order, heads and MAP mapping independently of device quantization. CUDA TF32 was disabled for this check.

## Reproduction

Export the fixture with `scripts/siglip-pretrained-fixture.py DIRECTORY`. Dependencies used are recorded with the artifacts. This downloads the pinned checkpoint and demo photos, writes little-endian FP32 logical tensors, and computes the independent embeddings. Large downloaded/generated files stay under `artifacts/`, outside git.

```sh
source .env
RAYON_NUM_THREADS=16 RUST_LOG=info target/release/ipu-trivial-test "$IPU_CONFIG" \
  --sdk "$POPLAR_SDK_ENABLED" \
  --runtime-source artifacts/attention-residual-20260911/source/static_runtime.S \
  --workload siglip-vit-benchmark --vit-layers 27 --vit-batch 1 \
  --fuse-qkv --fp8-scale=-4 --optimization-steps 8 --exchange-stream-words 1024 \
  --reference-run --reference-fp32 --reference-inferences 6 \
  --reference-fixture artifacts/pretrained-siglip-20260912/fixture --no-profile \
  --device-lock artifacts/layout-sweep/device.lock \
  --package artifacts/pretrained-siglip-20260912/full27/model.ipuexe
```

`--reference-fixture` uses the compiler's existing logical storage maps to pack checkpoint tensors into the selected physical layouts. It rejects missing or mismatched inputs rather than falling back to random values. It uploads weights once, sends a different image on each call, and compares every embedding component with the external reference. Numerical failure is reported for every completed case, and the command fails if any cosine does not exceed 0.99.

Artifacts: `artifacts/pretrained-siglip-20260912/`, especially `fixture/manifest.json`, `prepare.log`, `mapping.log`, `ranges.log` and `full27/run.log`.

## Dynamic range

The unchanged FP8 scale of −4 represents at most ±15. On `authors.jpg`, the FP32 input to encoder layer 24's MLP up-projection reaches 311.1; 22.58% of its elements exceed ±15. Layer 23 reaches 287.3, with 20.16% outside range. These are learned layernorm affine outputs, not an incorrectly scaled image. Some attention projection inputs and MLP down-projection inputs also exceed range.

An exploratory PyTorch probe that quantizes linear/convolution operands to F143 at scale −4, while retaining FP32 accumulation, produces approximately 0.905 cosine on that image. It is not a complete IPU numerical simulator and omits some internal attention quantization. It establishes that operand clipping alone is a substantial problem; the hardware check remains authoritative.

## Calibrated tensor scales and recovered reconstruction

The unchanged-scale hardware build was stopped before execution when the user redirected work to calibration. **There is no pretrained IPU cosine result from this run.** The following are host numerical experiments with FP32 accumulation and FP32 attention, not IPU throughput or accumulator validation.

`tools/calibrate_siglip_fixture.py` evaluates the exported logical graph, uses the first three photographs for calibration, and reserves robosign, fried_fish and cow_beach2 for evaluation. It chooses separate power-of-two activation and weight scales for every dense operation. A fused QKV/KV weight has one scale for the entire fused tensor. Calibration used no held-out-image statistics.

The historical algorithms were already recovered into `tools/quantize_siglip_f143.py` by `a93328b`, from `66d71b6`. Relevant earlier work includes activation-aware reconstruction (`082e6e3`), real-image calibration (`840bc68`), sequential calibration (`31633d6`), and bounded layernorm equalization (`fd3f207`). The tool now also exposes `--scale-granularity tensor`, and its checkpoint/module naming works with Transformers 5. The new frontend reuses its GPTQ, bias correction and equalization implementations. The 64-channel Hessian blocks limit reconstruction work; they do **not** introduce per-block runtime scales.

The key sensitivity result is the first image projection. Quantizing its input alone gives cosine 0.9693–0.9955. Each individual encoder operation family (QKV, attention output, MLP up, MLP down) quantized in isolation gives at least 0.9986. Quantizing all weights alone gives 0.9978–0.9986. Consequently the selected numerical experiment retains FP16 input projection, with FP8 operands for every encoder and MAP dense operation. Its embedding operands/output are rounded to FP16; multiplication still uses FP32 accumulation in the probe.

| Image | Split | All FP8, nearest | FP16 embedding, nearest | FP16 embedding, GPTQ + bias correction |
| --- | --- | ---: | ---: | ---: |
| authors | calibration | 0.986800 | 0.996425 | 0.995571 |
| siglip | calibration | 0.984483 | 0.997554 | 0.998426 |
| caffeine | calibration | 0.991502 | 0.997728 | 0.998133 |
| robosign | held out | 0.980060 | 0.994956 | 0.996656 |
| fried_fish | held out | 0.984517 | 0.997094 | 0.996908 |
| cow_beach2 | held out | 0.966660 | 0.997383 | 0.997833 |

GPTQ is not uniformly better on every image, but improves the minimum held-out cosine from 0.994956 to 0.996656. Bounded layernorm equalization preserved FP32 outputs but did not help the all-FP8 experiment: its GPTQ minimum was 0.959466, versus 0.962702 without equalization. It is not selected. A few activation elements still exceed the calibration extrema on held-out images (up to 103 with the selected GPTQ variant); the reported probe clips them. This is a six-image numerical check, not broad validation of a deployment calibration set.

```sh
artifacts/pretrained-siglip-20260912/venv/bin/python tools/calibrate_siglip_fixture.py \
  artifacts/pretrained-siglip-20260912/fixture --fp16-embedding \
  --output-weights artifacts/pretrained-siglip-20260912/calibrated-parameters.safetensors \
  --report artifacts/pretrained-siglip-20260912/fp16-embedding.json
```

The saved SafeTensors contain reconstructed FP32 logical parameters named `vit.*`, in the graph's input×output matrix order. They require the scales in the accompanying report when encoded as FP8. They are not a Hugging Face checkpoint or a device-ready packed blob. The original reference embeddings remain those of the unchanged pretrained model.

### Simpler fixed scales also pass

A subsequent test shares scales across all 27 instances of each repeated GEMM
position. Sharing one scale between the weight and activation as well also
passes. These use nearest weight rounding, without GPTQ or equalization:

| Image | Fixed per-position, separate operands | Fixed per-position, shared operands |
| --- | ---: | ---: |
| authors | 0.996512 | 0.995723 |
| siglip | 0.997774 | 0.997570 |
| caffeine | 0.997171 | 0.997705 |
| robosign | 0.995742 | 0.996026 |
| fried_fish | 0.997783 | 0.997192 |
| cow_beach2 | 0.996791 | 0.996705 |

No activation clipping was observed in either experiment. The shared-operand
scheme is the simpler candidate to integrate. Its encoder scales are:

| Repeated GEMM position | Common activation/weight scale |
| --- | ---: |
| QKV | −2 |
| Attention output | −4 |
| MLP up | +1 |
| MLP down | −1 |

MAP uses fixed scales −12 (query), −1 (KV), −5 (attention output and MLP up),
and −6 (MLP down). Input projection remains FP16. Each repeated position uses
the same scale in every layer. All product scales are twice the operand scale,
matching the existing kernel ABI.

```sh
artifacts/pretrained-siglip-20260912/venv/bin/python tools/calibrate_siglip_fixture.py \
  artifacts/pretrained-siglip-20260912/fixture --fp16-embedding \
  --scale-sharing repeat-role --shared-operand-scale --nearest-only \
  --output-weights artifacts/pretrained-siglip-20260912/shared-scale-parameters.safetensors \
  --report artifacts/pretrained-siglip-20260912/role-shared.json
```

**Neither independent operand scales nor per-layer Repeat arguments have been
shown necessary for this model.** Those changes would be needed to realize the
fully independent experiment exactly, but the simpler scheme has sufficient
margin in these host measurements. The remaining compiler integration is a
fixed precision/scale choice per GEMM position and an FP16 input projection.
It does not require changing Repeat's runtime arguments or the GEMM scale ABI.
Device accumulation accuracy and placement still need validation before making
an IPU accuracy/performance claim.

## Hardware port (2026-09-12)

Implemented in `4aefbd2`. `PipelineConfig::gemm_precisions` fixes the operand
precision of an individual graph GEMM, including operations inside Repeat.
The benchmark's `--reference-calibration REPORT` loads the shared scale policy
and binds weight precision accordingly. It rejects mismatched scales within a
repeated parameter sequence. The runtime ABI is unchanged.

FP8 GEMM specialization keys now ignore scale exponents, which remain call
arguments. Previously different scales could generate duplicate definitions of
the same specialization symbol. Row variants are now grouped together before
code generation, and calls still receive their own product scale.

The canonical, unoptimized full 27-layer plan **passed on IPU hardware**:

| Image | IPU cosine against independent FP32 reference |
| --- | ---: |
| authors | 0.994614293 |
| siglip | 0.997865842 |
| caffeine | 0.997627423 |
| robosign | 0.996131927 |
| fried_fish | 0.997170216 |
| cow_beach2 | 0.996680336 |

All weights were uploaded once and remained resident across the six distinct
image calls. Each invocation uploaded its preprocessed image and downloaded its
MAP embedding. The maximum absolute error across all images was 0.283580.
The reference was the original pretrained FP32 Hugging Face model, not an
already-quantized reference.

Baseline package, complete log and raw outputs:
`artifacts/pretrained-siglip-20260912/calibrated27-baseline/`.
Tests: 238 enabled codegen tests passed (four ignored), and all 12 runner tests
passed. Regression coverage includes independent fixed precision choices within
Repeat and shared kernel code with distinct FP8 scale arguments.

The hardware build command uses the same configuration as above, except replace
`--fp8-scale=-4` with:

```sh
--reference-calibration artifacts/pretrained-siglip-20260912/role-shared.json
```

Use `--optimization-steps 0` for this baseline. The normal optimized build uses
`--optimization-steps 8` and writes to `calibrated27/`.
