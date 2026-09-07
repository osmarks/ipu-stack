# Single-layer SigLIP vision benchmark

`--workload siglip-vit-benchmark` builds the image encoder from
`../big_vision/big_vision/models/vit.py`, using So400m/14 dimensions with encoder
depth reduced from 27 to one. Input is 378×378 RGB, the usable region of the
384-pixel model's VALID patch convolution. This gives 729 tokens of width 1152,
16 attention heads of width 72, and MLP hidden width 4304.

The graph includes patch projection and bias, learned positional embedding,
pre-normalized self-attention with Q/K/V and output projections, the attention
residual, pre-normalized MLP with biases and its residual, and final encoder
layernorm. MAP pooling uses a shared learned probe, its own Q/K/V and output
projections, and a layernorm/MLP residual. The returned tensor is `[batch, 1,
1152]`, retaining the singleton probe axis. Dropout is omitted. There is no
optional classifier, representation projection, text tower, or contrastive
embedding normalization.

```mermaid
flowchart TD
  pixels[378×378 RGB pixels in patch order] --> stem[588→1152 embedding GEMM + bias]
  stem --> pos[Add learned positional embedding]
  pos --> ln1[LayerNorm]
  ln1 --> sa[Q/K/V projections → self-attention → output projection]
  pos --> add1[Residual add]
  sa --> add1
  add1 --> ln2[LayerNorm → 1152→4304→1152 MLP]
  add1 --> add2[Residual add]
  ln2 --> add2
  add2 --> final[Encoder LayerNorm]
  probe[Learned probe] --> map[Cross-attention + output projection]
  final --> map
  map --> mln[LayerNorm → MLP]
  map --> out[Residual add → pooled output]
  mln --> out
```

The image binding contains all pixels in `[batch, patch_y, patch_x, y, x,
channel]` order, flattened to `[batch, 729, 588]`. Host patch packing is outside
the timed device program; the convolution itself is the timed embedding GEMM.
Weights use deterministic Gaussian Xavier-style initialization, positional
embeddings use standard deviation `1/sqrt(width)`, layernorm scales are one,
and biases are tiny Gaussian values. Each batch uses the same learned probe.
These are performance/implementation tests, not pretrained accuracy evaluations.

The default tensor precision is F16, including attention results, residuals,
bias additions and GELU. F32 is retained for layernorm statistics and internal
attention softmax/accumulation state; attention explicitly converts its result
back to F16. Selecting FP8 GEMMs does not change this policy.

Layernorm is a reusable graph operation with epsilon 1e-6 and affine parameters.
Its F16 codelet shares each row across all six workers, using separate local
sum, centered-variance and affine passes. Statistics stay F32, with 96 bytes of
shared scratch and local synchronization between passes. The kernel requires
even-width unpadded rows. Floating-point
addition supports equal shapes and contiguous suffix broadcasting, covering
residuals, bias vectors and positional embeddings. Add offers column sharding
and preserves compatible input layouts, including packed equal-shape residuals,
rather than always gathering complete rows. Neither operation silently
interprets unsupported storage as a compatible layout.

Example (native FP8 weights, F16 normalization and residuals):

```sh
RAYON_NUM_THREADS=24 target/release/ipu-trivial-test c600-init.ipucfg \
  --workload siglip-vit-benchmark --fp8-scale=-4 \
  --diagnostic-atol 0.2 --diagnostic-rtol 0.05 \
  --device-lock artifacts/layout-sweep/device.lock \
  --package artifacts/vit/so400m-fp8/model.ipuexe \
  --profile-output artifacts/vit/so400m-fp8/profile.ipuprof
```

Omit `--fp8-scale` and the tolerance flags for F16 GEMMs. Unlike the older isolated benchmarks, this
benchmark retains non-weight inputs in F16 when selecting FP8 GEMMs; the planner
inserts activation casts where required. All final outputs are checked against
the host reference automatically. `--vit-small --tiles 64` uses a 28×28 image,
width 144, two 72-wide heads and MLP width 288 while preserving the whole graph.
`--vit-batch` sets the batch size (default one).

The FP8 example explicitly loosens the comparison tolerance. Small upstream
rounding differences can cross activation-quantization thresholds and accumulate
through the 13 GEMMs; the host reference quantizes operands at each selected
GEMM but does not reproduce the hardware's intermediate rounding exactly.
The small complete graph has maximum absolute error 0.00293 with F16 GEMMs and
0.12988 with FP8 GEMMs. This comparison does not establish pretrained accuracy.

The initial full-size F16 search was rejected by memory accounting: its
shortlisted plans reach 540,544 tensor bytes plus 49,152 support bytes per tile
at the encoder down-projection, or exhaust SRAM later at MAP preparation.
This is a planner/layout limitation, not a requirement for FP32 arithmetic.
The reduced F16 model passes on hardware.

Initial full-size hardware validation (2026-09-07): native FP8 GEMMs, F16 exposed
activations, 2,656,824 cropped profile cycles (1.771216 ms at 1.5 GHz).
Maximum absolute reference error is 0.102051, using the explicit tolerance in
the command above. The rendered profile is
`artifacts/vit/so400m-fp8/profile.html`; raw samples and the run log are adjacent.
The measurement includes embedding, one complete encoder block, final norm,
and MAP pooling; host patch packing and parameter loading are outside it.

Validation exposed and fixed three memory hazards: short AMP-left packing
workers wrote beyond their row allocation; tensors could share instruction
memory elements with linked kernels; final attention padding could contain
FP32 values that faulted during the F16 cast. Final merge now clears just that
padding. Mixed-class multicast loopbacks and late exchange-row setup calls
also now receive the required placement and code-size reservations.

## Performance follow-up

The initial profile exposed scalar Add indexing, scalar single-worker layernorm,
and four separate instructions per 32-bit AMP-left packing word. Add now has
native half2 dense paths and column/layout-preserving planning choices. LN shares
a row across six workers and retains centered F32 statistics. Aligned AMP-left
packing uses 64-bit loads/stores with built-in address updates. LN now has an
arithmetic-work estimate in profile metadata instead of reporting N/A.

Broadcast operands are partitioned along the output's matching dimensions,
replicating only dimensions being broadcast. Equal-shaped packed operands retain
their physical padding. Batch validation also exposed missing matrix iteration
in row-major packing. Packing now follows the storage representation: AMP-left
panels flatten batch and row, while coefficient formats retain separate matrices.
Both F16 and FP8 two-image small models pass all 40 operator checkpoints; final
maximum absolute errors are 0.004395 and 0.130188, respectively.

The larger plans exposed excessive lazy heap refreshes in exchange scheduling.
Initially ready transfers with identical endpoint roles share one global heap
entry; later dependency releases remain individual entries. Multicast pressure
is now refreshed even when readiness has not changed, avoiding history-dependent
stale-pressure choices. Stale keys update in place, with a linear refresh/rebuild
when individual heap repairs would cost more. Randomized comparison with eager
priority selection covers unicast, multicast and memory hazards.

With the same 64-tile planning budget, the small FP8 model decreased from
255,804 to 137,268 cropped cycles. Its rendered profile is
`artifacts/vit/optimized-small-fp8/profile.html`. This is a kernel/planning
regression check, not a substitute for the full-size measurement: the full
model selects substantially different shard sizes and exchange patterns.
The release tests pass (158 codegen tests, six benchmark/CLI tests and the
graph doctest), as does Clippy with the repository's existing complexity
allowances.

A separate four-token diagnostic retains the full 1,152/4,304 channel widths
and 16 heads. It passes FP8 hardware/reference validation (maximum absolute
error 0.145020). Its 1,152-wide LN takes 15,558 cycles per row, versus 138,678
in the original full-size profile. Its final Add uses 288 tiles and spans 456
compute cycles (378 per tile), versus the original single tile's 26,850 cycles.
The complete diagnostic is 224,718 cropped cycles; this is **not** the
729-token model's runtime. Its profile and the diagnostic binary's exact shape
override are under `artifacts/vit/full-width-four-token-diagnostic/`.

Remaining structural limitations include standalone projection bias additions
(which require row-major traversal), separate Q/K/V GEMMs in this benchmark, and
large coefficient-layout packing operations with sparse ownership. The single-row
MAP normalization still resides on one tile, using all six workers; cross-tile
statistics would require a different normalization implementation.
