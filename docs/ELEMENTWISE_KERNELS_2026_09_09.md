# Add and layernorm kernel upgrade

Commits `67251e5` and `0729e12` improve the existing device kernels and update
their shared cost models. Ordinary planning remains in use.

## Implementation

- Dense and complete suffix-broadcast F16 adds use four-wide assembly, with
  post-increment loads/stores and six-worker distribution. Two-wide/general
  broadcasting and halfword tails remain supported, including four-byte-aligned
  addresses. The wide loop requires eight-byte alignment.
- Layernorm keeps FP32 mean, centered variance, normalization and affine
  arithmetic, with F16 inputs/outputs. The sum loop uses `f16v8acc`; the centered
  variance loop converts to FP32 and uses `f32v4sqacc`. Statistics do not use
  the cancellation-prone difference of raw second moments and squared mean.
- `f32v2gina` reads the FP32 accumulators. Each pass clears accumulator state
  first. Short or unaligned rows use vector-pair loops. All paths avoid
  speculative tail reads and support in-place F16 output.
- Normalization/application uses explicit vector loops and `f32oorx`. Ordinary,
  fused add–layernorm, and distributed moments/apply share these routines in
  `device/elementwise_vector.hpp`. The three local worker launches per row are
  retained; final scalar statistics still repeat across workers.
- The host compiler supplies the original source directory to popc for quoted
  C++ includes. The existing source-cache hashing already follows these headers.
- Primitive costing models row setup, worker rounding, wide statistics, affine
  application, and broadcast-row overhead instead of the previous fixed
  per-element prices. Eight-byte allocation alignment is assumed for wide
  pricing; exceptional shifted addresses may still take a slower runtime path.
  Profile useful-work estimates now reflect ACC/SQACC arithmetic issue counts.

## Hardware results

Direct full-invocation timings, including supervisor/worker setup:

| Local operation | Previous | New |
| --- | ---: | ---: |
| Dense add, 576 elements | 2,940 | 1,068 |
| Dense add, 1,728 elements | 8,124 | 2,220 |
| Layernorm, one 1,152-wide row | 15,540 | 9,378 |
| Add + layernorm, one 1,152-wide row | 18,468 | 11,598 |
| Moments, one 1,152-wide shard | 5,580 | 3,366 |
| Apply, one 1,152-wide shard, one stats part | 10,998 | 6,696 |

Small/unaligned cases can have slightly higher setup overhead; the wide-row
improvements are not universal speedups for every shape.

Full-size batch-1 one-layer SigLIP ViT, including MAP, fused QKV/KV and FP8
GEMMs at scale -4, measured with the renderer's startup cutoff:

| Version | Hardware cycles | Time at 1.5 GHz | Maximum reference error |
| --- | ---: | ---: | ---: |
| Before this upgrade | 779,868 | 0.519912 ms | 0.077393 |
| New kernels, previous costs | 737,892 | 0.491928 ms | 0.073853 |
| New kernels and costs | 715,350 | 0.476900 ms | 0.071045 |

The first comparison isolates kernel changes from cost-model changes: it retains
the original call counts. Ordinary layernorm's combined timeline span falls from
50,100 to 31,614 cycles; adds fall from 40,734 to 19,860 cycles. These spans can
overlap other work and should not be summed as independent critical-path costs.

The final result is 8.3% faster overall. With new costing, the encoder MLP uses
separate add and GeLU: its local add takes 2,652 cycles and GeLU 12,474, versus
18,198 for the old fused bias–GeLU call. Existing fusion stays available when it
is cheaper. Encoder layernorm still uses 729 owners; this change does not add
multi-output residual/statistics fusion or search layouts specifically to enable
it. Exchange and layout preparation remain substantial costs.

Selection/package validation took 310.6 s with old costs and 348.3 s with new
costs. These are individual build timings, not a compiler-performance claim.

Profiles and logs:

- Before: `artifacts/vit/qkv-fused-b1/`
- Kernels only: `artifacts/elementwise-upgrade/vit-kernels/`
- Final: `artifacts/elementwise-upgrade/vit-costed/`
- Direct checks: `artifacts/elementwise-upgrade/check/` and `baseline/`.

## Validation

The direct `elementwise_check` hardware diagnostic checks 478 cases against a
host reference, including output guards, multi-row execution, worker tails,
odd add lengths, shifted FP16 addresses, in-place output, zero variance and
large means relative to variance. All pass. The original kernels were measured
on the initial 312-case matrix; no deterministic benchmark was repeated without
a code or test change. The full ViT variants each pass their hardware/reference
check. These are randomized implementation checks, not pretrained accuracy tests.

213 Rust tests and one doctest pass (five ignored). Release Clippy passes for
codegen, benchmarks and ELF tooling. The cycle-model regression compares against
independent device timings. Fusion tests retain live residual outputs and verify
the profitable short-row case; they no longer require fusion for wide rows where
standalone add can be cheaper.
