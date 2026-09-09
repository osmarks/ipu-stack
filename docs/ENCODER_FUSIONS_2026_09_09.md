# Encoder cast/pack and elementwise fusion experiments

The benchmark is the batch-one, one-layer So400m/14 ViT with fused QKV, 378×378
input and native FP8 GEMMs (scale −4). Timings use the profile renderer's startup
cutoff, at 1.5 GHz. Each distinct package is measured once on hardware.

## Cast directly into FP8 panels

The local F16 row-major → FP8 AMP-left kernel now handles four-half-aligned tails
and padded panels. Complete panels retain the existing vector/pipelined loop.
Boundary panels use vector loads; wholly padded panels use zero stores without
reading F16 padding. Late conversion compares row-major distribution followed by
this kernel against the existing packed-F16 distribution/cast path. Early casting
can also use producer shards which previously failed the 32-column restriction.

| Build | Cropped cycles | Encoder body span (ops 3–23) |
|---|---:|---:|
| Previous elementwise upgrade | 715,350 | 521,412 |
| Direct cast/pack and tail support | 668,484 | 474,732 |

The full benchmark improves 6.55%; the encoder span improves 8.95%. The span is
an operation-boundary approximation, including intervening exchanges, rather
than an additive sum of kernel times. The large down-projection preparation is
much shorter. Some work moves into the preceding GeLU's preparation, so individual
operation totals should not be interpreted as pure kernel speedups.

Hardware/reference validation passed (maximum absolute output error 0.093262).
The kernel checker passed 738 bitwise cases across bank placements, scales,
row-major and packed inputs, tails and wholly padded output panels. For 38×80,
the first scalar-guarded tail took 10,374 cycles; the retained vector tail takes
3,528 cycles.

Profile: [direct cast/pack](../artifacts/cast-pack-tails/vit/profile.html).

## Direct FP8 producer output

`sort4x16lo/hi` support both main-register/main-slot and ARF/auxiliary-slot forms
(ISA §3.7.1.20–21). More useful here is `f16v2tof8` (§3.7.3.1.4): two converted
pairs can be combined with `sort4x16lo` and written as one word. GeLU reuses the
existing arithmetic macros. Layernorm keeps centered FP32 statistics and FP32
affine arithmetic, then converts pairs before storing.

| Local shape | F16 output only | Direct packed FP8 output |
|---|---:|---:|
| LN, 1×1152 | 9,378 | 9,936 |
| LN, 1×1728 | 13,338 | 14,040 |
| GeLU, 1×1152 | model 6,450 | 7,092 |
| GeLU, 1×1728 | model 9,510 | 10,386 |

The separate cast/pack is additional to the F16 column. A first eight-value
prototype incurred spills and repeated setup: 1×1152 LN was 18,330 cycles and
GeLU 13,470. It was replaced, not integrated. Packed multi-row LN still has
expensive address calculation and is deliberately priced accordingly.

The mid rewrite recognizes both format conversions and operator-created cast
primitives, consumes an explicit cast result, requires compatible ownership,
and preserves F16 intermediates with other readers. It compares both costs.
Hardware testing passed 550 cases, including reference numerics and write guards.
The first full ViT with these choices retained its separate producer/cast paths
and remained at 668,484 cycles; this experiment alone is not a model-level win.

## Residual addition and statistics

`AddLayerNormMoments` explicitly produces FP32 moments and an F16 residual sum.
Its sum pass stores the rounded addition while accumulating in FP32; the centered
variance pass reads the stored result. Mid and low expose both writes to placement
and lifetime analysis. The ABI passes the additional output pointer after inputs.

A costed mid rewrite handles both existing distributed moments and compatible
ordinary LN (moments followed by apply with one statistics group). It can preserve
a residual used by later operations or a Repeat result. It does not force a change
of ownership or conceal a write to an input buffer.

At local width 1152, fused add/statistics takes 4,050 cycles versus 1,644 + 3,366
for separate add and statistics. The direct checker passed 518 cases, checking
the residual and moments independently, including constant/large-mean inputs.

The final combined kernel check includes 690 cases and the MLP's 2,152-column
tail. FP8 GeLU for 1×2152 takes 13,680 cycles in packed order; fused residual
statistics takes 6,444 cycles. All numerical checks and write guards passed.

Fusion eligibility compares resolved shard geometry, rather than requiring
identical layout descriptions: explicit column grains and omitted whole-axis
specifications can describe the same storage. Identity copies may be bypassed
only with the appropriate reader and intervening-write checks. Compute allocation
aliases are explicit `(result index, input index)` pairs, so statistics and
residuals retain distinct alias rules inside Repeat as well as outside it.

Validation: 216 Rust tests plus the graph doctest; Clippy with the repository's
existing argument-count/type-complexity allowances. The regression coverage checks
fusion through identity copies, live F16 readers, separate residual/statistics
allocations, and carried outputs in Repeat.

## Final integrated ViT result

The final integrated package passes hardware/reference validation with maximum
absolute error 0.093262. Its cropped runtime is 668,484 cycles (0.445656 ms),
and the encoder body span remains 474,732 cycles (0.316488 ms). The selected
plan contains no direct FP8 GeLU/LN or fused residual/statistics kernels. Thus
the measured model improvement comes from cast/pack planning and tail support;
the other two paths have local hardware validation but no additional gain in
this workload. They require compatible producer/consumer ownership, which the
selected redistribution paths do not provide.

Final profile: [integrated ViT](../artifacts/encoder-fusions/vit-final/profile.html).
Package and run log are in the same directory. The remaining encoder opportunity
is choosing compatible ownership across these boundaries, rather than assuming
that an available fused kernel will automatically be selected.

## Larger batches

The same full-size benchmark was attempted at batches 2, 4 and 8 using the
normal planner limits. Batch 2 passes hardware/reference validation (maximum
absolute error 0.082764), at 996,756 cropped cycles and a 744,216-cycle encoder
span. Per image, the encoder span is 372,108 cycles, 21.6% below batch one.
Profile: [batch 2](../artifacts/encoder-fusions/vit-b2/profile.html).

Batch 4 exhausts five admitted finalists after about 17 minutes 40 seconds.
Two fail during host-support assembly (`insufficient tile SRAM for 12 host-data
bytes`); one fails a 122,880-byte standard allocation, and two fail 88,320-byte
interleaved allocations. Batch 8 fails at mid operation 18 (MLP-up); the final
reported rejected peak is 830,080 bytes including 247,152 estimated exchange
table bytes. Other planner configurations report 828,108 bytes. Neither is a
hardware batch-size ceiling: these are failures of the retained plans and
placement. Logs are under `vit-b4/` and `vit-b8/` beside the batch-2 artifacts.

These measurements precede the independent paired-receiver address correction
described in [the exchange reference](EXCHANGE_INSTRUCTION_REFERENCE.md#paired-sender-lane-reservation).

With that correction, full hardware/reference checks pass again:

| Batch | Cropped cycles | Encoder span | Encoder cycles/image |
|---|---:|---:|---:|
| 1 | 659,370 | 469,782 | 469,782 |
| 2 | 966,162 | 721,272 | 360,636 |

The batch-2 encoder uses 23.2% fewer cycles per image. Output errors remain
0.093262 and 0.082764 respectively. Rendered profiles:
[batch 1](../artifacts/encoder-fusions/vit-paired-b1/profile.html),
[batch 2](../artifacts/encoder-fusions/vit-paired-b2/profile.html).

Enabling paired sender-pair loopback subsequently passes both full model checks:
batch 1 takes 653,988 cycles (encoder 465,156), batch 2 takes 959,646 (encoder
715,698). The corresponding profiles are
[batch 1](../artifacts/encoder-fusions/vit-loopback-b1/profile.html) and
[batch 2](../artifacts/encoder-fusions/vit-loopback-b2/profile.html).
