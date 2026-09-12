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
Its default independently scaled 64x64 weight blocks are not integrated.
The runtime uses tensor-wide scales; per-block scale integration is not planned
for this path.

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
is incorrect. F16-to-FP8 conversion now redistributes into the consumer's
**packed F16 order**, then quantizes while regrouping pairs of panels. Automatic
host inputs are populated in that order directly. This removes the row-major
unpack/cast/repack intermediate and its byte-sized permutation kernels.

The regrouping is shared by AMP left, transposed left/right, and block-major
operand formats. A row comprises two contiguous 16-half source spans; four
`f16v8tof8` instructions produce one contiguous 32-byte FP8 row. An eight-bundle
pipeline combines vector conversion with simultaneous loads/stores when memory
elements are separate; the general fallback uses twelve bundles. Workers share large panels by rows and handle whole small
panels when at least six are available. Flat F16-to-FP8 casts use a two-bundle
pipeline or four-bundle fallback per eight values, with a masked tail. Other FP8 cast directions
retain the row-major fallback. GEMMs, reductions, GELU, and attention's QK/PV
products still write/use F16.

The same panel-aware cast price is used by conversion insertion, compact mid costing,
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

## Packed conversion performance update

After removing the row-major FP8 preparation path and replacing the quantizer
loops, the final batch-one hardware results are:

| Workload | Previous FP8 | Packed/vector FP8 | Reduction |
| --- | ---: | ---: | ---: |
| MLP, 729x1152x4304 | 215,166 | 151,998 | 29.4% |
| Fused projected attention, 16 heads | 240,780 | 176,994 | 26.5% |

These are renderer-cropped cycles. The retained F16 comparison profiles measure
175,578 for MLP (`artifacts/useful-work/mlp/execution.ipuprofile`) and 195,096
for fused projected attention (`artifacts/qkv-fusion/fused/execution.ipuprofile`).
The new FP8 runs are respectively 13.4% and 9.3% faster than those profiles;
these are end-to-end workloads, including operations that remain F16.

The attention quantizer's maximum duration falls from 14,124 to 4,914 cycles,
including its panel regrouping. The large FP8 `static_copy_u32` packing stage
is removed. F16 Q/K unpacking and other attention preparation remain. The MLP
planner now selects approximately **one resident copy of each weight matrix**:
4,958,208 bytes up and 4,976,640 bytes down, including padding. QKV remains at
46 resident copies (183,140,352 bytes). Automatic packed host-input bindings
are replicated: MLP input storage is 50,457,600 bytes and attention input
storage 27,131,904 bytes. This choice avoids device preparation for the external
input; intermediate activations still require the planned device exchanges.

Final rendered profiles are `artifacts/fp8-fast/mlp/profile.html` and
`artifacts/fp8-fast/attention/profile.html`; their matching binaries and raw
profiles are `model-panel.ipuexe` and `profile-panel.ipuprof` in each directory.
Gaussian checks pass (MLP maximum absolute error 0.026241; attention 0.000910).
Two-block uneven MLP and two-block four-head fused attention also pass (0.019749
and 0.000170). The storage-coordinate regression checks every element of
rectangular AMP and block-major matrices against the independent host codecs.
142 codegen tests, four workload tests, the doctest, and Clippy pass.

## Host FP8 and quantization before replication

`--fp8-scale` now applies to host activations as well as parameters. The host
encoder writes F143 directly into the selected input binding, removing device
input casts. The output checker also decodes FP8, including Repeat-carried
results. Generic FP8 views use the row-major view path; the specialized attention
view layouts describe F16/F32 storage and must not be reused blindly for FP8.

Mid conversion insertion prices quantization on the producer's owners against
quantization after redistribution. FP8 panel exchange shares the same physical
fragment mapping as F16. Consumer requests preserve 32-element inner groups
through GELU, so suitable producer formats survive shortlisting. Casts can
complete a final half-panel directly with FP8 zeros; no F16 padding copy is
needed. Compact costing treats such a local cast as local even when its physical
padding changes.

Padding *every* 16-column producer shard into a 32-column FP8 panel is not a good
way to enable early quantization. It introduces holes between useful rows and
fragments the following exchange into short packets. The first full-size build
using that approach was stopped during expensive scheduling. Early quantization
therefore requires complete producer panels, allowing a narrow final tail;
other layouts retain the later cast. Both conversion orders are priced.

A 128x128x512 MLP on 64 tiles validates the early path: its cast processes 1,024
values on each tile, exactly 65,536 values (one logical intermediate), before the
GEMM exchange. Maximum cast duration is 1,260 cycles; Gaussian maximum absolute
error is 0.005798. See `artifacts/fp8-before-fanout/small-early/profile.html`.

For canonical batch-one MLP, exact scheduling of four retained finalists gives
refined estimates 126,413, 135,606, 136,572, and 148,626 cycles. The first remains
best and retains quantization after redistribution; early quantization's compute
savings do not compensate for the other available plans' exchange costs. The
resulting hardware execution is **127,872 cropped cycles**, versus 151,998 before
host FP8 loading (15.9% less). Full fused projected attention is **171,588 cycles**,
versus 176,994 (3.1% less), with no activation cast before its QKV projection.

The selected single-block MLP's host-populated input binding occupies
251,353,600 bytes across tiles, including replication/padding; its two weight
bindings occupy 5,509,120 and 4,976,640 bytes. The host path removes input
preparation from device execution, so this timing must not be interpreted as
the cost of preparing the same replicated operand from an on-device producer.

Final Gaussian checks pass (MLP maximum absolute error 0.006378; attention
0.000054). The reference now starts from the host-quantized FP8 activations,
so these error figures are not directly comparable to the former F16-input
reference. Two-block uneven MLP and four-head attention also pass (0.031250 and
0.000977). A width-80 intermediate passes as well. A width-16 intermediate remains
outside the current native FP8 GEMM candidate catalogue.

Profiles and matching packages:

- `artifacts/fp8-before-fanout/mlp-ranked/profile.html` and `model.ipuexe`
- `artifacts/fp8-before-fanout/attention-final/profile.html` and `model.ipuexe`

144 codegen tests, four workload tests, the doctest, and Clippy pass. Tests cover
FP8 physical panel correspondence, preservation of early-cast finalists, and
conversion before replication.


## Row-major producer casting and pipelined casts (2026-09-08)

Row-major F16 producers can now cast directly into FP8 AMP-left panels on their
existing owners. Eligible operator inputs offer both local cast/pack followed
by FP8 redistribution and F16 redistribution followed by casting to the ordinary
region beam. Inputs choose independently, so one graph can mix the two orders;
there is no local winner selection or extra whole-graph cast-policy retry.
Exact local quantizations are reused within a region, so compatible projection
consumers share early quantization. Their availability is part of the beam's
future-state key, since it changes the cost of later consumers. Replicated consumer operands are materialized
separately to avoid extending their much larger allocations across consumers.

The producer must own complete 32-column groups. This deliberately does not pad
every narrow producer shard to enable the path. A single physical row already
has AMP-left's order: its cast uses the linear loop, with no packing pass. For
multiple rows the existing panel cast loop takes separate source and destination
row strides, combining row-major packing and conversion without an F16 staging
allocation. The same primitive cost is used for conversion selection and compact
mid costing; combined casting does not get charged a packing scratch buffer.
Transposed AMP and block-major row-major cast/pack kernels are still separate
missing capabilities, as shown by the optimistic diagnostic.

The linear F16-to-FP8 loop now has the SDK's two-bundle software pipeline:
`ld64step` paired with `f16v8tof8`, then `ldst64pace`. A prologue/epilogue handles
the first and last vector. Small worker assignments use the four-bundle loop.
A runtime range check enables combined load/store only when source and destination
occupy disjoint 32-KiB address groups. This conservatively covers standard banks
and interleaved bank pairs, handles Repeat pointers, and adds no placement
constraint. The old loop remains available for other allocations. Planning
currently retains the conservative four-bundle linear price because bank
separation is unknown until placement.

`cargo run --release -p ipu-tests --bin cast_check -- --sdk "$POPLAR_SDK_ENABLED"`
checks the actual device code against exactly representable FP8 values over
multiple scales, worker tails, AMP half-panel padding, and memory arrangements.
It checks output guards as well as all result bytes. The 8,320-element linear
case took about 2,706 cycles with pipelining versus 4,638 for the historical linear loop;
these include launch/setup. Raw checks and integration profiles are under
`artifacts/fp8-cast/`.

The checker reserves its timestamp storage and waits for the final deferred host
exchange before reading it. Without that wait, the final timestamp batch could
contain the preceding tensor chunk. All 315 current-kernel cases pass; the
historical kernel passes its 252 applicable cases. At that revision, existing packed casts had a
small setup cost increase (8,320 elements: 5,064 to 5,136 cycles); the two-bundle
pipeline applied to linear casts and compatible single-row panels. The
multi-row loop is now pipelined as described below.


Integration validation: the final small ViT (`--vit-small --tiles 64 --fp8-scale=-4`)
passes with maximum absolute error 0.129395 under the benchmark's documented
FP8 tolerance (`--diagnostic-atol 0.2 --diagnostic-rtol 0.05`). Its cropped profile
is 124,524 cycles, under `artifacts/fp8-cast/vit-small/profile.html`.

The full batch-two trial was stopped during final placement, without a hardware
run or new full-size profile. On the final sharing policy, finalists 6 and 4
failed contiguous exchange-table allocations of 119,756 and 105,852 bytes.
`artifacts/fp8-cast/vit-b2/run.log` retains that trial. Other finalists were not
exhausted: this is not proof that every batch-two layout is infeasible. An earlier
trial which also reused replicated consumer buffers is retained separately in
`vit-b2-shared-consumer/`; that broader sharing policy was removed. No memory
budget relaxation, forced layout, or arbitrary transfer-count cutoff was added.


Packed cast pipeline (2026-09-08): both 16-half source streams now pipeline
independently with `ld64step`/`ldst64pace`, issuing eight bundles per 32 outputs
instead of fourteen. A prologue loads each stream's first vector; the final row
drains both without reading a following row. This supports packed and row-major
sources. Runtime eligibility requires disjoint 32-KiB address groups, at least
16 rows per worker, and a source jump fitting the signed 10-bit packed stride.
The pipeline is a separate non-inlined helper to keep its register pressure off
small casts. The general fallback also uses post-increments, reducing its body
to twelve bundles; half-panel zero padding keeps its existing loop. Mid/low cost
estimates use the twelve-bundle price, since bank separation is unknown there.

The hardware checker now covers 441 cases, including the full B1 ViT's 184x96
local cast, pipeline threshold boundaries, a full panel followed by a padded
half-panel, and a row stride too wide for the pipeline. All outputs and guards
pass bitwise at scales -4, 0 and 3. Comparing the pre-change kernel with the final
kernel (scale 0, complete invocation including setup):

| Packed local shape | Placement | Before cycles | After cycles |
| --- | --- | ---: | ---: |
| 184x96 | Separate standard groups | 9,066 | 6,762 |
| 184x96 | Interleaved, separate groups | 9,066 | 6,786 |
| 184x96 | Shared standard groups | 9,066 | 8,172 |
| 65x128 | Separate standard groups | 5,124 | 4,794 |

Small cases can still pay extra setup (the 16-element padded case increases
from 960 to 1,170 cycles). These are kernel measurements, not a rebuilt full ViT
profile. Raw cases, assembly, packages and comparison logs are retained under
`artifacts/fp8-cast/packed-pipeline/` and `artifacts/fp8-cast/packed-baseline/`.

## Pretrained calibration follow-up (2026-09-12)

The global bring-up scale −4 is insufficient for real SigLIP weights: some learned
layernorm outputs exceed 300, whereas that scale represents at most 15.
[Pretrained validation and calibration](SIGLIP_PRETRAINED_VALIDATION_2026_09_12.md)
records the image preprocessing, independent FP32 references, tensor-scale
experiments, historical GPTQ reconstruction and the remaining compiler work.

`tools/quantize_siglip_f143.py --scale-granularity tensor` now retains one scale
per weight matrix while still using small Hessian blocks for GPTQ. Its default
`block` behavior remains available for reproducing historical experiments.
`tools/calibrate_siglip_fixture.py` additionally models independent activation
scales, fused QKV/KV tensors, and the complete MAP embedding output. These tools
write reconstructed floating-point parameters; their calibrated choices are not yet wired
into the benchmark planner. Fixed scales shared across all layers and both
operands also passed the host experiment, so dynamic Repeat scale arguments
are not currently required.
