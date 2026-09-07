# Q/K/V preparation and projection overlap

The materialized SigLIP attention plan currently selects the same projection
store format for Q, K, and V: `Amp(TransposedLeft)`, FP16, interleaved memory,
with a 46 × 10 reduction-result grid (460 owners). Each projection produces
three independent K partials before reduction. The compute kernel is
`gemm_f16_init_large_rows_interleaved_k384_c16_r114_r116`.

Q and K need unpacking before their attention operand packing. V's consumer
can copy panels from this projection order directly; it does not need that
unpack. Thus different *consumer* formats are not evidence of different
projection stores.

## Recovered unpack overlap

In the independent materialized QK plan, composition retains Q's standalone
head view before its replicated operand preparation. K's view is composed into
attention's preparation copy. Previously, copy batching always respected the
semantic-operation boundary between these two independent copies, even when
no diagnostic checkpoints were requested. This also hid the pair from the
existing disjoint reduction-root ownership candidate.

Normal tile expansion now batches across that boundary. Diagnostic expansion
keeps it, because the host must observe the view before later operators run.
Batch dependencies use storage groups, including aliases, rather than only
value IDs. Mixed-operator batches have no single operator provenance.

The ownership candidate still competes against the original owners through
concrete placement and costing. It moves reduction results, not GEMM partials.
For this plan, the two unpacks now occupy 920 distinct tiles in one preparation
phase instead of reusing the same 460 tiles in two phases.

| Materialized attention, batch 1 | Cropped cycles |
| --- | ---: |
| Prefix rename only | 215,832 |
| Grouped preparation and disjoint Q/K roots | 203,256 |

This saves 12,576 cycles (5.8%). Both packages passed the 839,808-element constant
input check with maximum absolute error 0.000930. Artifacts are under
`artifacts/qkv-preparation/{renamed,grouped}`. Timings use the profile renderer's
entry crop, not the full shared-clock interval.

## Projection fusion and reduction overlap

The common projection format makes a fused projection plausible: one larger
output-column problem with Q/K/V weight regions and views of its result. It
still needs a layout candidate that preserves useful per-head partitions;
concatenating matrices alone does not guarantee cheaper preparation. It should
compete with the separate projections, including exchanges and memory costs.

The more bounded alternative is to defer independent reductions until all
three projections have produced partials, assign their results disjoint owners,
and combine their corresponding exchange/reduction stages. The 3 × 460 roots
fit in 1,472 tiles. This does require simultaneous partial lifetimes: retaining
two additional three-partial FP16 tensors represents about 9.6 MiB of logical
data across the device (about 6.7 KiB per tile on average), before physical
padding and placement effects. The present change groups preparation only;
it does not defer or overlap the three projection reductions.

`ipu-trivial-test --reference-run` validates Gaussian final outputs while
retaining normal optimization and profiling. Unlike `--diagnostic-run`, it
does not insert operator checkpoints, so it can test cross-operator batching.
It reuses the existing host reference and logical output verifier.

Validation of the final implementation: Gaussian batch-1 materialized attention
passed with maximum absolute error 0.000061; a two-block, four-head materialized
Repeat passed with 0.000041. The final Gaussian package is byte-identical to
the timed grouped package. 135 codegen tests, four ELF tests, and the codegen
doctest passed. Clippy passed with the existing argument-count/type-complexity
allowances. The final rendered profile is
`artifacts/qkv-preparation/reference/profile.html`.
