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
padding and placement effects. The experiments below implement this reduction-overlap alternative.

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

## Reduction overlap and fused projection experiments

The next experiments use the same batch-1, 16-head materialized workload and
entry-cropped timing:

| Projection plan | Cycles | Change from 203,256 |
| --- | ---: | ---: |
| Separate, grouped unpack only | 203,256 | baseline |
| Separate, overlap two reductions | 203,736 | +480 |
| Separate, overlap all three reductions | 195,834 | −7,422 (−3.7%) |
| One fused QKV GEMM | 195,096 | −8,160 (−4.0%) |

The mid ownership pass offers deferred independent reductions in groups up to
`PipelineConfig::max_parallel_reductions` (default 3). It retains the original
schedule and disjoint-unpack candidate. Only results consumed by explicit
copies can move, and required outputs retain their owners. Dependency checks
include storage aliases. Deferral crosses copies and read-only products, but
stops at other compute, conversions, repeats, or dependent work. Diagnostic
checkpoint builds do not defer reductions. The pass currently operates on the
top-level region; it does not reschedule a Repeat body.

Low expansion prepares each sum with the same reducer used for standalone
sums, then emits a combined exchange and compute phase for corresponding
stages. The selected triple uses all 1,380 reduction roots together. Two-way
overlap is slightly worse here despite its lower temporary storage requirement;
the extra phase and changed traffic matter.

Fused projection is an explicit graph alternative, selected in the benchmark
with `--fuse-qkv`. Its parameter is `[1152, 3456]`, with Q, K, V concatenated
along the output columns. One GEMM produces the combined output, ordinary
axis slices select its thirds, and ordinary head views feed attention. This
is not an automatic rewrite of existing three-parameter graphs: callers must
supply concatenated weights. `slice` is a general graph/repeat-body operation;
it lowers to the existing mid Copy coordinate mapping, composition, and cost
model rather than a QKV-specific device kernel.

The selected fused projection uses two K partials and the kernel
`gemm_f16_init_small_rows_interleaved_k576_c16_r216_r216` on 1,472 tiles.
Projection compute finishes around cycle 44,640, versus 56,358 for the three
separate GEMMs with overlapped reductions. However, its Q/K preparation still
runs two unpacks on the same 736 owners, consuming most of that advantage. The
end-to-end gain over triple overlap is only 738 cycles (0.38%).

All alternatives fit using the existing allocator and default tile SRAM budget.
The triple retains three sets of three partials: roughly 14.4 MiB of logical
FP16 partials device-wide, compared with 4.8 MiB for one separate projection
at a time. The fused layout retains two wider partials, roughly 9.6 MiB.
These are logical tensor sizes, not measured peak tile allocations; padding,
root buffers, weight replication, and allocator bank constraints also matter.
Maximum packaged exchange-row storage was 8,168 bytes for pairs, 8,104 for
triples, and 7,872 for fusion (8,120 for the previous grouped-unpack plan).

Both overlap cases passed Gaussian final-output validation with maximum
absolute error 0.000061; fusion passed with 0.000056. The final fused package
hash matches the timed package, so it was not timed again. Unit tests cover
slice bounds, delayed-partial hazards, in-place compute barriers, and combined
reduction stages with different contributor counts. The full release codegen
and ELF tests and Clippy passed with the preexisting complexity allowances.

Artifacts and rendered profiles: `artifacts/qkv-fusion/{pairs,overlap,fused}`.
The controls are `--max-parallel-reductions 1|2|3` and `--fuse-qkv`.

A two-block, four-head fused Repeat also passed Gaussian validation, with
maximum absolute error 0.000030 (`artifacts/qkv-fusion/repeat-fused`).
