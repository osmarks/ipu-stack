# Cast ordering on the expanded mid graph

Cast ordering is now a local-search choice over mid primitives, including
casts introduced inside GEMM and attention implementations and Repeat bodies.
The input-conversion routine no longer implements an early-cast branch.

The sequence is:

1. Expand selected operator implementations, retaining their copy boundaries.
2. Enumerate eligible FP16-to-FP8 casts and apply the recipe's selected motions.
3. Compose copies, then run the existing producer/elementwise fusion passes.
4. Price the complete program and use existing local-search validation,
   including low expansion, scheduling, and physical placement.

A proposal moves one cast ahead of its single-use, coordinate-preserving copy
chain, using the earliest supported source layout. Independent casts are
separate proposals; search does not enumerate their Cartesian product. The
late alternative remains available by removing that choice from the recipe.
Matching early casts can share a result within a region when no intervening
in-place operation can change the source.

The pass uses the existing FP8 producer-layout capability checks. It handles
identity redistribution, replication, and compatible cropping/padding. It does
not cross arbitrary views/permutations, escaped intermediates, in-place
compute, or nested region boundaries. It visits each Repeat body internally.
A crop followed by padding cannot be collapsed if that would resurrect data
which the original chain discarded. Unsupported source layouts keep their
original cast location; this does not invent a new packing kernel.

Recipe keys identify a high-level source operation and the ordinal of its cast
before rewrites, rather than mutable value IDs. New checkpoints save individual
`cast_before_copies` choices. Legacy `early_casts` requests are translated at
expansion; their old search-visit records are discarded because they describe
a different choice space. The capacity baseline's initial early/late comparison
uses the same rewrite and one shared cycle/memory analysis of the resulting
fragment. Local-search logs include changed cast sites.

Two integration details mattered:

- Copy composition previously erased canonical staging between an unsupported
  linear GeLU output and a replicated GEMM operand. Moving cast search before
  composition preserves that legal location; moving the pass after composition
  regressed the existing MLP case.
- The FP8 layout helper could request shard padding while retaining a
  `Padding::Reject` axis. Producer-local packing now permits zero padding and
  verifies the resulting layout before offering it.

## Validation

Tests cover independent Q/K choices with odd channel/key tails, native K inputs,
low expansion and kernel arguments, checkpoint round trips, shared casts,
Repeat bindings, escaping values, in-place barriers, and preservation of the
intermediate staging location. Existing MLP early-cast and local-search tests
remain part of the regression suite.

Hardware artifacts and scripts are under `artifacts/mid-cast-order-20260913/`.
The QK smoke comparison uses 4 heads, 17 queries, 19 keys, 72 channels,
materialized FP8 QK, and FP16 PV. `qk-search` resumes the late recipe and lets
ordinary local search choose the casts. It selects both Q and K without an
operator-specific switch. Two resident inferences pass against reference.

The full-model replay loads the existing BS1 27-layer SigLIP checkpoint with
zero additional optimization steps and validates two resident inferences
against the FP32 reference. `bs1` is an intermediate implementation;
`bs1-final` is the final replay. Neither test changes the chosen GEMM grids or
requests FP8 QK for the full model.

Final measurements (renderer-cropped IPU execution):

| Workload | Previous / late casts | New path | Change |
|---|---:|---:|---:|
| FP8 QK smoke, ordinary local search | 24,276 | 20,976 | -13.59% |
| BS1 SigLIP, 27 layers, saved recipe | 10,371,582 | 10,131,456 | -2.32% |

The QK search accepts K's cast and then Q's as independent improvements; its
small tile-mapping change is also included. The controlled both-early run
without that mapping change measures 20,910 cycles. QK reference maximum
absolute error is 0.000717 for both resident inferences. The full-model final
runtime is 6.754304 ms; FP32-reference cosine is 0.994243808 (previously
0.994352454). This also permits different producer fusions, so embeddings need
not remain bit-identical. It passes the requested 0.99 cosine threshold.

The final full-model profile is `bs1-final/model.html` with its sibling `.data`
folder. Its replayed checkpoint has individual cast sites and no legacy
`early_casts` field. The codegen suite passes 292 tests with five ignored;
focused cast-motion tests, six local-search/checkpoint tests, and the workspace
check also pass. Logs are retained beside the hardware artifacts.
