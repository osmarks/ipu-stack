# Optimistic regional planning

`ipu_codegen::optimistic` is a separate diagnostic high-to-mid search. It does
not change production package selection and its result cannot be passed to a
package builder. A candidate contains whole-device values, explicit transforms,
and existing mid implementation fragments. Missing implementations are explicit
assumptions, not pretend executable kernels.

## Entry points

- `plan_region(graph, request, config, options)` searches a contiguous top-level
  high-graph region. `RegionRequest` fixes formats for its free inputs and
  escaping outputs. Every externally consumed result must be preserved,
  including values used by subsequent Repeat parameter sequences.
- `plan_graph(graph, config, outputs, options)` is the convenience form for a
  small complete graph. Automatic input precisions use row-sharded boundary
  layouts here; unlike production host bindings these boundaries are fixed.
  Use `plan_region` with explicit formats to model already-packed host weights.
- `enumerate_conversions(from, to, tile_count, options)` exposes the conversion
  search independently, without a high graph or operator candidate search.
- `SearchReport::reference()` returns the cheapest retained assumption-free
  candidate. `opportunities()` finds hypothetical candidates whose optimistic
  estimate beats that reference. Neither promises physical feasibility.
- `DiagnosticMidGraph::to_dot()` renders the actual dataflow, external results,
  ownership/replication, costs, and assumptions. Existing algorithms retain
  their inspectable `Arc<MidProgram>` implementation rather than copied IR.

Example (no SDK compilation or device needed):

```sh
cargo run --release -p ipu-codegen --example optimistic-planning -- /tmp/projections.dot
dot -Tsvg /tmp/projections.dot -o /tmp/projections.svg
```

The example constructs LN followed by two FP8 projections. Boundary inputs are
F16, including parameters, so their conversion costs are part of this example;
this is not a benchmark of host-prepacked FP8 model weights.

## Search space

Internal operator algorithms, ownership, replication, and output formats are
selected afresh using the existing unpruned operator candidate constructor.
Existing backward consumer demands and memoized implementation generation/costs
are reused. The normal candidate search's conversion-availability filter is
intentionally not applied. Existing algorithm fragments perform GEMM and
attention decomposition; the diagnostic does not duplicate those generators.

For each producer/consumer boundary, conversion nodes cross the two precisions,
the two storage orders, and the two ownership/memory-class choices. Only valid
resolved layouts with legal packed matrix dimensions survive; a hypothetical
kernel does not make an invalid encoding legal. This exposes producer-local packing and casting,
late casting, and different packing/redistribution orders. It does not insert
extra lossy precision round trips. Paths have one to four steps (default three).
Local cast-and-pack can be hypothetical. Available local kernels are checked
against the shared kernel ABI table; ordinary word-copy unpacking is recognized
separately so lack of a dedicated kernel is not misreported. Single-row
row-major/AMP-left equivalence is a distinct zero-work hypothesis.

Materializations are keyed by logical value and complete tensor format and can
be reused by multiple consumers. This lets Q/K/V-style branches share early
quantization without forcing projection fusion. Conversion paths are cached per
search. Candidates retain their complete dataflow, including where sharing
extends a value's lifetime.

A separate regional rewrite offers short fused Add/GELU/LN chains, including
multiple live outputs. Independent parameter conversions can be moved before
such a group; conversions that depend on its results stop the group. Original
high operations are retained as the fused kernel's semantic definition. The
original unfused candidate remains available. Existing bias/GELU and
add/layernorm fusions reuse the production eligibility contract and mid cost;
only unsupported groups receive a missing-kernel assumption.

## Bounds, costs, and diagnostics

Defaults: eight high operations, sixteen beam entries plus a protected supported
reference, 2,048 operator expansions, eight conversion routes plus a supported
reference, and four operations per proposed fusion. Expansion budget is divided
among remaining operations and beam states so an optimistic branch cannot use
all effort before a reference is visited. Format diversity is retained before
filling a beam with variants of one layout; exact states are deduplicated.
Conversion paths use cost/assumption-set dominance. `truncated` reports bounded
search/pruning; it does not imply exhaustive search or global optimality.

Known operators use existing compact mid prices. Known casts and movement use
existing cost models. Unavailable local kernels use maximum-shard read/write
traffic at eight bytes/cycle, vector-conversion work, and launch overhead;
four times the local estimate supplies a deliberately loose sensitivity price.
Unavailable exchange prices use endpoint traffic and an explicit missing-model
assumption. These are engineering estimates, not proven cycle lower/upper
bounds. There is no exact exchange scheduling or physical tile expansion in
normal search.

Memory reporting sums maximum shard sizes over live intervals and includes
existing implementation scratch conservatively (some endpoints are counted
twice). A separate device-wide live-byte average gives a capacity lower bound;
alias hypotheses do not allocate an extra copy. The search rejects individually
oversized tensors and proven aggregate capacity overflow, including the supplied
support reservation. It does not reject merely because the conservative peak
exceeds SRAM. Bank separation, contiguous placement, exchange rows and generated
code still require physical validation. Region-external model state must be
reflected in the caller's memory budget/boundary inputs.

Assumptions distinguish missing kernels, missing equivalence rules, the normal
early-cast eligibility restriction, and missing cost models. An assumption-free
reference means no known missing capability within this search vocabulary. It
is not a claim that the production beam retained that plan. If there is no
reference, candidates and assumptions are still returned, but `opportunities()`
makes no savings claim.

## Deliberate limits

This is the conversion-focused first application of regional search, not an
arbitrary algorithm synthesizer. It reuses the current operator/layout families;
it does not invent tree algorithms, new sparse encodings, joint physical tile
mappings, or fused/concatenated GEMM projections. Existing attention algorithm
choices are available, but their internal mid fragments are not recursively
rewritten. Elementwise fusion is currently proposed after regional beam search,
so its savings cannot yet rescue a producer layout pruned earlier. Boundary
formats are fixed for one invocation; callers can explore other boundaries or
larger regions. Repeat is rejected explicitly rather than silently flattened.

Regression coverage includes the missing early FP8 cast/pack route, preservation
of a late-cast reference, single-row equivalence, an existing word-copy path
validated through low expansion, shared conversions across branches, live
residual preservation in fusion, fixed boundary validation, memory rejection,
known fusion recognition, packed-geometry validation, and bounded high-region
FP8 search.

## Example result

The LN/two-projection example expanded 1,694 operator candidates in about nine
seconds on the development host, retaining 17 plans including a supported
reference. With the default bounded search (`truncated=true`), the reference
cost was 19,663 cycles; the best hypothetical plan was 16,801 optimistic and
20,923 conservative. Missing combined cast/pack kernels appear explicitly in
its graph. Other retained plans also expose early-cast eligibility and single-row
equivalence opportunities. The overlapping estimates warrant implementation or
microbenchmark investigation, not a hardware speedup claim. These numbers include
conversion of the example's F16 weights.
