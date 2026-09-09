# Low patterns and reuse evaluation

Implemented after the compact-byte-traversal follow-up:

- Regular traversal pairs produce contiguous/strided copy descriptors directly.
  A contiguous endpoint can be split symbolically to match a strided endpoint.
  Irregular pairs retain the previous ordering/coalescing fallback. Both routes
  share the same worker-utilization policy for wide, short row lists.
- Complete 16-by-16 panel grids have an explicit low copy order. Their traversal
  retains nested row/column repeats instead of creating a ShardView pair for each
  panel. Compatible global row-tail padding is retained by a shared helper.
  Irregular boundaries still use clipped physical rectangles. No new mid
  planning layer was introduced.
- Kernel contracts are shared before creating formats, requirements and metadata
  allocations. Tile-specific operand views remain separate. Matrix-call splitting
  retains the shared contract.
- Primitive costing borrows either planned tensor dimensions or emitted physical
  extents. It no longer constructs temporary TensorTypes, shapes or format clones
  for every emitted call. The mid and low cost formulas remain shared.

## Measurements

First retained finalist, pre-placement expansion/simplification/costing only.
Individual native release measurements; host concurrency and compiler activity
introduce noise. Artifacts: `artifacts/low-patterns-20260909/`; previous results:
`artifacts/low-expansion-followup-20260909/`.

| Workload | Previous | New |
|---|---:|---:|
| FlashAttention B1 | 0.933 s | 0.880 s |
| Materialized attention B1 | 0.741 s | 0.747 s |
| MLP B1 | 0.289 s | 0.241 s |
| MLP B2 Repeat3 | 1.408 s | 1.346 s |

These are modest savings, not another order-of-magnitude improvement. The sampled
finalists contain no complete-panel-order exchanges, so their timings do not
measure that pathway's benefit. A dedicated integration test verifies that a
complete grid stays one logical exchange, with the same source/destination byte
pairs as semantic copying. Randomized layout/precision tests compare panel
traversal with independently enumerated physical panels and direct copy generation
with the old span-based copy implementation.

The full existing 197 codegen tests and doctest passed, followed by the new panel
integration test (198 tests in total, five additional tests ignored). FP8
materialized attention passed hardware/reference validation: maximum absolute
error 0.000061, diagnostic tolerances 0.2 absolute / 0.05 relative. That hardware
run used the initial complete-grid implementation; the subsequent shared padding
helper was checked by the full tests. It is not evidence for hardware execution
of the complete-grid path, which that selected plan does not use.

## Caching evaluation

The expansion benchmark now reports `selection_reuse`. This counts normalized
operation selections, retaining boundary tensor types, layouts, tile rotations,
and alias-group relationships while omitting IDs and provenance. It runs outside
the timed expansion. These are potential template reuse counts, not safe cache
hits: resolved deferred views and neighboring assembly state are not included.

| Sampled workload | Compute occurrences / distinct | Copy occurrences / distinct | Sum occurrences / distinct |
|---|---:|---:|---:|
| FlashAttention, 4 finalists | 208 / 13 | 120 / 36 | 12 / 4 |
| Materialized attention, 4 finalists | 32 / 7 | 48 / 17 | 16 / 5 |
| MLP B1, 4 finalists | 12 / 9 | 7 / 6 | 8 / 5 |
| MLP B2 Repeat3, 1 retained finalist | 3 / 3 | 2 / 2 | 2 / 2 |

Attention offers substantial repetition. The retained MLP B2 case does not: Repeat
already retains one body instead of expanding three copies. Counts are unweighted
by expansion time and describe retained finalists, not the entire beam search.
They do not imply the same hit rate or speedup throughout search.

A useful cache should retain immutable, relocatable low fragments with explicit
input/output bindings, not clone a complete TileGraph. Cloning still allocates
its many shards, views, calls and transfer recipients. The cache key must include
resolved input views/ownership and alias relationships, and the fragment must
express any output alias/materialization effects. IDs and profiling provenance
can be supplied at instantiation.

Keep final assembly outside that cache:

- `full_view` resolves deferred materializations from builder state.
- Copy/reduction prefixes group neighboring independent operations.
- Local loopback eligibility depends on receivers in the combined exchange.
- Adjacent exchanges can merge based on storage roots and intervening work.
- Repeat establishes aliases and iterated bindings across its boundary.

Cache the corresponding grouped preparation result or assemble cached preparation
fragments through the existing rules; do not cache an operation's final phases
using only its MidOperation and tensor shapes. A safe implementation therefore
needs an explicit fragment boundary first. No production graph cache is enabled
by these changes. Kernel-contract sharing is the implemented, narrower reuse.
