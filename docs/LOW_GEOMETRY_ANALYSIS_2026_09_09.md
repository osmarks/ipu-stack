# Symbolic exchange analysis and shared geometry facts

## Implemented

Regular source/destination traversals now share one symbolic row-matching helper
with local-copy generation. It supports matching strided rows and a contiguous
endpoint split to the other's rows. Fragment counts are row count multiplied by
chunks per row, rather than a walk through every row. A million-row test checks
that the representation remains symbolic. Incompatible row widths, irregular
pieces and other patterns retain exact span enumeration.

Exchange table analysis consumes receive-row descriptors and computes chunk
counts, long-send counts, pointer resets and continuation arithmetically.
Ordered row-sharing fingerprints use associative sequence concatenation and
binary repetition. This makes repeated row/chunk signatures logarithmic in the
repeat count while preserving the event sequence. Single-chunk transfers retain
the inexpensive event-at-a-time path. Loopback preserves interleaved TX/RX event
order. Sharing remains a prediction; encoded row comparison is still authoritative.

Each finalist has a GeometryAnalysis shared by execution-cycle and exchange-table
analysis. It interns allocation-relative view geometry and retains one copy recipe
per pair of geometry IDs. Relative keys retain precision, element order,
allocation extents and view extents, including padding. Tile IDs and addresses
are bound by the consuming phase walk. Multicast sources are looked up once per
logical transfer, not again for every recipient. Incoming tile load, shared TX
lane load, maximum fragmentation and cross-transfer pointer state remain explicit.

Kernel costing retains the few physical shape variants associated with each
interned kernel contract. Lookups compare borrowed operand widths; no temporary
TensorTypes, owned shape keys or per-tile cost vectors are cloned. Per-tile
timeline accumulation and barrier/Repeat composition are unchanged.

These changes stay within low expansion/analysis. They add no mid planning layer
or new search choices. The caches are scoped to a finalist; they are not global
or placement-address caches. The older local-copy and preparation caches remain
shared across finalists.

## Measurements

Native release binaries, eight pinned cores, serial finalist expansion, with
planning outside the measured regions. Baseline is b35c7b9 plus diagnostic-only
footprint timing; its source patch and binary are retained in
artifacts/geometry-analysis-20260909/baseline/. New commands, binaries, logs and
JSON are in artifacts/geometry-analysis-20260909/final/.

First finalist: expansion (including analytical execution costing) plus exchange
footprint screening, milliseconds. Search, placement, scheduling, linking and
hardware execution are excluded.

| Workload | Before: expansion + footprint | After: expansion + footprint |
|---|---:|---:|
| Materialized attention B1 | 763 + 237 | 465 + 78 |
| FlashAttention B1 | 893 + 281 | 816 + 140 |
| MLP B1 | 298 + 104 | 276 + 76 |
| MLP B2 Repeat3 | 798 + 369 | 647 + 166 |

The execution-costing subphase itself, milliseconds:

| Workload | Before | After |
|---|---:|---:|
| Materialized attention B1 | 177 | 64 |
| FlashAttention B1 | 170 | 110 |
| MLP B1 | 91 | 66 |
| MLP B2 Repeat3 | 257 | 121 |

Four finalists were measured for each B1 workload, and one retained finalist for
MLP B2. Materialized attention's more fragmented third finalist changed from
1450 + 549 ms to 741 + 152 ms. All sampled shard, kernel, local-copy, logical
transfer, recipient and phase counts match the baseline.

Host measurements vary with allocation state and CPU activity. The diagnostic
clone probes below run outside the measured regions but can perturb later
finalists' allocator state. The first-finalist table and execution-cost subphase
are the clearest comparisons; these are not claimed end-to-end build speedups.

The first FlashAttention finalist has 93 distinct view geometries and 96 distinct
copy recipes. The materialized-attention finalists with approximately 118,000
recipients retain 1,174 view geometries, 3,182 pairs and 4,470 receive-row
descriptors; those with approximately 238,000 recipients retain 4,877 geometries,
7,204 pairs and 8,492 descriptors. Geometry equivalence is therefore useful even
without a whole-fragment representation.

## Evaluation of shareable low fragments

The benchmark separately clones the complete shard array and kernel-run array,
timing allocation/copying but excluding destruction. Combined probe times in
milliseconds, in finalist order:

| Workload | Shard + kernel-run clone time |
|---|---|
| Materialized attention B1 | 14, 17, 19, 16 |
| FlashAttention B1 | 79, 68, 63, 49 |
| MLP B1 | 9, 8, 8, 9 |
| MLP B2 Repeat3 | 12 |

These probes quantify the copying that immutable sharing could avoid. They are
not predicted savings for a fragment implementation: complete arrays include
data outside any reusable fragment, while actual fragment reuse could also avoid
some construction and validation work that the probes do not measure.

The earlier whole-compute cache had 185 hits with only 11 distinct templates in
FlashAttention, yet regressed expansion. These measurements support a more
specific conclusion: avoiding output cloning alone is a modest opportunity,
especially for MLP and materialized attention. A larger benefit would require
retaining executable fragments through low consumers, not caching templates and
then flattening/cloning them before costing.

A real implementation would need instance-local references, explicit boundary
bindings and a way to handle alias/materialization effects. Low simplification,
phase grouping, placement and final emission currently consume mutable flat
arrays with global IDs. An eager flattening step would largely reintroduce the
cost. Copy-on-write fragments could also lose sharing at those mutation points.

Recommendation: do not introduce the full fragment representation solely on the
strength of cache hit counts. First consider sharing immutable TensorTypes and
operand/view storage in the existing low representation, then profile construction
again. That would address measured allocation costs and reduce the cost of any
future fragment implementation without changing the program's execution model.
No production fragment-graph rewrite was made in this change.

## Validation

202 codegen tests and the doctest pass; five existing tests remain ignored.
Release Clippy passes. During tests, the production phase analysis is checked
against an independent copy-span/chunk walk, including traffic loads and complete
row-storage state. Kernel cache hits are checked against direct costing.
Additional tests cover million-row symbolic counting and repeated row/chunk
signatures, including zero lengths/counts, short/long chunks, tails, disjoint
allocations, pointer continuation, and loopback.

These are compiler analysis changes; no new hardware benchmark was required for
the timings above. Local-copy generation uses the shared row matcher and retains
the existing randomized comparisons against span-based generation.
