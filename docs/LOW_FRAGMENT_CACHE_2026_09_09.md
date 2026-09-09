# Low expansion caching

The production expansion path shares two bounded caches across the finalists
in one selection attempt:

- Relative local-copy descriptors, rebound to the actual source/destination IDs.
  Keys normalize allocation origins but retain precision, element order, padding,
  view geometry, copy order and whether both endpoints are the same buffer.
- Copy preparation plans: padding coverage, staging and packing decisions.
  Keys retain the exact destination type/allocation and ordered source formats,
  allocation extents and source/destination views used by CopyPlan.
  Lookups hash borrowed inputs and allocate owned keys only on retained misses.

Both cache immutable results behind Arc. Hash collisions are checked by full
key equality. Each cache admits at most 32,768 entries; this is an entry bound,
not a byte budget. Generation happens outside the mutex. Concurrent equivalent
misses may generate twice but retain only one result. Cheap whole-buffer cases
bypass caching. Keys contain no placement addresses or profiling provenance.
Deferred materialization, alias updates, loopback selection and phase grouping
continue through the existing assembly code. Nothing is persisted between builds.

## Measurements

Native release build, four serial finalists (only one retained for MLP B2),
same binary with caching enabled/disabled. Includes low construction,
simplification and analytical costing, but excludes search, placement and
exchange scheduling. Individual measurements have host timing noise.

| Workload | Uncached expansion, ms | Cached expansion, ms |
|---|---|---|
| FlashAttention B1 | 900, 860, 980, 934 | 869, 831, 1021, 938 |
| Materialized attention B1 | 744, 729, 1492, 1458 | 777, 716, 1332, 1368 |
| MLP B1 | 292, 215, 238, 244 | 296, 211, 229, 254 |
| MLP B2 Repeat3 | 1351 | 773 |

The useful improvement is MLP B2: about 43%. Materialized attention saves about
5% across all four expansions, despite a small cold-first-finalist regression.
FlashAttention and MLP B1 are essentially unchanged. These are not end-to-end
compiler speedups or evidence that larger search spaces are now cheap.

Copy-plan entries/hits/misses: FlashAttention 7,168/147,712/7,168; materialized
attention 9,436/34,788/9,436; MLP B1 648/5,184/648; MLP B2 967/2,692/967.
Repeat still expands one body, rather than caching three expanded invocations.

Reproduction: use the existing expansion benchmark with
--benchmark-expansion-uncached for the comparison. JSON reports cache
entries/hits/misses. Commands, JSON and logs are under
artifacts/fragment-cache-20260909/final/.

## Rejected broader caching

Per-output GEMM and then whole-compute templates were implemented and measured.
The latter reused 185 compute fragments with only 11 distinct templates in a
FlashAttention sample, but expansion was about 0.99 s versus 0.88 s previously.
Cloning/rebinding the flat output still allocates operand/view vectors and shards;
hashing adds overhead. Those caches were removed from the final implementation,
rather than using hit rates as evidence of speedup. Experiments remain in commits
0be6b62 and 9aae16a. Earlier copy-plan caching also regressed when key allocation
and a too-small entry limit dominated; borrowed lookups and capacity handling
address that in the retained version.

The subsequent [symbolic geometry analysis](LOW_GEOMETRY_ANALYSIS_2026_09_09.md)
implements shared exchange facts and evaluates immutable fragment storage.

## Remaining bottlenecks at this point

A software CPU-clock profile of four materialized-attention expansions identifies:

- Allocation/free: malloc and free alone account for about 12% of sampled
  main-thread CPU. BlockValue cloning is another 2.25% directly attributed.
  TileGraphBuilder clones TensorType for each shard; KernelRun inputs and
  ShardView extents also own vectors. Sharing these immutable structures is a
  more promising prerequisite for broader fragment reuse.
- Storage traversal construction, iteration and summaries: SpanIter::next,
  axis_boxes and byte_traversal individually account for roughly 3.4%, 3.1%
  and 2.6%; summary operations add further cost. Exchange geometry costing
  remains a consumer of these traversals after copy preparation is cached.
- Hashing: DefaultHasher::write is about 7% in this profile. Compact/interned
  geometry keys or a measured faster internal hasher could reduce cache
  overhead; correctness must continue to use exact key comparison.

The capture includes main-thread graph destruction and benchmark setup, not
just expansion, and excludes separate planner worker threads. It used the
earlier 4,096-entry copy-plan limit, before the final 32,768-entry limit.
Percentages identify targets, not precise final-version savings forecasts.
Kernel metadata still uses a linear interning search, but this profile does
not establish it as a major bottleneck. Raw capture: artifacts/fragment-cache-20260909/perf.data.

## Validation

200 codegen tests and the doctest pass (five existing ignored tests); release
Clippy passes. Existing randomized expansion tests compare entire TileGraphs
with caching disabled, cold and warm, including Repeat and deferred aliasing.
Dedicated tests check translated-copy reuse, hash collisions and capacity bounds.

FP8 MLP B2 Repeat3 passed hardware/reference validation, maximum absolute error
0.015625 with diagnostic tolerances 0.2 absolute / 0.05 relative.
The hardware run preceded the final capacity/statistics changes; graph equality
and the full test suite cover those changes. Its rendered profile is
artifacts/fragment-cache-20260909/hardware/profile.html.
