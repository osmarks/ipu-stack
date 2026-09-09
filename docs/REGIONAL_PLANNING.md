# Bounded regional planning experiment

An opt-in production path now constructs a complete seed, validates it, and
tries bounded replacements of annotated high-graph regions. The ordinary
whole-graph planner remains the default. This experiment implements the proposed
baseline/regions/incumbent/boundary/bounds design; it does not yet demonstrate a
fast, generally reliable replacement for whole-graph search.

## Interface

After constructing the graph, annotate nonoverlapping ranges of top-level
operations (end exclusive):

```rust,ignore
graph.add_planning_region(0..3)?;
config.regional_planning = Some(RegionalPlanning {
    max_evaluations: 4,
    ..Default::default()
});
```

Only annotated regions are optimized. With no annotations, each top-level
operation is a region. Repeat is one structured operation; its body can be
replanned as part of that region, but there are no nested annotations yet.
Precision, operator availability, and explicit input formats remain caller
choices. This does not use the separate optimistic missing-kernel diagnostic.

The test CLI exposes `--regional-planning`, `--regional-evaluations N`, and
repeatable `--planning-region START:END`. Zero evaluations builds only the seed.
Library defaults are three seed attempts, beam width eight, at most four proposals
per region, twelve global evaluations, at most four evaluations per region, and one pass. The CLI defaults to four
global evaluations. `--regional-evaluations-per-region` independently caps full
validation after ranking each region's local proposals. Budgets bound candidate counts, not wall-clock runtime.

`--benchmark-selection report.json` runs selection through placement and exchange
scheduling without compiling executable support. Its report explicitly marks
`executable_support_validated: false`. Use a normal package build and reference
run to establish executable feasibility and correctness.

## Implementation

1. **Constrained deterministic seed.** Reuse the existing planner with divisor
   GEMM grids, balanced row-major operation boundaries, one concurrent reduction,
   and no extra shape-derived active tile counts. Automatic host inputs receive
   the same balanced row/grid ownership; parameters remain eligible for direct
   packed loading. Three attempts use beam widths 2/8/16 and progressively
   stronger exchange-table penalties. This is a bounded beam, not a greedy
   algorithm or a proof that a seed always exists. An attempt must pass the full
   production package finalizer before becoming the incumbent.
2. **Explicit regions.** Source operation IDs locate each contiguous region in
   the unresolved mid program. The existing operator candidate generator searches
   only that high region. Outside selections and compact implementation references
   are retained.
3. **Feasible incumbent.** Low expansion, footprint checks, placement, scheduling,
   and executable support validation run on a proposed complete replacement.
   Only a strictly lower finalized package cycle estimate replaces the incumbent. Failed
   search, memory, exchange, or package validation retains the previous artifact.
4. **Boundary contracts.** Live activations and shared parameters preserve their
   tensor types and tile ownership. Automatically laid-out parameters used only
   inside the region may select a new host-loaded layout. Precision and shape
   remain fixed; explicit caller-specified formats are respected. Escaping outputs are restored to the incumbent's tensor type and
   ownership. Internal layouts and operator implementations remain searchable.
   Suffix consumers, aliases, and structured Repeat references are rebound.
   The caller's physical tile map is fixed; absent one, regional mode uses identity.
5. **Bounds before detailed ranking.** A cheap GEMM throughput lower bound uses
   total work and available compute tiles, including Repeat counts. Its deliberately
   optimistic ceiling is 1024 FLOPs per tile-cycle, above supported kernels' peak.
   It ignores non-GEMM work. Exchange estimates rank candidates; they are not
   promoted to mathematical bounds. Existing memory and exchange footprint screens
   still run during validation.

For supported boundaries, proposals first receive local low-graph costing. This
ranks alternatives without expanding the whole graph for each one. Local failure
is not treated as proof of global infeasibility. Full validation remains necessary
for phase grouping, allocation lifetimes, and executable storage. Copy descriptor
and preparation caches are shared across local scoring and global validation.
Regional candidate generation uses the existing bounded eight-thread planner pool.

## Deliberate limits

- SRAM placement is still global. A local layout change can alter global addresses
  and require another schedule. No arena isolation or per-region reservation was
  added. The implementation retains outside mid choices, not all physical artifacts.
- Some boundaries are conservatively skipped: noncontiguous source attribution,
  multiple live representations of one high value, or escaping deferred producers.
  Region-local costing also falls back when input aliases cannot be represented
  as independent boundary allocations.
- The seed's canonical boundaries can create substantial extra conversions. A
  validated seed is an upper bound on cost, not necessarily a good execution plan.
- The throughput lower bound is weak. No candidates were eliminated by it in the
  initial full-size trials. Stronger defensible memory/work bounds remain future
  work; heuristics must not silently reject feasible improvements.
- Fixed output contracts constrain one-region-at-a-time improvement. Annotate a
  larger region to search across an internal canonical boundary. Changing public
  boundary contracts is a separate optimization problem.

## Validation and measurements (2026-09-09)

Artifacts and logs are under `artifacts/regional-planning-20260909/`.

The full-size FP8 MLP B1, one block, used region `0:3` and one replacement
validation. The first seed failed placement; the second passed executable package
validation. It scored 251,525 scheduled cycles. The replacement scored 273,219
cycles after final placement, so the seed was retained. Building and validating
both took approximately 151 seconds, dominated by physical scheduling and package
placement work. Hardware execution and sampled numerical validation passed
(maximum absolute error 0.005859). The rendered profile is
`hardware/profile.html`. These are feasibility results, not a speedup claim.

The small complete FP16 ViT with two global evaluations passed compiler-only
selection in 2.51 seconds at 223,858 scheduled cycles. Neither proposal improved
the incumbent. The tiny FP8 ViT rejects its first GEMM with both regional and
ordinary planning; this is not a new regional-planner regression. Changing host
row splitting did not fix it and was not retained.

Earlier `canonical-*` MLP measurements predate the fixed identity tile map and a
singleton-axis bug fix. They are retained for diagnosis, not final performance
comparisons. The full-size B1 hardware run above includes those fixes.

The full-size FP8 MLP B2 with three distinct Repeat iterations also passed
executable package and hardware reference validation (maximum absolute error
0.015625). Seed attempt two was retained at 1,154,313 scheduled cycles across
all three iterations; one regional proposal was rejected. Selection and package
validation took 137 seconds. See `mlp-b2-final/profile.html`; detailed profiling
still covers only the first iteration. Its temporary whole-row input-policy
experiment produces the same input layout at this shape (all available column
partition capacity is already consumed by rows).

Regression tests cover deterministic seeds, residual consumers, preserved
boundary types, Repeat with distinct iterated weights, late-finalizer failure
retaining the incumbent, and a small MLP whose faster complete replacement is
accepted together with its matching artifact.

A subsequent full package build of the small FP16 ViT also passed hardware and
sampled numerical validation (maximum absolute error 0.003662), retaining the
same 223,858-cycle incumbent after two proposals. Its profile is
`vit-small-hardware/profile.html`. The final codegen suite passed 206 tests
(five ignored), its documentation test passed, and release Clippy passed for
codegen and ipu-tests with the repository's existing argument-count/type-complexity
allowances.

The subsequent full-size ViT trial and measured MLP regression comparison are in
[REGIONAL_VIT_2026_09_09.md](REGIONAL_VIT_2026_09_09.md). That trial found and fixed
the FP8 CLI's loss of smaller GEMM tile families, which caused the tiny FP8 failure
reported above. Default batch-2 Repeat3 MLP performance remains 846,120 measured
cycles; the regional seed is slower at 1,084,056 measured cycles.

Region-private weight layouts are now searched using the same automatic-input
retargeting as initial planning. Outside high-graph consumers (including Repeat
sequences and graph outputs) and outside physical storage-group references keep
shared parameters fixed. Input IDs/names remain stable; packaging reads the
selected input type, so the host supplies the new representation directly. Tests
cover avoiding device weight conversions, shared weights, explicit formats, and
Repeat. Final package costs are also propagated back to both regional and
ordinary finalist selection instead of retaining the provisional schedule score.
