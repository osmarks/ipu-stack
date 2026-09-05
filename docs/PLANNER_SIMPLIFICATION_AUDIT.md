> This document records historical findings and possible directions, not a task
> checklist. Current priorities are shared mid-level copy/view semantics, clear
> kernel contracts, and fewer representations. The two-pass exchange scheduler
> can remain while those boundaries are improved.

# Planner simplification audit

At the original audit baseline, `ipu-codegen` contained about 32,500 lines of
Rust. The production parts of the former `mid.rs` and `low.rs` accounted for
about 12,500 lines. Most of the
avoidable complexity does not come from supporting several algorithms; it
comes from representing the same decisions independently during candidate
generation, beam search, costing, conversion planning, tile lowering, and
package construction.

A release-mode build of the canonical batch-one SigLIP MLP on 1,472 tiles at
commit `43c947e` measured the current search shape:

- mid-level planning took 28.7 seconds;
- its three operators expanded 1,424, 231, and 2,197 complete branches before
  retaining 64 at each boundary;
- individual GEMM searches generated as many as 6,472 precise variants before
  retaining 64; and
- package construction spent another 10.5 and 9.3 seconds scheduling the same
  exchanges before and after final storage placement.

These measurements are a useful baseline for judging simplifications. They
also show that reducing the number of independently expanded choices is more
important than micro-optimizing the beam container.

## September 4 checkpoint

The complete compiler is retained. The graph-only replacement is saved at
`archive/graph-only-restructure-2026-09-04` (`5f60b5a`). The audit below records
the original priorities, not a claim that all remain unimplemented.

Completed: shared logical shard geometry; operation variants owning their plans;
a single conversion construction path and format authority; removal of duplicate
GEMM kernel specifications; paired transition/deferred cost queries; common exact
Pareto metrics; removal of unused target/scheduling/aliasing options and the
profiling/memory-relation wrappers. GEMM and MLP smoke verification now follows
compiler storage metadata instead of assuming slice order defines tensor order.

The follow-up pass normalizes catalogue candidates and selected operations onto
one `OperatorPlan`, removes the intermediate private plan representation, and
passes that plan directly to costing. GEMM candidate expansion constructs final
dispatches once without intermediate result-layout lists. Physical GEMM operand
indices and matrix axes are shared by planning, estimation, and lowering.

Remaining structural work: canonical resolved layout caching and capacity queries, generic semantic views, shared
address-independent compute/exchange stages, and avoiding the second physical
exchange scheduling pass. These need separate measured changes; this checkpoint
does not introduce an adapter around the existing scheduling algorithms.

Validation of this checkpoint:

- `cargo test --release --workspace`: 132 tests pass, including randomized
  logical-I/O permutation and corruption checks. Existing compiler tests remain.
- Full-device GEMM: 138,674,176 exact checks pass on 1,472 tiles.
- 64-tile GEMM and batched GEMM: 262,144 and 786,432 exact checks pass.
- Canonical batch-one SigLIP MLP: maximum absolute error 0.011719.
- Random-input attention smoke: 4,896 checks, maximum error 0.000113.
- Projected SigLIP attention: 839,808 checks, maximum error 0.000930.
- Clippy passes with `-D warnings -A clippy::too_many_arguments -A
  clippy::type_complexity`. The unqualified README command still flags existing
  long signatures and composite types; this change does not hide those lints
  or add parameter objects solely to satisfy them.

The numerical GEMM fault was in the harness: it decoded output using AMP
`Output` order while the F16 planner selected AMP `Left` result order. It also
inferred logical ownership from binding-slice position. Input packing and
verification now use the compiler's shard maps, retaining exact one-hot checks.
Device initialization occasionally needed a reset/retry independently of that
numerical fault; this checkpoint does not claim to repair that startup issue.

Follow-up validation: all 132 release tests and the same Clippy command pass.
The pass removes 142 production Rust lines (92 net including test adaptations).
Generated 64-tile GEMM, canonical batch-one MLP, and projected attention packages
are byte-for-byte identical to the preceding checkpoint. MLP and attention pass
again on hardware with the same numerical errors; GEMM passes all 262,144 exact
checks after startup retries. Startup code remains unchanged.

## Modularization and materialization follow-up

The former `mid.rs` and `low.rs` are split by responsibility, retaining their
public interfaces. Cost and memory estimation now share the `estimate` module,
with separate geometry, traffic, liveness, and cycle-pricing implementations.
See [Architecture](ARCHITECTURE.md) for the module map.

Mid planning now commits deferred-output offers and conversion materialization.
Previously, low lowering reconsidered streaming using the next operation; two
operand conversions could prevent the first from streaming despite the selected
plan and its memory estimate. Lowering now follows `DispatchSlices` directly.
A regression test forces this nonadjacent-consumer case and checks that canonical
result shards stay unmaterialized. Unclaimed deferred offers are cleared during
planning instead of being rediscovered during lowering.

The pass also consolidates matrix-layout constructors and the region-liveness
peak calculation, and removes the temporary startup reset/retry implementation.
The posted-write readback fix remains in the driver. Startup/run validation and
its limits are recorded in [Bring-up](BRINGUP.md).

Validation: all 133 release workspace tests pass, as does Clippy with the two
existing allowances documented above. GEMM, batched GEMM, canonical batch-one
MLP, projected attention, and attention smoke pass on hardware without automatic
recovery. All five generated packages are byte-for-byte identical to the prior
checkpoint packages. This pass removes 113 Rust lines overall, including test
changes; the largest newly split module is 1,706 lines.

The remaining structural work listed above is still open. In particular, physical
fragment/staging choices still use concrete spans in low lowering; this pass does
not introduce a shared execution-stage representation or resolved-layout cache.

## September 5: shared resolved geometry

`mid/resolved` now owns layout validation, axis partition resolution, logical
bounds, and physical capacities. `Layout::shard_extents`, GEMM traffic, memory
estimates, and operator capacity checks use that representation. Independent
`TileAxisPlan` arithmetic and the operator validator's average-shard formulas
are removed. Traffic retains a resolved operand/output across its tile loop;
source capacity is also calculated once outside the outgoing-bus loop.

This fixes two discrepancies:

- GEMM traffic's simplified bounds included padding from the next logical group
  and omitted per-shard padding in maximum-extent queries. Logical ownership and
  allocation capacity now have distinct queries over the same partitions.
- Estimates independently rebuilt parallel-GEMM partial storage using the kernel
  column block. Only low lowering preserved the selected ownership grain and
  per-shard padding. `OperatorDispatch::gemm_partial_tensor` now supplies both.

New tests cover exact grouped-padding traffic, partial capacities in both GEMM
orientations, and randomized agreement with physical storage for grouped,
replicated, and linear layouts.

Hardware validation also exposed an existing completion-check mismatch: the
runtime's terminal branch to zero can leave an explicit invalid-PC exception.
The checker now recognizes only that named terminal instruction with its
completion flag set. Runtime instruction bytes are unchanged; see
[Bring-up](BRINGUP.md#completion-state-checking) for the evidence and limits.

Validation: 136 release workspace tests and Clippy with the existing two
allowances pass. Fresh GEMM, batched GEMM, canonical MLP, projected attention,
and attention-smoke workloads pass numerically; 20 additional attention-smoke
loads/runs also pass consecutively. Every tile image is byte-for-byte
identical to the preceding checkpoint; packages add one completion symbol.
Final-run MLP planning took 25.5 seconds versus the preceding recorded 30.2;
attention took 15.0 versus 15.5 seconds. These single-run timings are observations,
not a controlled performance benchmark. Production Rust is 44 lines smaller;
155 lines of regression tests bring the overall Rust total up by 111 lines.

Resolution is reused within each consumer; there is no compilation-wide layout
cache yet. Generic semantic views, shared execution stages, and eliminating the
second physical exchange-scheduling pass remain separate follow-ups. Low GEMM
still constructs dispatch slices when its compute grid differs from final result
ownership; this is not yet a shared stage representation.

## Intended architecture

The desired flow is:

```text
ComputeGraph
  -> semantic operators and views
  -> typed, shape-dependent implementation plans
  -> resolved layouts and address-independent execution stages
  -> cost, memory, and beam selection over those same stages
  -> physical placement
  -> tile kernels and exchange programs
```

Costing and lowering should consume the same resolved layout and stage
descriptions. A later layer may add physical information, but it should not
reconstruct an earlier layer's decisions from loosely related fields.

## Priority changes

### Resolve layouts once

One canonical `ResolvedLayout` should be constructed from `(TensorShape,
Layout)`. It should own the padded shape, logical and physical shard extents,
tile ownership, per-tile allocation sizes, and axis-partition facts.

The previous implementation separately reconstructed these facts in `mid`,
`estimate`, and `low`, then interpreted them again while producing physical
byte spans in `storage`. This made every new grouping or padding feature a
cross-layer change and allowed validation, estimates, and emitted shards to
disagree.

Physical element order remains a separate concern: `ResolvedLayout` describes
which physical tensor elements each tile owns, while `storage` maps a resolved
view to byte spans for row-major, block-major, and AMP encodings.

### Separate implementation families from the search domain

The operator-candidate list currently combines kernel availability, precision,
active tile counts, layouts, GEMM grids, memory classes, and staging policy.
`plans()` then adds separate shape-dependent SplitHeads, attention, pointwise,
GEMM, and parameter-storage variants.

Replace this with a small internal implementation catalogue and an explicit
search domain. Active tile counts, allowed precisions, memory classes, and
diagnostic restrictions should be planner inputs, not duplicated seed
candidates. Layout-transparent pointwise implementations should be described
once rather than once per tile count.

### Normalize plan representations

`OperatorCandidate`, the private `Plan`, and `OperatorPlan` repeat most of the
same data. `MidOperationKind::Operator` repeats the operator again. They should
share one immutable implementation core, with candidate-only format policy and
selected-plan estimates stored separately.

The crate's public API should also be narrowed. Only `ipu-tests` depends on
`ipu-codegen`, and it does not name most of the publicly re-exported planner
internals. There is no compatibility requirement that justifies making these
representations difficult to change.

### Represent view operations generically

SplitHeads currently appears as a graph operator, mid-level operator, dispatch,
the only deferred-transform variant, several cost paths, and several lowering
paths. Replace that stack with a generic affine/index view representation.

A view remains free until a consumer requires storage. The ordinary conversion
planner can then alias it, retile it directly, materialize it, or populate
bounded consumer slices. Split, reshape, transpose, and future view-like
operations should use the same mechanism.

### Normalize GEMM plans

GEMM precision and blocking currently appear in `MidOperator`,
`OperatorDispatch`, two `TileKernelSpec` values, and operand layouts. Candidate
generation mutates these copies together and relies on a large validator to
reject inconsistent combinations.

Introduce a compact GEMM plan containing orientation, grid, block dimensions,
parameter placement, reduction buffering, and output ownership. Derive layouts,
kernel modes, memory constraints, and initialize/accumulate calls from it.
Output-stationary execution is the `k_partitions == 1` case; complete and
streamed reduction buffering remain real choices inside the same stage model.

### Share an address-independent stage plan

The cost model currently reconstructs compute work, exchange work, row-table
footprint, transition cost, deferred-input work, and scratch memory separately
from tile lowering. An operator plan should instead expand once into abstract
compute, exchange, local-transformation, and buffer-lifetime stages. Both the
estimator and tile lowering should consume those stages.

This is the structural way to ensure that a plan is priced as it will be
emitted, without materializing every per-tile record during beam search.

### Use one Pareto vocabulary

Cheap GEMM-grid pruning, precise operator pruning, and global beam pruning use
three objective structures and manually maintained compatibility projections.
Use one `PlanMetrics` vocabulary and one canonical layout-family key. A cheap
GEMM estimate may remain as an explicit lower bound, but should not become a
second independent cost model.

### Avoid scheduling exchanges twice

Package construction currently performs an exact physical schedule to size
exchange-row storage, places final storage, then performs the exact schedule
again. Reserve a conservative row-table bound from logical transfers and run
the physical scheduler once. When exact finalist scheduling is requested,
cache reusable results rather than immediately recomputing them.

## Low-risk removals

The following currently add representation or API surface without production
behavior:

- the one-variant, unread `HardwareTarget` and `SchedulingPolicy` fields;
- `ProfilingConfig`, which wraps one Boolean;
- `OutputAliasing::MustAliasInput`, which is never constructed;
- the one-variant `MemoryRelation` enum;
- the public `CostModel` abstraction, whose only non-IPU21 implementation is a
  test fake; and
- public re-exports of planner internals unused outside `ipu-codegen`.

Diagnostic GEMM constraints, forced attention strategies, forced conversion
materialization, checkpoints, and exchange diagnostics are useful, but should
be separated from the ordinary planner configuration.

The pointwise whole-head attention implementation and matching-wave exchange
scheduler are plausible removal candidates, but should first be evaluated over
the stored benchmark and schedule corpus.

## Complexity to retain

The following describe genuine hardware constraints or useful memory/performance
tradeoffs and should be unified rather than removed:

- standard versus interleaved tile memory;
- normal versus swapped GEMM orientation;
- complete versus streamed reduction buffering;
- blocked versus materialized attention while each wins for some shapes;
- structured Repeat;
- compact `TileWork` arenas; and
- exchange schedule replay, diagnostics, full-duplex validation, and randomized
  exchange tests.

## Suggested order

1. Add canonical layout resolution and migrate every layout consumer.
2. Remove dead configuration and narrow the public planner API.
3. Consolidate candidate and selected-plan representations.
4. Replace deferred SplitHeads machinery with generic views.
5. Normalize GEMM plans and their lowering.
6. Introduce shared address-independent execution stages.
7. Consolidate Pareto pruning and remove the second package scheduling pass.

Each step should preserve numerical hardware tests and compare canonical MLP
and attention planning/runtime results against the profiles recorded before the
change.

## View and kernel checkpoint (2026-09-05)

- Compiler implementation modules are private, with explicit external exports.
- Mid-level SplitHeads and the one-off deferred transform are replaced by one
  rank-independent axis-factor view. Both materialized and deferred mapping use
  its shape validation and coordinate transform. General permutation and view
  composition remain open; existing graph SplitHeads is semantic syntax.
- Kernel specialization keys are derived once by shared code for collection and
  call lookup. One symbol map replaces the per-family maps and redundant GEMM row
  map. ABI scalar values are typed rather than string-dispatched.
- Exchange optimization choices can be replayed with final addresses, checked
  against hazards and normalized rows; this retains the existing two-pass model.

Remaining concrete boundary problem: low/conversion still decides between direct
word-fragment exchange and staging plus a local transform, calling the estimator
from low-level lowering. That policy belongs with the selected mid-level copy
plan. It needs actual relative span geometry, not another independent heuristic.

Validation for this checkpoint: 140 workspace release tests (including doctests),
strict Clippy, and all five canonical hardware workloads pass. Every tile image
is identical to its preceding completion-final package, including linked code.

The earlier low-expansion boundary notes are superseded by the executable mid
block migration. GEMM, attention, copy materialization, and reduction construction
now happen in mid candidate builders; low only projects the resulting program.
See ARCHITECTURE.md and the latest section of REFACTOR_PROGRESS.md for the current
representation and validation. Analytical screening still uses recipe-level
estimates, and general conversion-chain composition remains future work.
