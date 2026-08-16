# Planner simplification audit

`ipu-codegen` currently contains about 32,500 lines of Rust. The production
parts of `mid.rs` and `low.rs` account for about 12,500 lines. Most of the
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

### Make plan generation operator-directed

Each semantic `OperationKind` should invoke its own typed plan generator. GEMM
generation owns GEMM grids, blocking, parameter placement, and reduction
staging; pointwise generation propagates compatible input layouts and offers
useful ownership transitions; attention generation owns its materialized and
blocked algorithms. A flat cross-operator implementation catalogue obscures
these dependencies and still requires a second dispatch layer.

The generators share an explicit search domain for genuinely global choices:
active tile counts, permitted precisions, weight memory classes, attention
strategy, and diagnostic restrictions. These values are planner inputs rather
than placeholder plans. Shape-dependent layouts and dispatches are emitted
only by the generator for the corresponding high-level operator.

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
2. Factor out the shared search domain and make plan generation
   operator-directed.
3. Consolidate candidate and selected-plan representations.
4. Replace deferred SplitHeads machinery with generic views.
5. Normalize GEMM plans and their lowering.
6. Introduce shared address-independent execution stages.
7. Consolidate Pareto pruning and remove the second package scheduling pass.

Each step should preserve numerical hardware tests and compare canonical MLP
and attention planning/runtime results against the profiles recorded before the
change.
