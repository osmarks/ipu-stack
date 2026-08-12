# Planner suggestions

This note records suggested next steps for the `ipu-stack` operator planner after
comparing it with the matrix-multiplication path in poplibs. The intended scope
is deliberately narrow: `ipu-stack` compiles a known set of inference workloads,
not arbitrary training graphs or the full range of shapes supported by poplibs.

The current foundations should be retained. GEMM remains a native operation;
layouts remain explicit in the mid-level IR; casts and rearrangements remain
explicit operations; planning remains aware of graph-level layout transitions
and live memory; and final placement remains lifetime- and memory-class-aware.
There is little value in reproducing poplibs' implementation of matrix
multiplication as a degenerate convolution.

## Priorities

The recommended order of work is:

1. Calibrate local kernel and staging costs from hardware measurements.
2. Replace simple beam retention with Pareto-aware retention.
3. Let selected GEMM candidates compare a small number of tile-grid orders.
4. Validate predicted exchange critical paths against captured profiles.
5. Expand the physical-placement search only if measurements show a remaining
   topology-dependent bottleneck.

The first two items are likely to improve plan selection more than a general
topology search.

## Measured kernel-estimator database

The analytical model is useful for extrapolation and for explaining a plan, but
the planner should use measured costs for the retained device kernels wherever
possible. Measurements should describe tile-local work independently of the
whole-device tile grid.

A GEMM measurement key should include at least:

- multiplication and accumulation precision;
- initialize versus accumulate mode;
- small- and large-row specialization, or the exact scheduled row mixture;
- inner block size;
- output-column block size;
- standard versus interleaved weight load;
- direct, repacked, or locally staged weights;
- relevant alignment and access-tail conditions.

The value should include the measured kernel cycles and enough provenance to
detect stale data, such as the device-kernel build identity and target revision.
Measurements should cover the finite set of kernels and shapes that the planner
can actually emit. Interpolation should be conservative; unsupported points
should fall back to the analytical estimator rather than silently borrowing an
unrelated measurement.

Packing and staging helpers should have their own measured entries. In
particular, a standard-to-interleaved population should not be hidden inside a
GEMM measurement because it may be absent, amortized, or repeated depending on
the selected schedule.

The cost composition should remain explicit:

```text
operator cycles =
    measured local kernel cycles
  + modeled exchange critical path
  + measured local packing and staging cycles
  + launch and synchronization costs
```

Keeping topology out of the local-kernel database avoids measuring the same
kernel again for every M-by-N tile grid.

Profile-guided calibration should compare estimates and measurements at the
same boundaries: local kernel calls, local copies, exchange phases, and the
complete operator. Store prediction error as well as the fitted value so that
the planner can prefer a well-characterized plan when two estimates are close.

## Pareto-aware plan retention

The current bounded beam search is valuable because it lets a producer's layout
be selected for later consumers. Ranking every partial graph state by one
scalar cycle estimate, however, can discard a state that is slightly slower so
far but uses less SRAM or avoids an expensive future conversion.

Before applying the beam-width limit, retain only non-dominated states within
each future-visible layout signature. A useful initial state vector is:

```text
(estimated cycles,
 peak standard bytes,
 peak interleaved bytes,
 peak simultaneous bytes,
 standard contiguous-allocation requirement,
 exchange-row bytes)
```

State A dominates state B when A is no worse in every component and strictly
better in at least one. Exact future-visible value formats, automatic-input
status, aliases, and any structured-region equality obligations must remain
part of the signature; they are semantic state, not objective components.

After dominance pruning, cap the remaining frontier to control compile time.
If the frontier exceeds the cap, retain diversity rather than simply taking
the lowest current cycle estimates. For example, keep representatives near the
minimum of each memory component and then fill the remaining slots by cycle
estimate. Deterministic candidate order should continue to break exact ties.

This is not intended to turn planning into a general multi-objective optimizer.
The final objective can still be minimum estimated latency subject to hard SRAM
constraints. Pareto retention merely delays an irreversible decision until
more of the graph is visible.

## Tile-grid order and physical topology

For an output-stationary GEMM distributed across a two-dimensional tile grid,
each tile has an `(m_partition, n_partition)` coordinate. Two simple
linearizations are relevant:

```text
columns-fast: logical_tile = m * n_partitions + n
rows-fast:    logical_tile = n * m_partitions + m
```

This does not change arithmetic work or logical communication volume. It
changes which logical peers occupy nearby physical tiles and which multicast
recipients share an IPU21 exchange-bus pair.

With the current C600 logical-to-physical mapping, consecutive logical tiles
are intentionally assigned to paired hardware positions. Consequently:

- columns-fast placement tends to pair tiles with the same M partition. Those
  tiles consume the same left-activation panel and different weight panels;
- rows-fast placement tends to pair tiles with the same N partition. Those
  tiles consume the same weight panel and different activation panels.

When paired receivers consume the same multicast payload, the shared exchange
path can deliver it more efficiently than two unrelated payloads. Grid order
can therefore change effective receive bandwidth, sender and receiver pressure,
route conflicts, and the completion time of an exchange phase even though the
byte count is unchanged.

The current AMP layouts are effectively columns-fast: column partitions use
the innermost tile coordinate, left shards are replicated across column groups,
and right/output column partitions use stride one. This is a sensible default
and matches the usual choice of optimizing activation broadcast. The existing
cost model also recognizes a special paired-receiver case for streamed K
panels. The missing capability is not topology awareness in general; it is the
ability to compare the default order with one or two relevant alternatives.

### Bounded plan representation

Add a small categorical choice to applicable GEMM plans:

```rust
enum GridOrder {
    ColumnsFast,
    RowsFast,
    // Applicable only to selected activation-stationary schedules.
    InnerFast,
}
```

The order should determine tensor-axis tile strides and the logical tile used
by each dispatch role. It must be represented consistently in layouts,
lowering, exchange generation, and the cost model; it should not be a late
permutation applied after layout selection.

Do not generate every order for every candidate. Useful rules are:

- retain columns-fast as the default for output-stationary plans with resident
  or local weights;
- compare rows-fast when weights are streamed or their phase traffic dominates
  activation traffic;
- compare inner-fast only for an activation-stationary plan when it improves
  weight distribution or the following partial-reduction groups;
- omit variants whose multicast grouping is identical after mapping.

### Topology cost

For each proposed order, form the actual operand multicast groups, map logical
tiles through `c600_logical_to_physical`, and estimate the critical phase using:

- number of physical receiver pairs consuming the same payload;
- number of pairs consuming unrelated payloads;
- maximum send work assigned to one physical pair;
- maximum receive work assigned to one physical pair;
- route/start horizon derived from the same route timing used by exchange
  lowering;
- synchronization and transfer-fragment overhead;
- dependencies between staging, exchange, and the consuming kernel.

Cost the slowest role and dependency chain, not only total bytes. The exact
exchange scheduler remains authoritative; this planner estimate only needs to
rank a small number of alternatives reliably.

The final exchange rows already contain physical routes and timings, so planner
diagnostics should record both the estimated phase horizon and the finalized
one. A persistent systematic error for one order is evidence to improve the
model or measure an additional effect.

### What not to search initially

Do not initially search:

- arbitrary permutations of all tiles;
- arbitrary active-tile subsets;
- every start-tile offset or direction;
- general graph partitioning across disjoint virtual tile groups.

Those choices substantially enlarge the search and are most valuable for
multiple concurrent workloads. A sequential fixed inference workload can use
the current snake mapping and contiguous active prefix unless profiling shows a
specific deficiency.

## Candidate-space guidance for fixed inference workloads

The planner does not need poplibs' complete search over convolution
transformations, training passes, and unusual shapes. Candidate generation can
be driven by the shapes and operators present in the supported models.

Useful retained dimensions include:

- active tile count;
- M-by-N grid factorization;
- output block width supported by an available kernel;
- resident, replicated, or K-sharded parameter storage;
- standard versus interleaved parameter memory;
- direct versus locally staged consumption;
- output-stationary versus the existing activation-stationary reduction;
- the bounded grid orders described above.

Serial K or N panels should be added only when required by SRAM or by a known
workload. Similarly, alternate small-row or narrow-output kernels should be
introduced in response to retained workload shapes and measurements, not to
match poplibs' generality.

## Planning and final placement feedback

Mid-level memory estimation is intentionally conservative, while final
placement has exact linked-code reservations, exchange tables, lifetimes,
aliases, access tails, and contiguous ranges. A valid estimated plan can still
fail exact placement, and an overly conservative estimate can reject a plan
that would fit after lifetime reuse.

Package construction should eventually provide bounded feedback:

1. Plan and lower normally.
2. Attempt exact placement with final package-support reservations.
3. If placement fails, report the precise class, tile, allocation group, size,
   and lifetime conflict.
4. Replan with a constraint or penalty derived from that failure.
5. Stop after a small deterministic retry count.

This should blacklist a concrete infeasible state rather than globally
inflating the memory model. It is lower priority than measured costs and Pareto
retention because the current planner already performs strong memory checks.

## Validation plan

For each supported inference workload, retain a small regression table
containing:

- selected layouts, active tile count, grid dimensions, and grid order;
- estimated local-kernel, staging, exchange, and total cycles;
- measured values for the same components;
- peak standard/interleaved SRAM and finalized placement usage;
- numerical error bounds;
- compile time and number of candidate/frontier states.

Planner changes should be evaluated on both latency and prediction error. A
faster measured result with a worse estimator may be acceptable temporarily,
but the discrepancy should remain visible rather than becoming another hidden
constant. Forced-plan benchmark modes are useful for comparing candidates the
planner did not select and for detecting search-retention mistakes separately
from cost-model mistakes.

## Summary

The most useful lesson from poplibs is not its large general-purpose search
space. It is the separation of legal kernel configurations from a calibrated
cost comparison. For `ipu-stack`, that idea should be combined with the
existing strengths of explicit layouts, graph-level transition planning, and
exact memory placement.

The near-term target is therefore a small, measurable planner:

- hardware-calibrated local costs;
- Pareto-aware retention across layout boundaries;
- at most a few topology-relevant grid orders;
- explicit comparison against finalized schedules and profiles.

That provides most of the likely inference benefit without building a general
replacement for all of poplibs.
