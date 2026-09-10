# Planner memory profiles

Pass `--memory-profile-directory PATH` to `ipu-trivial-test`, or set
`PipelineConfig::memory_profile_directory`. Each planner finalist and the
smallest rejected candidate of an exhausted memory shortlist produces a JSON
report and a standalone HTML report. Filenames include the scope, process ID,
and a serial number; different planner configurations can produce reports with
the same scope. This is opt-in and does not retain a timeline for every search
candidate.

The HTML timeline shows standard, interleaved, and total live storage on the
selected tile. It starts on the tile and step with the largest total usage.
Peak buttons select both the step and tile; the tile selector keeps that tile
fixed while moving through time. “Step’s peak tile” selects the most loaded
tile at the current step. Filter by parameter name or expand an allocation row
to see layouts and aliases. Parameter sequences are grouped by default. The
table separates shard payload from alignment/AMP-tail overhead and shows how
many distinct copies are resident. Scratch is shown separately.

Version 2 JSON contains the same data: `timeline.values` describes allocation roots,
semantic origins, layouts, shard/aligned sizes and copy counts;
`timeline.steps` contains live roots, scratch, memory usage, source operation
and execution count. `tile_shards` records payload before copy multiplicity;
`tile_bytes` records aligned allocation bytes including multiplicity. For each
live root and tile, take the maximum `tile_bytes[tile]` among its aliases, then
add `tile_scratch[tile]` to reconstruct `tile_usage[tile]`. The last two arrays
contain `[standard, interleaved]` pairs. `usage` is the total-peak tile's usage
at that step; `coarse_usage` preserves the former sum-of-maxima comparison.
The old scalar allocation fields remain per-allocation maxima. Executing a
Repeat body multiple times does not multiply scratch. Distinct resident
parameter sequence members do multiply their storage.

These are **per-tile capacity estimates, before address placement**. The
estimator sums simultaneous live allocations using selected ownership and tile
offsets, including uneven shards and AMP alignment/tails. It computes class
peaks independently and total peaks on individual tiles. A later global tile
permutation changes tile labels but preserves these peaks. Scratch remains
estimated: existing reduction/conversion scratch scales with the output shard
and is charged on its owners. This does not diagnose fragmentation or prove
address placement will succeed. Planner finalists precede subsequent fusion
and placement. Capacity screening replaces
the separate exchange-row estimate with the configured package-support reserve;
the report displays both so they are not accidentally added twice. An exhausted
search reports where candidates were rejected, but selecting an input layout
can increase an earlier peak: the timeline identifies that earlier operation.

Search first uses the cheap sum-of-maxima upper bound. If it fits, refinement
is unnecessary; otherwise it computes the per-tile footprint before rejecting
the candidate. Both modes use the same liveness and allocation walker. Reports
always compute per-tile storage, including for candidates accepted by the upper
bound. Their `coarse_usage`/“Old bound” comparison shows whether this refinement
matters for that candidate.

Tests reconstruct each tile at every step across Repeat aliases and padded
GEMMs, check body-only parameter multiplicities, and validate JSON/HTML output.
Ownership tests compare overlapping and disjoint allocations, including wrapped
tile offsets; randomized geometry tests compare per-tile sizes against concrete
shard storage without using the estimator to generate the expected sizes.

Memory profile version 3 reports tensor-only `peak.standard` and `peak.total`.
`peak.exchange_rows` is separate; it remains a ranking estimate rather than a
capacity proof. Older version-2 reports included rows in those two peaks.
