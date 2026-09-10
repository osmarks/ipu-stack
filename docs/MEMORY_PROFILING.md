# Planner memory profiles

Pass `--memory-profile-directory PATH` to `ipu-trivial-test`, or set
`PipelineConfig::memory_profile_directory`. Each planner finalist and the
smallest rejected candidate of an exhausted memory shortlist produces a JSON
report and a standalone HTML report. Filenames include the scope, process ID,
and a serial number; different planner configurations can produce reports with
the same scope. This is opt-in and does not retain a timeline for every search
candidate.

The HTML timeline shows standard, interleaved, and total live storage. Click a
step or a peak button to inspect its allocations, filter by parameter name, or
expand a row to see layouts and aliases. Parameter sequences are grouped by
default. The table separates shard payload from alignment/AMP-tail overhead
and shows how many distinct copies are resident. Scratch is shown separately.

The JSON contains the same data: `timeline.values` describes allocation roots,
semantic origins, layouts, shard/aligned sizes and copy counts;
`timeline.steps` contains live roots, scratch, memory usage, source operation
and execution count. For each live root, take the maximum `bytes` among its
aliases, then add scratch to reconstruct the step's estimate. Executing a
Repeat body multiple times does not multiply scratch. Distinct resident
parameter sequence members do multiply their storage.

These are **planner estimates, not placed per-tile allocations**. They use the
actual estimator's liveness and padding calculations, summing each allocation's
maximum tile shard. Maxima need not occur on the same physical tile. Planner
finalists precede subsequent fusion and placement. Capacity screening replaces
the separate exchange-row estimate with the configured package-support reserve;
the report displays both so they are not accidentally added twice. An exhausted
search reports where candidates were rejected, but selecting an input layout
can increase an earlier peak: the timeline identifies that earlier operation.

Tests reconstruct every step's estimate across Repeat aliases and padded
GEMMs, check body-only parameter multiplicities, and validate JSON/HTML output.
