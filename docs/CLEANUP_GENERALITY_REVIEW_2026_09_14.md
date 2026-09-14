# Generality review of September 14 cleanup

Scope: today's 139 commits through `7aacc0e`, with the net diff against
`7359e89` (127 changed files). Reviewed the subsystem diffs and traced the
invariants behind removed checks, derived metadata, shared helpers, and changed
ordering. This is a source review, not a hardware revalidation of every commit.

## Corrections

| Original commit | Change | Finding and correction |
| --- | --- | --- |
| `500fc99` | Order allocator candidates using addresses instead of an explicit standard/interleaved preference | Conflated policy with the current SRAM arrangement. Already reverted by `7aacc0e`. |
| `980efed` | Centralize diagnostic context validation | Actual regression: `clear_tile_exception` bypasses the common context-selection helper. Restored its range check before shifting or writing. Added a fake-BAR test covering all valid contexts and invalid IDs, including oversized shift counts; rejected calls must leave the register unchanged. Fixed in `bc041d7`. |
| `fbfc277` | Calculate maximum linear shard size by querying tile 0 | Correct under today's remainder assignment, but needlessly couples sizing to owner ordering. Restored the balanced-shard maximum formula. |
| `4e3183b` | Replace exhaustive padding-policy match with a rejection check | Correct for the two current variants, but would silently give any new policy zero-padding semantics. Restored the exhaustive match; retained checked rounding. |
| `8bfaeb0` | Calculate profiler payload by subtracting the access tail from allocation size | Correct with today's allocation formula, but makes the profiler depend on its internal arithmetic. Query allocation sizing without access requirements again. |
| `b91cd48` | Flatten aliases with a single grandparent lookup | Correct because today's unions point to smaller IDs. Use the existing root traversal, retaining storage reuse without depending on that union orientation. |

The last four are extension hazards, not demonstrated wrong results for current
models. The exception-clear issue is a present correctness bug.

## Changes retained after checking

- Canonical value/shard and sequence tables: the owning builders establish their
  indexing invariants; the cleanup removes duplicate representations rather than
  assuming a particular tile count or memory arrangement.
- Shared ABI, cast and rearrangement descriptions: these describe implemented
  kernels. Unsupported kernels still fail explicitly; the change does not erase
  a formerly working format path.
- Dependency-depth exchange ordering: dependencies increase depth, so ordering
  by depth retains precedence. This relies on graph structure, not physical
  address ordering or the current topology.
- Interval complement/union helpers and checked alignment: callers retain their
  bounds and alignment constraints. Sharing the arithmetic does not replace the
  allocator's placement policy.
- Default candidate generation: early tile-count deduplication retains the
  existing catalogue. Current candidate families include their active count;
  no supported geometry was removed. Count-independent families would need this
  deduplication contract revisited.
- Profile serialization and binding helpers: checked file extents retain gaps
  and replicas. Shared wire enums and typed payloads preserve the existing
  formats.
- Device assembly macros and quantization extraction: they preserve the actual
  instruction bodies and numerical algorithms. Seven extracted quantization
  functions have identical Python ASTs; tensor scaling delegates the unchanged
  scalar formula. Diagnostic helpers retain their pre-existing C600 scope.

## Validation

- `cargo test -p ipu-driver`: 8 tests passed, including the new invalid-context
  regression test.
- `cargo test -p ipu-codegen --lib`: 303 passed, 5 ignored.
- `git diff --check`: passed.

No hardware run is claimed for this review. The corrections to metadata/policy
boundaries preserve current semantics; the driver change rejects invalid calls.
