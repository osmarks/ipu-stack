# Compiler refactor continuation

User priorities (2026-09-05): smaller, clearer compiler; general mid-level
copies/rearrangements and views; sane kernel contracts; fewer layer violations.
Audits are leads, not requirements. Keep existing two-pass exchange scheduling.
Commit tested chunks frequently. Preserve the user-generated callgraph artifacts.

## Validated checkpoint

The first checkpoint includes the accumulated previous refactors and runtime
fixes, plus shared axis-factor view geometry and kernel specialization keys.
Workspace release tests (140 total, including doctests) and strict Clippy pass.
Five hardware workloads pass: 64-tile GEMM and batched GEMM, canonical batch-one
MLP, projected attention, and attention smoke. All five tile images are identical
to the preceding `/tmp/completion-final-*` packages. Logs/packages:
`/tmp/views-kernels-*`; test log `/tmp/views-kernels-workspace.log`.

Keep the configuration replay CCSR readback after every MMIO write. It fixed the
observed startup/run instability. Autoreset has been removed. Completion checking
accepts only the known terminal InvalidProgramCounter at the named completed
symbol with the completion word set; it still rejects other faults. The debug
interface's post-host-exchange visibility is not claimed to be fixed.

## Completed work

1. Kernel responsibilities are now split into ABI, specialization collection,
   geometry, build recipes, and placed-call materialization. Four kernel tests
   pass after extraction; duplicate rearrangement symbol insertion is removed.
2. Physical storage-span geometry now accepts borrowed format/extents, independent
   of LowShard; the low-level adapter checks shard identity. Physical and logical
   spans share one index traversal. All 80 codegen tests and its doctest pass.
   Mid CopyPlan now owns per-destination direct-word/staging decisions, staging
   tensor geometry, and optional transform kernels. CopyOperation<Buffer> owns
   relative contiguous/strided regions; low binds it to LowShardId. Low no longer
   imports the estimator in production. CopyPlan is expanded with concrete shard
   extents during lowering, not yet retained in the selected MidOperation.
   A shared span zipper replaces duplicate copy/count walkers. 142 workspace
   release tests and Clippy pass. Five `/tmp/mid-copy-*` hardware workloads pass;
   tile images remain identical to `/tmp/views-kernels-*`.
3. Graph and mid both have View(AxisFactorView); split_heads is now only a
   rank-three graph helper. Shape and inverse-slice geometry live in graph/view;
   mid/view adapts storage extents. A row-major fallback handles arbitrary axis
   pairs. An end-to-end low-copy interpreter test checks ranks 2–4 and every
   distinct axis pair against an independent forward mapping. Generic reshape/
   permutation composition remains open; attention fast paths are still specific.
   143 workspace release tests and Clippy pass. Both attention hardware cases
   pass with identical tile images (`/tmp/general-view-*`).
4. Attention-stage compilation no longer assumes one configuration or two row
   sizes. Tail softmax assembly receives query/key counts as scalar arguments;
   merge assembly receives query count. Workers specialize only actual constant
   dimensions and share code across block sizes. Full-block C++ softmax retains
   per-query-row specialization. Regression coverage includes three query/key
   sizes and multiple head/value configurations in one build plan.
   142 release workspace tests and Clippy pass (two older view formula tests
   were subsumed by the end-to-end rank-2–5 copy test). Both attention hardware
   workloads pass with unchanged numerical error (`/tmp/attention-contract-*`).
   Linked code ends 152/120 bytes earlier; generated call code grows 12 bytes;
   SRAM reservations are unchanged for projected/smoke attention respectively.

## Subsequent simplifications

- ABI now stores an input count and static typed scalar slice; fixed register
  constants are shared with emit_compute. TileKernel's single Planned wrapper
  and ViewSlice's single-field wrapper are removed. Replica intersection grouping
  and selection are shared between cached/uncached gathering and conversions.
  142 workspace release tests and strict Clippy pass for this chunk. This removes
  83 Rust lines without adding tests; existing numerical paths are unchanged.
- Operator and conversion kernels now share StorageRequirements. Low binds its
  formats/arity to actual kernel buffers before metadata interning, preserving
  access requirements and applicable separation constraints. The prior attention
  metadata incorrectly carried the enclosing operator's three input formats into
  one/two-input kernels. Existing randomized deferred-view coverage failed under
  the new arity check before the fix (`/tmp/operand-constraints-before.log`), then
  passed with checks against each actual buffer. All 142 workspace release tests
  pass (`/tmp/storage-requirements-workspace.log`); hardware recheck follows the
  common graph-builder cleanup.
- ComputeGraph and RegionBuilder now expand one inherent operation-building
  API from graph/builder.rs. No trait imports or call-site API changes are needed.
  142 workspace release tests and Clippy pass for this extraction.
- Parameter ownership rotations are now recorded in MidValue, and MidGraph owns
  the target tile count. Low follows this decision and takes only the diagnostic
  checkpoint flag instead of the entire PipelineConfig. Rotation scoring checks
  only affected tiles without cloning load arrays; randomized tests compare its
  exact choices with the old algorithm. 143 workspace release tests and Clippy
  pass. All five hardware workloads pass (`/tmp/ownership-*`); tile images are
  byte-identical to their preceding checkpoints, including the operand-constraint
  and common graph-builder changes.
- Kernel object recipes are now separated by GEMM, rearrangement, and attention
  families. The shared build orchestrator is 135 lines. Three supervisor wrapper
  files are replaced by worker_call.S, with argument-register order declared in
  each recipe. Five hardware cases pass (`/tmp/worker-recipes-*`), with identical
  tile image bytes. Block-major worker symbols now include both block dimensions;
  regression coverage checks that multiple layouts coexist without object/symbol
  collisions. Attention inventories keep normalized specialization keys in a set,
  avoiding kernel clones and repeated linear duplicate scans. ABI needs only one
  generic specialized marker. 144 release workspace tests and Clippy pass; both
  attention cases pass again after the inventory change (`/tmp/kernel-final-*`).
- Local and inter-tile copies now share mid's CopyOrder. One local-copy binder
  replaces the separate logical/physical helpers and duplicate call-site branches,
  removing 48 Rust lines. All 144 workspace release tests and Clippy pass. Five
  final hardware workloads pass (`/tmp/copy-order-*`) and have byte-identical tile
  images to `/tmp/ownership-*`. This also validates the final kernel recipe split.
- CopyPlan is still expanded with shard geometry during lowering. Do not claim
  full materialization recipes are retained in MidOperation yet.


No subagents were used. Work is on `refactor/compiler-views-kernels`.

## Remaining design boundaries

The inexpensive consolidation opportunities found in this pass are implemented.
General permutation/view composition still needs a traversal representation that
preserves reordered coordinates; simply zipping canonical source/destination
spans would be incorrect. Retaining concrete copy recipes before low expansion
also needs a common representation for operator-created staging buffers. Avoid
adding a second parallel schedule only to move policy calls earlier. Flexible
kernel output layouts and shared GEMM execution/cost stages remain substantial
extensions, rather than mechanical cleanup. These are useful starting points for
a subsequent design pass, not claims that the compiler is fully generalized.

## Executable mid blocks (2026-09-05, subsequent request)

User explicitly requested GEMM block/reduction/copy expansion in mid, removal of
old pathways, and ideally the same migration for attention. Implemented the
boundary change: mid/implementation builds a whole-device BlockRegion containing
explicit Compute, Copy, Exchange, Repeat, and Checkpoint operations. BlockValue
is shared by canonical tensor shards, partials, and staging outputs; no separate
temporary-value representation. GEMM/attention/conversion expansion files moved
out of low and now populate that region directly. Low is only a per-tile
projection, shares the immutable MidProgram through Arc, and has no whole-GEMM or
attention dispatch. Repeats retain one shared mid body, projected recursively.

The analytical beam still uses implementation recipes as candidate-builder
inputs (ImplementationCandidate, formerly MidGraph). Production lower_finalists
constructs executable MidPrograms before scheduled finalist selection; those
programs do not retain the opaque operation plans. Logical-value metadata and
checkpoint boundaries remain for diagnostics. Planner unit tests inspect the
builder inputs; execution tests explicitly build the mid program before low.

144 workspace release tests and Clippy pass (`/tmp/mid-program-*`). Five hardware
cases pass (`/tmp/mid-blocks-*`), with byte-identical tile images to the preceding
`/tmp/copy-order-*` packages. The subsequent Arc-sharing cleanup passes the same
unit checks and changes ownership only. No device code changed.

Next bounded step: extract the parallel-reduction builder from GEMM into a
reusable mid sum builder over ordinary block views, deriving staging from the
number of contributors rather than GEMM grid metadata. Add direct mid-region
validation/coverage and repeat hardware checks after substantive changes.

Reusable sum construction and mid copy cleanup are implemented. The sum builder
accepts groups of ordinary block views, derives its complete/streamed stages
from each group's contributor count, and handles one-contributor groups as
copies. Its interpreter regression checks independent groups of one, two, and
four contributors without invoking GEMM or low projection. Contributors must
have matching coordinates and storage order; callers use explicit rearranges
for other formats. Existing GEMM access requirements are retained for now.

mid/passes merges adjacent contiguous copies on each tile, respecting compute,
exchange, repeat, and checkpoint boundaries and avoiding aliasing allocations.
It compacts the copy arena afterward. This is contiguous-run merging, not yet
arbitrary composition/elimination of multi-operation layout conversions.

Builder methods and files now use construction terminology. Buffer/view helpers,
repeat handling, pointwise operations, block emission, three GEMM strategies,
attention panel/blocked/materialized paths, deferred views, and mapping geometry
are separated. Production implementation files are at most 590 lines (tests
remain consolidated). 146 workspace release tests and Clippy pass in
`/tmp/mid-modular-*`. Five `/tmp/mid-optimized-*` hardware runs pass and remain
byte-identical to `/tmp/mid-blocks-*`; the subsequent method/file extraction
changes no emitted work and passes the same unit checks.

Final API cleanup: low projection is infallible for a constructed MidProgram and
imports only executable-block types. Repeat arena identities belong to low;
boxed mid repeat payloads keep ordinary block-operation entries compact.
Packaging and diagnostics use low's shared program instead of passing another
copy of the mid handle through every API. ARCHITECTURE.md now describes the
current boundary and distinguishes transient recipes, executable blocks, and
per-tile projection.

Final checks: 146 workspace release tests and strict Clippy pass in
`/tmp/mid-complete-*`. Five `/tmp/mid-final-*` hardware workloads pass and have
byte-identical images to the pre-migration `/tmp/copy-order-*` packages. A
structured two-block MLP repeat passes (`/tmp/mid-extra-repeat.*`, maximum error
0.000046). Both attention strategies pass when forced on projected attention:
`/tmp/mid-strategy-flash.*` max error 0.001230 and
`/tmp/mid-strategy-materialized.*` max error 0.000930. Forced attention smoke
also passes for both. After the final packaging cleanup, the rebuilt GEMM smoke
passes with 262144 exact checks and an identical tile image
(`/tmp/mid-complete-gemm.*`). No runtime/device/kernel-module changes were needed.

The requested block-IR migration is complete: low has no GEMM/attention/conversion
expansion path. Further work can price expanded blocks directly or compose more
general layout-conversion chains; neither is claimed by the contiguous-copy pass.

## Primitive call contracts (2026-09-05)

- Kernel calls now derive their contracts from actual operand formats and kernel
  kind. Removed inherited contract truncation, format rewriting, separation-group
  pruning, and unused requirement arguments in attention, pointwise and sums.
- Executable contracts omit candidate aliasing/materialization/staging policy.
  GEMM read tails and output/left SRAM separation apply to GEMMs inside attention;
  reduction and softmax calls no longer inherit enclosing GEMM constraints.
- Release workspace tests: 146 passed. Strict Clippy passed. Hardware GEMM,
  batched GEMM, SigLIP MLP, SigLIP attention and attention smoke all passed.
  Logs: `/tmp/kernel-contract-*`.
- Remaining authorized work: shared panel materialization and mid-derived cost /
  allocation analysis used by detailed planning, removing obsolete estimates.
- Deferred at the user's request: copy/view chain composition (#4). Remind the
  user later to discuss its complexity and interaction with planning before
  implementing it. Adjacent-copy behavior remains as before.

## Shared destination materialization (2026-09-05)

- Added one materialization batch at an exchange boundary: destination copy plans
  select staging and padding, populate local/remote slices, and emit final copies
  or transforms. GEMM operand slices, deferred attention panels, eager conversions
  and views use this construction. Ownership, panel size and reuse stay with the
  implementation strategy.
- Removed attention query-receive and per-panel row-major buffer bookkeeping,
  separate attention rearrangement emission and duplicated word-exchange preflight.
  Removed the now-unused CopyPlan direct-word flag and deferred gather emitter.
- Net source reduction for this chunk: 187 lines (138 added, 325 removed).
- Release tests: 146 passed; strict Clippy passed. Hardware: standard five cases,
  forced blocked SigLIP attention, and two-block repeated MLP all passed. Logs:
  `/tmp/shared-materialization-*`. The default SigLIP case covers materialized
  attention; forced flash max error .001230, materialized .000930, repeat .000046.
- Next: replace operator-shaped cost reconstruction with executable-mid pricing
  and shared allocation analysis in detailed planning. Copy/view chain composition
  remains deferred for later discussion with the user.

## Executable-mid costing and allocation analysis (2026-09-05, in validation)

- Deleted operator-shaped GEMM/attention/deferred-view cycle, scratch and traffic
  reconstruction. Primitive prices consume actual calls/copies/exchanges. Shared
  call geometry lives in `mid/call`; shared allocation analysis includes alias
  groups, access tails, element rounding, repeats and lifetimes.
- Detailed branch scores use executable region timelines, including deferred
  consumer movement. Operator fragments are retained and rebound with explicit
  ownership/format/capacity checks. Cache retention is bounded; expanded branches
  are pruned between parents as well as at the end of an operation.
- Exchange pricing and encoding share semantic/physical span selection. Scheduled
  phase prices use repeat execution counts, while row storage stays static.
- Updated curated data-flow graphs and architecture. Copy/view chain composition
  (#4) remains deferred for discussion with the user.
- Validation so far: 144 release workspace tests and strict Clippy passed before
  the final retention tuning; retained-vs-fresh program equality and scheduled
  repeat pricing regressions pass. GEMM and batched GEMM pass on hardware. The
  first full-size MLP attempts exposed excessive planning time/host memory and
  were stopped during planning. Further performance and hardware validation is
  required before considering #1 complete. Logs: `/tmp/concrete-mid-*`,
  `/tmp/bounded-mid-*`, `/tmp/scheduled-repeat-test.log`.

Planning retention follow-up:

- Fixed the candidate-shortlist width bug when diversity representatives exceeded
  the requested width. Added a regression. Screening does not discard candidates
  when the entire set already fits the shortlist.
- Cheap boundary costs shortlist branches before executable-fragment construction;
  detailed ranking and memory feasibility still use complete emitted regions.
  A shared eight-worker pool bounds simultaneous construction on large hosts.
- Replaced speculative strong-cache retention/eviction with weak references to
  fragments owned by surviving branches. Removed `MemoryEstimate`, redundant
  operator exchange summaries and `ImplementationEstimate`; allocation analysis
  now computes only the region peaks actually consumed by planning.
- All 146 release workspace tests and strict Clippy pass (`/tmp/lazy-fragment-*`).
  GEMM and batched GEMM hardware pass at about one second end-to-end and ~100 MiB
  peak host RSS (`/tmp/concrete-mid-v8-*`). Full-size MLP/attention performance and
  hardware validation remain in progress; do not claim them complete yet.

Planning traversal checkpoint:

- Profiled full-size MLP. Lifetime analysis scanned every device-wide transfer
  for every tile. It now indexes touched blocks by tile once. Output lifetimes
  also follow actual block ownership instead of assuming tile-ordered outputs.
- The next profile was dominated by element-by-element storage span traversal.
  Shared span enumeration now walks contiguous storage lanes, retaining canonical
  order for semantic transfers and permitting row-fast traversal for physical
  transfers. Random partial-view tests cover every storage order and precision,
  including nonzero shard origins, against an element-by-element oracle.
- Avoid rebuilding unchanged final branch analyses; screen at twice beam width.
- All 148 release workspace tests and strict Clippy pass (`/tmp/lane-*`). All
  seven device cases passed after lifetime indexing (`/tmp/indexed-mid-*`),
  including full MLP, materialized/Flash attention and repeated MLP. MLP took
  440 seconds and attention 215 seconds before the storage traversal fix.
  Final traversal performance/device checks are running (`/tmp/lane-mid-*`).

Concrete exchange-cost correction:

- Final traversal run passed GEMM, batched GEMM and full MLP. MLP mid planning
  fell to 66.7 seconds, but total build/run remained 365 seconds because the
  chosen layout produces far more physical exchange fragments than the old
  analytical model's choice. Remaining attention cases were stopped in planning
  to test the following scoring correction; they are not recorded as passed.
- Detailed exchange cycle prices now include the existing materialization
  fragment-event calibration, taking the maximum of bandwidth and event costs.
  Previously actual fragments affected table storage only, making scattered
  transfers too cheap in detailed ranking. Scheduled phase prices still replace
  the approximation. The repeat regression now verifies equal-payload fragmented
  exchanges cost more and scheduled prices override that difference.
- All 148 release tests and strict Clippy passed after the pricing change; the
  extended regression and Clippy passed afterward. Final eight-case device run
  (including scheduled-finalist repeated MLP) is `/tmp/fragment-priced-*`.

Final validation record (2026-09-05, complete):

- Combined #1–#3 source size: 52,655 -> 51,545 lines, **1,110 fewer lines**.
  Count includes comments/tests in tracked `crates/` and `device/` Rust, C++ and
  assembly sources, compared with `531f1dc`; moves are not counted as deletions.
- Workspace release tests: 148 passing. Strict workspace/all-target Clippy:
  passing. Extended fragmentation/scheduled-repeat regression: passing.
- Final MLP passed numerically (maximum absolute error 0.011719). Mid planning
  took 64.8 seconds; total package build/device validation took 298.6 seconds.
  Earlier byte-only scoring with optimized traversal took 365.1 seconds total.
- **Remaining performance limitation:** pre-cost-refactor MLP mid planning was
  24.9 seconds, and its exchange schedules were much cheaper to construct.
  Final MLP peak host RSS is approximately 8.94 GiB. The bounded concrete search
  remains more expensive and selects different layouts. Fragment calibration
  improves the score but does not eliminate this regression. Future tuning must
  consider shortlist diversity, fragment pricing and scheduler construction
  together; reducing the shortlist blindly can discard good layouts.
- General copy/view chain composition (#4) remains deferred. Discuss its overlap
  with planning before implementing it.
- GEMM, batched GEMM, full MLP, automatic attention, attention smoke, forced Flash,
  repeated MLP and repeated MLP with `--exchange-schedule-finalists 3` all passed
  on the final binary. The repeated workload retained only one finalist, so that
  flag did not exercise multi-finalist reranking; scheduled repeat multiplicity
  and override behavior are covered by the focused unit regression.
- Forced materialized attention also passed on the final binary: 839,808
  numerical checks, maximum error 0.000930. Automatic/forced Flash attention
  maximum error was 0.001233; repeated MLP maximum error was 0.000046.
  All nine final device invocations passed without startup/runtime failures.
- Updated curated graphs in `docs/COMPILER_DATA_FLOW.md`. #1–#3 are implemented
  and committed; compilation performance remains the explicit limitation above.

Profiled hardware measurement (2026-09-05):

- Rebuilt the same selected full MLP plan with profiling enabled: batch 1,
  tokens 729, dimension 1152, hidden dimension 4304, one block, no biases.
  Hardware passed, maximum absolute error 0.011719.
- Measured maximum tile start-to-end duration: **336,318 cycles**; minimum
  263,682 cycles. At the benchmark's configured 1.5 GHz this is 224.212 us,
  64.484 effective GEMM TFLOP/s. Analytical estimate was 549,747 cycles.
  Artifacts: `/tmp/fragment-priced-mlp-profile.{log,json,ipuexe}`.
- A five-second CPU sample during physical scheduling attributed 63.9% to
  `BinaryHeap<ReadyTransfer>::pop`, 3.4% to its push, and 4.6% to
  `earliest_transfer_offset_impl`. This is a short sample, not an attribution
  across the whole build. `/tmp/final-mlp-scheduler-{perf.data,report.txt}`.

Whole-device mid rewrite, representation checkpoint (in progress):

- Renamed the selected whole-device recipe `MidProgram` and the expanded
  per-tile buffers/calls `TileGraph`. Moved expanded IR, access contracts, copy
  realization and tile builders under `low`.
- This checkpoint preserves behavior; it is not the completed boundary rewrite.
  Beam costing still expands tile graphs and will be replaced by compact mid
  primitives and geometry-based estimates. Operator decomposition must become
  explicit before generic tile expansion, not merely move behind another name.
- All 148 release workspace tests pass after the mechanical separation.

Whole-device primitive checkpoint (2026-09-05):

- Mid now contains distributed tensor copies/mapped windows, selected kernel
  grids, explicit partial-axis sums and structured repeats. Final selection
  resolves recipes before low tile expansion. Removed the old tile-level GEMM,
  attention, streamed-conversion and fragment-remapping builders.
- Beam costing traverses compact primitives and shares pure kernel geometry
  prices with final timelines. It no longer builds tile graphs or invokes
  physical allocation analysis. Exchange and memory estimates are coarser.
- Selected deferred views become mapped copies into ordinary consumer tensors;
  resident operand views and accumulating/in-place versions remain supported.
- All 148 workspace release tests pass. Regression coverage includes compact
  size independent of tile count, fresh/cached implementation equivalence,
  explicit partial dimensions and resolved attention view windows.
- GEMM and batched GEMM pass on hardware. Full hardware suite and performance
  measurements are still in progress at this checkpoint.
- Attention keys and values are currently separate materializations, so there
  may be two exchange phases per key block. This is explicit in mid and costing;
  the former test bound assumed their combined tile-builder batch.

Boundary enforcement and uneven-grid correction (2026-09-05):

- Low expansion rejects unresolved operator recipes; test-only helpers explicitly
  resolve them in mid. Split mid decomposition into shared construction, GEMM
  and attention modules.
- Full MLP initially failed numerically. Diagnostic execution localized the
  failure to its first GEMM. New partial tensors had reconstructed unpadded row
  partitions instead of inheriting the left operand's padded compute rows.
  Fixed the mid geometry and added a full uneven-MLP coordinate regression;
  low also rejects mismatched operand/output matrix bounds.
- Added offline physical-exchange coverage for the 64-tile GEMM smoke shape and
  a retile coordinate regression. All 151 release workspace tests and strict
  Clippy pass. Hardware revalidation, including profiled MLP, remains in progress.

Physical-copy padding correction (2026-09-05):

- The row correction reduced the MLP mismatch substantially but exposed an
  independent unwritten-tail bug: physical materialization could leave newly
  padded K storage uninitialized. Copy realization now zeroes destination
  storage not covered by mappings, including any destination staging buffer.
- The regression covers both declared padding and a mapped view whose logical
  allocation is wider than its source. Workspace tests passed before adding
  this focused regression; the added regression and strict Clippy also pass
  (152 tests combined). Full profiled MLP and attention hardware validation
  are still pending at this checkpoint.

Explicit attention panel distribution (2026-09-05):

- Restored the old algorithm's distributed packing followed by broadcast as
  ordinary mid copies. Directly transforming at every consumer had produced
  excessive exchange tables and failed package construction. Low still has no
  attention strategy code. Intermediate kernel precision is explicit.
- Physical copies retain shared source/destination padding. This is necessary
  for word-aligned broadcast of the final 25-row key block in the 729-token
  attention benchmark. Uncovered padding is still initialized separately.
- All 152 workspace release tests and strict Clippy pass. Automatic attention,
  forced Flash, attention smoke, repeated MLP, and scheduled-repeat MLP pass on
  hardware. Automatic attention build/validation took 38.73 seconds; forced
  Flash took 39.27 seconds. Forced materialized validation remains pending.
- The earlier profiled MLP passed at 332,640 cycles, maximum error 0.011719,
  8.15 seconds mid planning and 233.32 seconds build/validation, about 1.28 GiB
  peak RSS. A final run will include the shared-padding change above.

Materialized attention layout correction (2026-09-05):

- Its value matrix now uses a row block equal to the full padded key count,
  matching the single PV product. The 64-row Flash layout is valid only for
  Flash's 64-row products. A focused mid-layout regression covers 73 logical
  key rows padded to a 128-row product.
- All 153 release workspace tests and strict Clippy pass. Forced materialized
  hardware validation is running at this checkpoint.

Completed whole-device mid rewrite (2026-09-05):

- Final selection contains compact distributed primitives; low rejects unresolved
  operator recipes and only expands selected primitives. Beam costing uses mid
  geometry, approximate traffic and coarse liveness without constructing tile
  graphs or running physical allocation analysis.
- Attention packs complete distributed K/V tensors before its key-block sequence.
  Flash broadcasts K/V together; materialized attention separates their resident
  lifetimes. These are ordinary mid copies, with no attention strategy in low.
  Parallel GEMM also preserves the selected MatchRemote materialization policy.
- Final validation: all 153 workspace release tests and strict Clippy pass. Hardware
  passes GEMM, batched GEMM, full MLP, automatic attention, attention smoke, forced
  Flash, forced materialized attention, repeated MLP and scheduled-repeat MLP.
  The scheduled-repeat run retained only one finalist, so it does not demonstrate
  hardware reranking of multiple finalists; the pricing/multiplicity unit test
  covers that logic. No driver or reset workaround changes were needed.
- Final full MLP (729 tokens, width 1152, hidden 4304) measures 327,144 cycles at
  1.5 GHz, versus 336,318 before this rewrite: about 2.7% faster. Maximum error
  is 0.011719. Mid planning falls from 64.8 s to 7.181 s; total build/validation
  falls from 298.6 s to 244.67 s. Peak RSS falls from about 8.94 GiB to 1.31 GiB.
- Final projected attention (16 heads, 729 tokens, width 72) measures 782,088 cycles
  versus 739,284 before this rewrite: a remaining 5.8% device regression.
  Upfront packing removed most of the initial regression (1,226,070 cycles).
  Automatic build/validation takes 49.62 s, with 2.495 s mid planning versus
  32.673 s previously. Maximum error is 0.001230; forced materialized attention
  also passes with maximum error 0.000930. Final attention has 21 exchange phases
  versus 19 previously; further scheduling/transfer work remains possible.
- Source size is 50,491 lines, down 1,054 from the 51,545-line baseline `3dc35a8`.
  Counts include tracked Rust/C/C++/assembly sources and headers under `crates`
  and `device`, excluding documentation and generated artifacts.
- These final measurements supersede the intermediate validation checkpoints
  above. Logs and packages are under `/tmp/whole-mid-*`; architecture and data-flow
  documentation describe the completed boundary. General copy/view chain
  composition (#4) remains deferred pending discussion of its planning overlap.


## Execution-cost shortlist and historical timing diagnosis

Replaced boundary-memory screening with compact mid execution prices in both
operator shortlisting and preliminary beam ranking. Preserved reduction geometry
diversity and priced fragmented packed-linear movement more conservatively.
Search-scoped strong caching now makes sense because implementations are compact
whole-device regions; this supersedes the earlier weak-cache/boundary-screening
checkpoint above. Fixed in-place reuse for multiple linear shards on one tile.

Hardware: automatic MLP 327,144 -> 229,314 maximum tile cycles; projected attention
782,088 -> 523,980 profile cycles. Full validation, cache/input-replication
tradeoffs and the historical padding/redistribution regression are recorded in
[the profile diagnosis](PROFILE_LAYOUT_DIAGNOSIS.md#execution-cost-shortlisting-2026-09-05).
Q/K/V batching remains on its separate experiment branch.
