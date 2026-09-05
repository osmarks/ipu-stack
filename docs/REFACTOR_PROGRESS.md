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
