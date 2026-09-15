# Compiler refactor: growth and representation ledger

Audited on 2026-09-15, comparing **`1501698` → `b3b1bcd1`**.
This is an accounting record, not a declaration that the refactor is complete or that every addition is justified.

## Totals and scope

| Point | Non-test implementation lines | Relative to baseline |
| --- | ---: | ---: |
| Before structural refactor, `1501698` | 48,402 | 0 |
| Before growth cleanup, `9756530f` | 51,070 | +2,668 |
| Current implementation, `b3b1bcd1` | 49,987 | **+1,585** |

- Growth-producing commits added **3,308** lines net at their respective commit boundaries.
- Other pre-cleanup commits removed **640**, leaving **+2,668** at the peak.
- Seven subsequent cleanup commits removed **1,083**, leaving **+1,585**.
- The five named-work/scoped-choice commits alone added **1,230** before cleanup. They expanded the decision representation and its supporting machinery, rather than merely relocating existing code.

The same AST-based counter excludes comments, `cfg(test)` items, test/bench files and the `ipu-tests` benchmark crate. It includes Rust and device assembly/C++/header/definition sources; it does not count documentation or HTML/JS. These are the previously reported non-test figures, not raw `git diff --stat` lines.
File moves contribute zero when their contents are unchanged. The columns below are **historical commit deltas**, not independent estimates of how much each feature costs today: subsequent commits touch overlapping code. Summing only growth rows would overstate the current increase.

## Every commit with positive net growth

| Commit | Net lines at that commit | Change and representation cost |
| --- | ---: | --- |
| `87b2e7d7` | +45 | **Make the low graph authoritative for padding removal and execution.** Low graph traversal/retention and shared Repeat binding. `requires_finite_scratch` moved from the projection to `TileGraph`; this field was not newly invented. |
| `d80f5adb` | +16 | **Separate exchange scheduling policy from cache ownership.** Explicit policy plumbing and replay checks. `stream_words` moved from `ExchangeScheduleCache` into each `ScheduleRecipe`. |
| `eb8b1442` | +26 | **Give family construction ownership of executable fragment caching.** A separate `FragmentCache` with `entries`, plus key/entry aliases and explicit plumbing. Replaced the implementation cache inside `MemoizedCostModel`. |
| `5a4b1a0a` | +167 | **Give products explicit mid semantics and GEMM-owned expansion.** `Product` records precision, mode, block sizes, axes, windows and aliases. Product validation and local matrix splitting grew. `ProductAxes` and `OperandWindow` already existed. |
| `6ceef8f3` | +5 | **Separate cast access geometry from donation profitability.** Cast access geometry moved to the kernel family. `CastChunks { axis, ranges }` already existed; this was not a new representation. |
| `675d4542` | +55 | **Move exchange grouping out of low construction.** Standalone exchange/copy-motion pass, phase-ID compaction and remapping. No retained new data type from this slice. |
| `6b6944c1` | +84 | **Separate copy geometry from destination packing policy.** `PackingPolicy::{Automatic, Direct, Staged}` and the Copy `packing` field; separate population geometry and selection. Original cache/geometry types were subsequently consolidated. |
| `6768a9ad` | +8 | **Bind every kernel result through indexed output contracts.** Indexed `MemoryOperand::Output(u16)` and one `outputs` list in runs/requirements, replacing primary/additional output fields. The code grew despite fewer split fields. |
| `21158ca1` | +106 | **Declare distributed operand indexing and share broadcast geometry.** `OperandIndexing::{Elementwise { result }, Local(window)}`, `Broadcast` and `BroadcastError`; explicit indexing validation and projection. |
| `dbdf25ea` | +54 | **Bind final kernel views before low construction.** Early complete-view binding, physical access checks and `KernelError` replacing `KernelMaterializationError`. Interned run metadata already existed. |
| `dc76daa3` | +43 | **Resume checkpoints after redundant graph registry removal.** Checkpoint migration for a changed graph registry. This migration is now removed. |
| `487ea06d` | +123 | **Make compiler search and physical evaluation explicit.** Explicit compiler orchestration; `EvaluatedCandidate`, `PackageSupport`, `AddressProposal`, and a shortlist record. Replaced callback return state, `ScheduledPlan` and `BuiltApplication`; not all fields were additional. |
| `5b9fcbd0` | +436 | **Separate planner construction from executable mid binding.** Planner/mid separation, fragment substitution and validation. Most files moved, but construction/binding scaffolding and argument plumbing produced a net +436. Includes `ValueBuilder`, `FragmentBuilder`, separate program errors and family/dispatch integration. Later cleanup removed output preallocation, storage wrappers, discarded validation errors and duplicated boundaries. |
| `fa792ca6` | +166 | **Represent mid ownership with shared tile embeddings.** `OwnerMap { rotation, embedding }`; `MidValue.owners` replaced `tile_offset`. Added arbitrary embeddings and the validation/remapping needed throughout lowering and costing. |
| `120a4372` | +132 | **Separate IPU21 architecture from loader and runtime contracts.** New `ipu-target` crate and shared definition inputs/loader ABI. Most constants and `Topology` moved; `TopologyError`, `InstructionError` and error-conversion arms were added. |
| `f654f00b` | +192 | **Retain exchange relocation sites through row encoding.** `EncodedRow`, `RowBuffer`, `SendAddress`, `OutgoingBaseWrite`; retained send IDs on scheduled/prepared/staged transfers and a register operand on outgoing-base events. Replaced production instruction re-decoding with stored metadata. |
| `c8e655a4` | +257 | **Name family work for stable recipe replay.** `LocalSite`, `WorkSite`, `MidOperation.site`; naming in constructors/rewrites, availability checks and checkpoint handling. `WorkSite` replaced traversal-number `CastSite`; naming and validation remain, migrations do not. |
| `6a89697b` | +195 | **Scope cast-storage choices to families and work sites.** `CastStorage`, `CastStoragePolicy { default, operators, sites }`, `Recipe.cast_storage`, `Candidate.cast_storage_sites`; override resolution, proposals and validation. Replaced the global `in_place_casts` option. |
| `e040adf8` | +323 | **Move ownership choices and mapping search into recipes.** `OwnerChoices { inputs, operators, results }`, `ResultSite`, `Recipe.owners`, `RecipeProposal { recipe, estimated_cycles }`, and retained traffic multiplicities. Adds home assignment/conflict checks and explicit operand-copy insertion; removes the separate global mapping optimizer. |
| `96545d07` | +191 | **Scope panel packing to named copies and workspaces.** `PanelPacking { rows, workspace }`, per-site `Recipe.packing` and `Candidate.packing_choices`; workspace remapping and unavailable-choice checks. Replaced global `packing_rows`. |
| `454155c9` | +264 | **Separate named work grouping from result ownership.** `ReductionGroup { members }`, `GroupProposal { reductions, homes }`, `Recipe.reduction_groups`, `Candidate.grouping_choices`; discovery, request validation and reordering. Replaced global parallel-reduction/disjoint-copy-source switches. |
| `15ce9043` | +105 | **Represent attention state as typed compute results.** `RowWorkspace { precision, leading, trailing }` and explicit statistics/worker-workspace result values and pointers. Mixed attention storage became typed outputs; `AttentionMerge.key_block_columns` was removed. |
| `22c5a08e` | +28 | **Retain concrete views in low value bindings.** `TileGraphBuilder.logical_values`, concrete `bindings`/`value_views`, and `DiagnosticShard.view`. Replaced canonical shard-ID bindings plus the separate borrowed-view repair map. |
| `7955237e` | +47 | **Centralize storage binding and alias-aware copy checks.** `BoundView { shard, extents, backing }`, `StorageAccess { alignment, access_tail_bytes }`, `AddressError`; shared root/origin/access binding. `StorageAccess` replaces placement `Requirement` and fields formerly embedded in `KernelAccess`. |
| `6b41664a` | +136 | **Bind local copy helpers before they enter low.** `CopyKernel` and `CopyRun { movement, kernel, access }`; low stores checked copies. Replaces late helper selection but adds retained binding/access state and rebinding during coalescing. |
| `9756530f` | +104 | **Measure warm expansion and retained geometry cache payload.** Warm expansion measurements, process-memory snapshots and retained-cache-size accounting. `ExpansionBenchmark.warm`, `ExpansionTiming.process_memory`; cache statistics have since moved to `GeometryCacheStats`/`CacheStats`. |

## Offsetting reductions

These are needed to reconcile the growth table with the current total. A net-negative change can still introduce a type; those representations are listed below.

| Commit | Net lines | Change |
| --- | ---: | --- |
| `a97e4119` | -60 | Derive Repeat sequence bindings from complete storage placement |
| `ab9c778b` | -118 | Construct executable mid directly from selected families |
| `2210c522` | -88 | Unify executable mid copies, casts and distributed compute |
| `3cd756dc` | -10 | Give tensor geometry a neutral owner and separate layout preferences |
| `a7405222` | -102 | Bind kernel inputs directly to shard views |
| `f8958170` | -181 | Construct complete kernel calls within their families |
| `14b5012d` | -18 | Separate fragment working ownership from result homes |
| `125375bb` | -63 | Use the ordinary call ABI for halfword copies |
| `cb572e07` | -361 | Remove obsolete search checkpoint migrations and legacy choices |
| `f1976091` | -206 | Share movement geometry between lowering and costing |
| `c144c369` | -109 | Remove unused cost annotations from executable mid operations |
| `677ba620` | -57 | Return fragment results directly instead of preallocating output bindings |
| `d6dfa3f3` | -192 | Remove redundant planner storage wrappers and duplicated local kernel selection |
| `7c04210c` | -120 | Check planner feasibility directly on resolved layouts |
| `b3b1bcd1` | -38 | Construct costed and selected operator boundaries through one emitter |

## What the accounting does and does not establish

1. **The scoped-choice work is a major source of complexity.** Choices now live in several maps/sets, with related discovery lists in `Candidate`, checks during construction, explicit clearing in proposals, remapping and serialization. These are real extra responsibilities. Their existence does not demonstrate that this organization is the simplest way to support scoped choices.
2. **The planner/mid split did not pay for itself in source size.** Its +436 commit included substantial file movement, but the net addition was real. The general fragment binder, construction state and validation were added while other construction scaffolding remained. Subsequent cleanup removed some of that duplication.
3. **The driver extraction also grew.** `PackageSupport` has fifteen fields; `EvaluatedCandidate` has eight. Several collect previously scattered locals or replace old records, but this is still a larger explicit interface that deserves review.
4. **Some additions encode required information more directly.** Product semantics, operand indexing, relocation sites and concrete backing views replace inference or reconstruction. That can be useful without being a net simplification; the table does not excuse their implementation size.
5. **Tests establish only the properties they exercise.** Passing tests and byte-identical checked packages do not establish that the new structure is comprehensible, minimal or properly factored. The proposal's accumulated implementation paragraph did not provide that assessment.

## Current types introduced or renamed during the refactor

This inventory is against the baseline, with fields from the current tree. It includes private helper records and type aliases so that small wrappers are visible. **New names are not a net type count**: the replacement notes identify older representations where applicable. Existing types such as `OperandWindow`, `ProductAxes`, `CastChunks`, `CoordinateMapping`, `KernelInventory`, `KernelRunMetadata` and `ProcessMemory` are not newly introduced types.

- **[EvaluatedCandidate](../crates/ipu-codegen/src/compile.rs)** (struct): `program: LowProgram`; `placement: crate::Placement`; `exchanges: crate::exchange::LoweredExchanges`; `application: ipu_package::Application`; `support_memory: TileMemoryMap`; `exchange_code_base: u32`; `cycles: u64`; `cache: crate::ExchangeScheduleCache`. Consolidates `ScheduledPlan`, `BuiltApplication` and evaluation/cache state.
- **[ShortlistedCandidate](../crates/ipu-codegen/src/compile.rs)** (struct): `proposal: usize`; `estimated_cycles: u64`; `baseline: Candidate`; `recipes: Vec<Recipe>`. Replaces the old function-local shortlist `Candidate`; adds an explicit estimate.
- **[AddressProposal](../crates/ipu-codegen/src/compile/placement.rs)** (struct): `placement: crate::Placement`; `baseline_score: u128`; `score: u128`; `offset: u32`. Names the address-alternative result previously handled within placement optimization.
- **[RowWorkspace](../crates/ipu-codegen/src/kernel/attention.rs)** (struct): `precision: Precision`; `leading: Option<u32>`; `trailing: Option<u32>`. Additional named representation; its role is described in the growth table above.
- **[KernelError](../crates/ipu-codegen/src/kernel/binding.rs)** (enum): `Abi { 0: KernelAbiError }`; `Storage { 0: StorageError }`; `Address { 0: crate::low::storage::AddressError }`; `FragmentedView { shard: u32, spans: usize }`. Replaces `KernelMaterializationError`; adds shared address-error handling.
- **[KernelImplementation](../crates/ipu-codegen/src/kernel/binding.rs)** (enum): `Exact { 0: &'static str }`; `Gemm { 0: Precision, 1: GemmWeightLoad, 2: u32, 3: u32, 4: GemmKernelMode, 5: u32, 6: u32 }`; `Attention { 0: AttentionKernelShape }`; `Softmax { 0: u32, 1: u32, 2: u32, 3: Precision }`; `Merge { 0: u32, 1: u32, 2: Precision }`; `Rearrange { 0: (RearrangeTarget, u32, u32, u32, u32) }`; `Unpack { 0: (UnpackSource, u32, u32, u32, u32) }`. Replaces `KernelSpecialization` and parts of the separate ABI/symbol selection.
- **[KernelCall](../crates/ipu-codegen/src/kernel/binding.rs)** (struct): `implementation: KernelImplementation`; `arguments: Vec<u32>`. Replaces ABI/scalar-getter reconstruction (`KernelAbi`, `ScalarValue`); not an additional executable IR node.
- **[CopyKernel](../crates/ipu-codegen/src/kernel/copy.rs)** (enum): `U16`; `U32`; `U64`; `StridedU32`; `StridedU64`. Additional named representation; its role is described in the growth table above.
- **[CopyRun](../crates/ipu-codegen/src/kernel/copy.rs)** (struct): `movement: LocalCopy`; `kernel: CopyKernel`; `access: [StorageAccess ; 2]`. Additional named representation; its role is described in the growth table above.
- **[CopyPolicy](../crates/ipu-codegen/src/low/copy.rs)** (enum): `Automatic`; `LocalKernel`; `DirectRetile`; `StageLogicalThenTransform`. Renames/reuses `ConversionStrategy` for unified Copy operations.
- **[PackingPolicy](../crates/ipu-codegen/src/low/copy.rs)** (enum): `Automatic`; `Direct`; `Staged`. Additional named representation; its role is described in the growth table above.
- **[BoundView](../crates/ipu-codegen/src/low/storage.rs)** (struct): `shard: &'a BlockValue`; `extents: &'a [ShardExtent]`; `backing: (BlockValueId, i64)`. New shared view/root/origin binding; replaces caller-local reconstruction.
- **[StorageAccess](../crates/ipu-codegen/src/low/storage.rs)** (struct): `alignment: u32`; `access_tail_bytes: u32`. Replaces placement `Requirement` and extracts `KernelAccess` alignment/tail fields.
- **[AddressError](../crates/ipu-codegen/src/low/storage.rs)** (enum): `UnplacedShard { 0: u32 }`; `Overflow`. Additional named representation; its role is described in the growth table above.
- **[CastStorage](../crates/ipu-codegen/src/mid/cast.rs)** (enum): `Separate`; `ReuseIfSmaller`. Additional named representation; its role is described in the growth table above.
- **[CastStoragePolicy](../crates/ipu-codegen/src/mid/cast.rs)** (struct): `default: CastStorage`; `operators: BTreeMap<crate::OperationId, CastStorage>`; `sites: BTreeMap<WorkSite, CastStorage>`. Additional named representation; its role is described in the growth table above.
- **[OperandIndexing](../crates/ipu-codegen/src/mid/compute.rs)** (enum): `Elementwise { result: usize }`; `Local { 0: OperandWindow }`. Additional named representation; its role is described in the growth table above.
- **[Product](../crates/ipu-codegen/src/mid/compute.rs)** (struct): `multiply: Precision`; `accumulate: AccumulationPrecision`; `mode: GemmKernelMode`; `inner_block: u32`; `output_columns: u32`; `axes: ProductAxes`; `operands: [OperandWindow ; 2]`; `output_aliases: Vec<(usize, usize)>`. Replaces GEMM-specific fields inside `Primitive::Compute`/`TileKernelSpec`; axes/windows/aliases already existed.
- **[Compute](../crates/ipu-codegen/src/mid/compute.rs)** (enum): `Product { 0: Product }`; `Kernel { kernel: TileKernelSpec, operands: Vec<OperandIndexing>, output_aliases: Vec<(usize, usize)> }`; `Sum { axis: u16, staging: ReductionStaging }`. Replaces arithmetic variants of `Primitive`; Copy is now directly in `MidOperationKind`.
- **[ReductionGroup](../crates/ipu-codegen/src/mid/grouping.rs)** (struct): `members: Vec<WorkSite>`. Additional named representation; its role is described in the growth table above.
- **[GroupProposal](../crates/ipu-codegen/src/mid/grouping.rs)** (struct): `reductions: Vec<ReductionGroup>`; `homes: BTreeMap<ResultSite, OwnerMap>`. Additional named representation; its role is described in the growth table above.
- **[OwnerChoices](../crates/ipu-codegen/src/mid/ownership.rs)** (struct): `inputs: BTreeMap<crate::ValueId, OwnerMap>`; `operators: BTreeMap<crate::OperationId, OwnerMap>`; `results: BTreeMap<ResultSite, OwnerMap>`. Additional named representation; its role is described in the growth table above.
- **[PanelPacking](../crates/ipu-codegen/src/mid/packing.rs)** (struct): `rows: NonZeroU16`; `workspace: OwnerMap`. Additional named representation; its role is described in the growth table above.
- **[LocalSite](../crates/ipu-codegen/src/mid/site.rs)** (struct): `role: String`; `coordinates: Vec<u32>`. Additional named representation; its role is described in the growth table above.
- **[WorkSite](../crates/ipu-codegen/src/mid/site.rs)** (struct): `source: OperationId`; `local: LocalSite`. Replaces traversal-number `CastSite`; adds constructor-defined local identity.
- **[ResultSite](../crates/ipu-codegen/src/mid/site.rs)** (struct): `work: WorkSite`; `result: u32`. Additional named representation; its role is described in the growth table above.
- **[ProgramError](../crates/ipu-codegen/src/mid/validate.rs)** (enum): `Invalid { 0: String }`; `Layout { 0: crate::tensor::LayoutError }`. Separates executable-program validation failures from planner errors.
- **[ProgramResult](../crates/ipu-codegen/src/mid/validate.rs)** (alias): `Result<T, ProgramError>`. Result alias for executable-program validation.
- **[PackageSupport](../crates/ipu-codegen/src/package/support.rs)** (struct): `memory: TileMemoryMap`; `available_ranges: Vec<(u32, u32)>`; `profile_requests: Vec<Vec<crate::place::AuxiliaryRequest>>`; `exchange_code_base: u32`; `objects: Vec<Vec<u8>>`; `kernel_plan: KernelBuildPlan`; `retained_runtime: Vec<String>`; `layout: LinkedImage`; `physical_to_logical: Vec<u16>`; `code_address: u32`; `generated_code_bytes: u32`; `host_code_base: u32`; `host_code_bytes: u32`; `host_data: Option<MemoryAllocation>`; `exchange_rows: Option<MemoryAllocation>`. Extracts package reservations, objects and linking state from package-building locals; fifteen-field persistent handoff record.
- **[TilePlacement](../crates/ipu-codegen/src/place.rs)** (struct): `addresses: BTreeMap<BlockValueId, u32>`; `sequence_strides: BTreeMap<BlockValueId, u32>`; `unused: Vec<(u32, u32)>`; `auxiliary: Vec<AuxiliaryAllocation>`. Names the tile-placement result and retains established sequence strides.
- **[ValueBuilder](../crates/ipu-codegen/src/planner/bind.rs)** (struct): `values: Vec<MidValue>`; `automatic_inputs: BTreeSet<MidValueId>`; `parameter_values: BTreeSet<MidValueId>`; `copies: BTreeMap<MidValueId, u32>`; `conversion_cycles: u64`. Replaces `LoweringState`; its current `copies` field moved from the outer builder and `conversion_cycles` replaces discarded per-operation cost annotations.
- **[Key](../crates/ipu-codegen/src/planner/cache.rs)** (alias): `(OperatorPlan, Vec<TensorType>, TensorType)`. Internal fragment-cache key alias.
- **[Entry](../crates/ipu-codegen/src/planner/cache.rs)** (alias): `Arc<OnceLock<Option<Arc<MidProgram>>>>`. Internal fragment-cache synchronization/result alias.
- **[FragmentCache](../crates/ipu-codegen/src/planner/cache.rs)** (struct): `entries: Mutex<HashMap<Key, Entry, FixedState>>`. Replaces the implementation map formerly inside `MemoizedCostModel`.
- **[FragmentBuilder](../crates/ipu-codegen/src/planner/fragments.rs)** (struct): `program: MidProgram`. Replaces the old family implementation `Builder`; wraps a complete `MidProgram`.
- **[OperatorFamily](../crates/ipu-codegen/src/planner/operator.rs)** (enum): `Gemm { options: GemmOptions, multiply: Precision, accumulate: AccumulationPrecision }`; `Gelu`; `LayerNorm`; `Add`; `View { 0: AxisFactorView }`; `Slice { 0: crate::graph::AxisSlice }`; `FlashAttention { options: AttentionOptions, accumulate: AccumulationPrecision }`. Renamed `MidOperator`; the compatibility export remains.
- **[RecipeProposal](../crates/ipu-codegen/src/planner/proposals.rs)** (struct): `recipe: Recipe`; `estimated_cycles: Option<u64>`. Additional named representation; its role is described in the growth table above.
- **[Map](../crates/ipu-codegen/src/storage/geometry.rs)** (alias): `HashMap<K, V, foldhash::fast::FixedState>`. Internal foldhash map alias.
- **[GeometryCacheStats](../crates/ipu-codegen/src/storage/geometry.rs)** (struct): `views: CacheStats`; `pairs: CacheStats`; `destinations: CacheStats`. Additional named representation; its role is described in the growth table above.
- **[CacheStats](../crates/ipu-codegen/src/storage/geometry.rs)** (struct): `entries: usize`; `hits: u64`; `misses: u64`; `retained_bytes: usize`. Additional named representation; its role is described in the growth table above.
- **[GeometryView](../crates/ipu-codegen/src/storage/geometry.rs)** (struct): `id: u64`; `traversal: ByteTraversal`. Adds a unique geometry ID to an existing byte traversal representation.
- **[CopyPair](../crates/ipu-codegen/src/storage/geometry.rs)** (struct): `bytes: u64`; `rows: Vec<[StridedSpan ; 2]>`. Replaces cached unit-buffer copy recipes with matched strided rows.
- **[DestinationKey](../crates/ipu-codegen/src/storage/geometry.rs)** (struct): `allocation: u64`; `coverage: Vec<u64>`; `pairs: Option<Vec<(u64, u64)>>`; `same_element_order: bool`; `maximum_fragment_bytes: u32`. Replaces `PlanKey`/`PlanSource` owned geometry keys with normalized IDs and population facts.
- **[DestinationGeometry](../crates/ipu-codegen/src/storage/geometry.rs)** (struct): `bytes: u32`; `coverage: ByteTraversal`; `uncovered: OnceLock<StorageResult<Vec<ByteSpan>>>`; `fragments: Option<u64>`; `semantic: bool`; `destination_word_aligned: bool`; `same_element_order: bool`; `padding: bool`. Replaces destination `CopyGeometry`/`CopyPlan` facts; holes are lazy and selection is outside the cache.
- **[GeometryCache](../crates/ipu-codegen/src/storage/geometry.rs)** (struct): `views: Memo<(CopyOrder, ViewGeometry), GeometryView>`; `pairs: Memo<(u64, u64), CopyPair>`; `destinations: Memo<DestinationKey, DestinationGeometry>`. Replaces both `ExpansionCache` and `GeometryAnalysis`; has three geometry caches.
- **[Broadcast](../crates/ipu-codegen/src/tensor.rs)** (struct): `input: &'a [u32]`; `output: &'a [u32]`; `offset: usize`. Additional named representation; its role is described in the growth table above.
- **[BroadcastError](../crates/ipu-codegen/src/tensor.rs)** (struct): `left: u32`; `right: u32`. Additional named representation; its role is described in the growth table above.
- **[OwnerMap](../crates/ipu-codegen/src/tensor/owners.rs)** (struct): `rotation: u16`; `embedding: Option<Arc<[u16]>>`. Additional named representation; its role is described in the growth table above.
- **[RowBuffer](../crates/ipu-exchange/src/encoding.rs)** (struct): `words: Chunked<u32>`; `sends: Chunked<SendAddress>`; `receive_pointers: Chunked<u32>`; `outgoing_bases: Chunked<OutgoingBaseWrite>`. Replaces the encoder's bare `Chunked<u32>` with words plus relocation streams.
- **[EncodedRow](../crates/ipu-exchange/src/row.rs)** (struct): `words: Vec<u32>`; `sends: Vec<SendAddress>`; `receive_pointers: Vec<u32>`; `outgoing_bases: Vec<OutgoingBaseWrite>`. Replaces raw row words with words plus retained relocation sites.
- **[SendAddress](../crates/ipu-exchange/src/row.rs)** (struct): `word_offset: u32`; `message: u32`; `byte_offset: u32`; `item_shift: u8`. Additional named representation; its role is described in the growth table above.
- **[OutgoingBaseWrite](../crates/ipu-exchange/src/row.rs)** (struct): `word_offset: u32`; `register: u8`. Additional named representation; its role is described in the growth table above.
- **[TopologyError](../crates/ipu-target/src/ipu21/fabric.rs)** (enum): `Tile { 0: u16 }`; `InvalidMapping`. Additional named representation; its role is described in the growth table above.
- **[Topology](../crates/ipu-target/src/ipu21/fabric.rs)** (struct): `logical_to_physical: Vec<u16>`. Moved from `ipu-exchange` into `ipu-target`; its mapping field is not new.
- **[InstructionError](../crates/ipu-target/src/ipu21/instruction.rs)** (struct): `0: &'static str`. Additional named representation; its role is described in the growth table above.

## Added or expanded fields on existing representations

- **Recipe:** added `owners`; replaced scalar `packing_rows` with per-site `packing`, `parallel_reductions`/`disjoint_copy_sources` with `reduction_groups` plus result homes, and `in_place_casts` with `cast_storage`. `cast_before_copies` now uses `WorkSite`. `plans`, `open_boundaries` and `early_casts` predate the refactor.
- **Planner Candidate:** renamed from baseline `Baseline`, retaining `program`, `recipe`, `alternatives`, `cast_sites`; added `cast_storage_sites`, `packing_choices`, `grouping_choices`. This is not the old same-named function-local shortlist record, which became `ShortlistedCandidate`.
- **Recipe::changes debug record:** added corresponding ownership/packing/group/storage change descriptions; replaced the previous scalar-option diff fields. It is diagnostic bookkeeping, not another executable representation.
- **Planner Builder:** added an explicit `fragments: &FragmentCache`; `state: ValueBuilder` replaces `LoweringState`; `copies` moved into that state. Checkpoint `State` lost `mapping` and `mapping_checked` when ownership moved into Recipe.
- **MidOperation:** added `site: Option<LocalSite>`; removed both per-operation cycle-estimate fields. `source`, `inputs`, `results` and `kind` existed.
- **MidValue:** `owners: OwnerMap` replaces `tile_offset: u16`; the optional shared embedding is additional expressiveness/state. `storage_group` existed.
- **MidOperationKind::Copy:** now retains `policy` and `packing` alongside the previous mapping/reuse facts. `CopyPolicy` replaces the separate conversion strategy; `PackingPolicy` is a new choice.
- **Compute::Kernel:** operand windows became explicit elementwise/local `OperandIndexing`. Result/input alias pairs already existed. Sum's axis/staging were moved under Compute, not invented anew.
- **MemoryOperand:** `Output` now contains an index. `KernelRun.outputs` and `KernelRequirements.outputs` replace `output` plus `additional_outputs`. Kernel inputs are direct `ShardView`s instead of `KernelOperand` wrappers.
- **KernelAccess:** `storage: StorageAccess` replaces two direct alignment/tail fields. `KernelBuildPlan.symbols` and `KernelInventory.attention_stages` use the new implementation identity instead of `KernelSpecialization`.
- **TileGraphBuilder:** added `logical_values`; `bindings: Vec<Vec<ShardView>>` replaces canonical shard IDs and the separate borrowed-view map. Its copy/cache collections now contain `CopyRun`/`GeometryCache`.
- **TileGraph:** `value_views` replaces `value_shards`; `requires_finite_scratch` moved here from `LowProgram`; `local_copies` now retains checked `CopyRun`s.
- **DiagnosticShard:** added `view`, preserving selected extents as well as the backing shard's data.
- **Placement:** added `sequence_strides`. `IteratedGroup.argument` identifies the binding whose placed stride is returned; provisional stride/alignment fields were removed. `RepeatRun.binding` replaces three separately stored binding vectors.
- **Relocation:** added `message: u32` to codegen `ScheduledTransfer` and exchange `ScheduledSenderRow`, `PreparedTransfer`, `StagedTransfer`. `ReceiveEventKind::OutgoingBase` gained a register operand. `EncodedSchedule.row`, `PhasePrograms.programs` and `PhysicalExchangePhase.programs` carry metadata-bearing rows instead of raw words.
- **Exchange policy:** `ScheduleRecipe.stream_words` moved from the enclosing cache; policy changes are now checked per retained recipe.
- **MappingTraffic:** added retained `multiplicities`, formerly computed/passed separately by the global mapping optimizer.
- **Expansion diagnostics:** added `ExpansionBenchmark.warm`, `ExpansionTiming.process_memory`, and structured `geometry_cache` statistics; removed the old tuple-shaped cache metrics.
- **Geometry Memo:** replaces custom bucket/entry bookkeeping with `limit` and mutex-protected map/hit/miss state. Keys and normalized IDs are listed above.
- **Error enums:** `LayoutError` gained `InvalidTilePermutation`/`InvalidOwnerMap`; `LoweringError` gained unavailable cast/storage-choice and program-error cases. Package/exchange/codegen errors gained target instruction/topology and program/planning conversions. Tile lowering now shares `AddressError`; expansion shares `KernelError`. These are extra validation/error surfaces, not operation capabilities.

## Additions that have already been removed

- Checkpoint migration functions and deferred legacy-choice fields.
- Preallocated fragment-result bindings, their reconciliation map and synthetic returned-input copies.
- Per-operation and per-Repeat-region cost annotations.
- `StorageRequirements`, `OutputAliasing`, input-only fields on result requirements, and the duplicate local-kernel selection inside `OperatorDispatch::Pointwise`.
- `OperatorPlanError` and temporary planned tensors used only to discard validation errors into a boolean.
- The old separate expansion/cost geometry caches and cached unit-buffer copy recipes.

The original proposal's long implementation paragraph mixed these transient additions, surviving changes and incomplete work. This ledger replaces that paragraph's accounting role; it does not imply that the remaining organization has passed a structural review.
