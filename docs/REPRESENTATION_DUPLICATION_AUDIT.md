# Representation boundaries

This document records the canonical representations shared across planning,
lowering, packaging, profiling, and diagnostics. A decision is stored once;
later stages either refer to it or derive a representation with additional
resolution.

## Compiler plans

- `OperatorPlan` is the selected whole-device operator plan. A
  `MidOperationKind::Operator` stores it directly, so semantic operator,
  dispatch, operand requirements, deferred inputs, and deferred output have
  one owner.
- `GemmGeometry` owns the block shape, orientation, final result grid, grid
  order, and distribution. `ParallelReductionPlan` adds only its compute grid
  and staging policy. `GemmPlanConstraint` refers to the same geometry type.
- `GemmKernelFamily` contains only shape-independent tile-kernel choices.
  Concrete `TileKernelSpec::Gemm` calls are derived from the family and the
  canonical block shape.
- `AttentionPlan` owns its algorithm, blocking, padding, and GEMM kernel
  family. Attention GEMM calls and `AttentionBufferShape` are derived from it.
  `AttentionTask` and prepared panels are transient lowering state.
- `AllocationRequirements` is merged and consumed by both estimation and
  placement. Alignment, access tails, and memory-element separation are not
  re-expressed in estimator- or allocator-private schemas.
- `ConversionPlan` owns the target-selected direct-retile, direct-logical, or
  staged execution strategy and its `CopyPlan`. The cost model selects it and
  the mid-level conversion stores it; low lowering binds that plan to shards
  without selecting a different materialization.
- `CopyRun` is the shared address-independent local-copy representation.
  Planning and lowering derive it from the same physical span sequences, so
  contiguous/strided call selection and its target threshold are not a
  low-level peephole pass.

## Cost and memory

- `CostEstimate` contains total cycles, exchange cycles, and
  `ExchangeFootprint`.
- `PlanMetrics<M>` pairs a cost with the relevant memory record.
  `OperationMetrics` and `RegionMetrics` are aliases over this type.
- Pareto retention uses `RegionMetrics::dominates`; compatibility classes may
  restrict which candidates are comparable without defining another metric
  vocabulary.
- Deferred materialization stores its complete `CostEstimate`, so restoring a
  deferred operation cannot lose exchange cost or row-footprint information.

## Tensor and memory regions

- `TensorRegion` is a rectangular semantic or padded tensor region composed of
  axis-labelled `ShardExtent`s. Layout intersections, replica identities,
  deferred slices, and lowering views use it directly.
- `AddressRegion` is a half-open tile-SRAM interval. Placement arenas, package
  maps, and host-planning free ranges use this physical type.
- `ByteSpan` remains an offset-plus-length description inside one storage
  object. It is not an address interval or a semantic tensor region.
- Cycle intervals remain cycle pairs; they are not memory regions.

## Exchange

- `PhysicalTransfer` and `TransferEndpoint` are the address-resolved exchange
  core. Schedule snapshots serialize `PhysicalTransfer` directly.
- Codegen `PendingTransfer` wraps the physical transfer only with low-IR
  provenance, source-slice information, and paired-resource reservations.
- `ResolvedTransfer` adds topology-derived point-to-point encoding. Encoded
  exchange rows are the next resolution boundary.
- `PhaseTransferTiming` is the encoder's detailed timing result. The scheduler
  retains only dependency-chain summaries needed by its search; profile
  activities and emitted diagnostics are observations of the selected
  schedule.
- Exchange replay uses the captured physical transfer list as its source of
  truth and verifies activity metadata against it. Stress records attach
  expected payloads and requested timing to the same physical transfer.

## Packages and profiles

- `CompiledPackage` is the result of ordinary and diagnostic compilation.
  Diagnostic builds populate its checkpoint list rather than constructing a
  parallel package result.
- `CompiledTensor` owns logical tensor metadata and placed shards. Host
  `Binding`s and checkpoint tensor descriptions are derived from it.
- `ProfileStepKind` and `ExchangeActivityKind` are defined by `ipu-package` and
  used directly by codegen, profile queries, tests, and the CLI.
- Application profile plans and standalone profile reports share the Cap'n
  Proto `profile_common.capnp` step schema and the same Rust reader/writer.
  `TileProfilePlan` and `CycleSample` remain distinct containers because one
  is a static instrumentation plan and the other is a measured interval.

## Configuration

- `HardwareTarget` remains the dispatch point for target cost models and
  memory constraints, even while IPU21 is the only implemented target.
- `ProfilingConfig` is the explicit `Disabled`, `Overall`, or `Full` policy.
  It is not a Boolean wrapper.

## Stage boundaries

The following identities intentionally remain distinct:

- `Operation`, `MidOperation`, low tile work, and address-resolved `TileStep`;
- `ValueId`, `MidValueId`, and `LowShardId`;
- logical exchanges, physical transfers, resolved transfers, and encoded rows;
- graph, mid-level, low-level, and finalized structured-repeat records;
- symbolic tensor regions, storage byte spans, and physical address regions;
- `LinkedSegment`, which borrows a linked image, and package `Segment`, which
  owns serialized bytes and permissions; and
- internal device exchange transfers and host-exchange protocol transfers.

These boundaries add placement, topology, ownership, serialization, or
measurement semantics. Shared nested value types cross them whenever the
underlying decision is unchanged.
