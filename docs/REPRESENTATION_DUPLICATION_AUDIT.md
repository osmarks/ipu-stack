# Representation duplication audit

The repository contains several cases where one decision is copied into
multiple structs and then kept consistent by destructuring, reconstruction,
or validation. These copies are more costly than ordinary layer boundaries:
they can let planning, estimation, lowering, diagnostics, and serialization
describe different programs.

The audit covers all Rust crates and both Cap'n Proto schemas. The 28
`too_many_arguments` suppressions are useful evidence, but a long function is
only a refactoring target when its arguments reconstruct an existing concept.

## Immediate targets

### One blocked-GEMM plan

GEMM geometry is represented by:

- private `GemmPlan` generation records;
- `OperatorDispatch::BlockedGemm`;
- `GemmDistribution::ParallelReduction`;
- `AmpGridShape` and `ParallelGridProxy` during search;
- `GemmPlanConstraint` for diagnostic selection;
- both `TileKernelSpec::Gemm` values; and
- the scalar arguments to the three GEMM lowering functions.

The row, column, K, and result partition counts are the same decision in all
of these places. `inner_block` and `output_column_block` also occur in the
whole-device dispatch and in both initialize and accumulate kernel specs.
Validation currently detects some disagreements after constructing them.

Introduce shared value types:

```text
GemmGrid { rows, columns, inner }
ResultGrid { rows, columns }
GemmBlockShape { inner, output_columns }
ParallelReductionPlan { compute_grid, result_grid, staging }
BlockedGemmPlan { block, orientation, distribution, kernel_family }
```

`OperatorDispatch` should contain `BlockedGemmPlan`. Search proxies should
contain or refer to its grid types. Low lowering should accept a plan reference
rather than its fields. Tile-kernel specs should retain only tile-local ABI
choices derived from the plan.

Diagnostic forcing remains useful, but `GemmPlanConstraint` should not define
another schema for GEMM geometry. It should pair an operation ID with a key or
selector derived from the same `BlockedGemmPlan` components. Diagnostic
overrides should also move out of the ordinary planner search domain.

### One selected operator plan

Private `Plan` and public `OperatorPlan` repeat the operator, dispatch,
requirements, and deferred output. `Plan::supports` constructs a temporary
`OperatorPlan` merely to validate it. A selected `MidOperation` then stores the
operator in both `MidOperationKind::Operator` and `OperatorPlan::operator`.

Use one operator-plan type throughout generation and selection. Estimates and
claimed deferred inputs should be annotations on the selected operation, not
reasons to copy the plan. An operator operation should obtain its semantic
operator from one location.

### One allocation-constraint representation

Allocation requirements are independently represented by:

- `OperandRequirement` in the selected plan;
- `estimate::AllocationRequirement`; and
- `place::Requirement`.

The estimator and placer separately collect access-tail and distinct-memory-
element requirements. The placer adds alignment to its copy. This is a direct
risk that estimated feasibility and physical placement apply different rules.

Define one `AllocationConstraints { alignment, access_tail, element_policy }`
type. Operand requirements should contain it, alias groups should merge it,
and both memory estimation and placement should consume the same merge logic.
`MemoryRequest` remains a later address-range request derived from these
constraints.

### One attention plan vocabulary

`BlockedAttention` and `MaterializedAttention` repeat their two kernels, query
block size, and padded query/value dimensions. The same dimensions occur in
attention `TileKernelSpec` variants, `AttentionBufferShape`, kernel compilation
keys, and long lowering argument lists.

Introduce an `AttentionPlan` with a common block shape and an algorithm enum
for blocked versus materialized execution. Scratch-buffer shapes and local
kernel calls should be derived from it. `AttentionTask` and prepared panels are
genuine lowering state and should remain separate, but should not restate plan
geometry.

### One metrics vocabulary

`GemmPlanObjective` and `BeamObjective` flatten nearly the same cycle and
memory fields and implement separate dominance rules. `MidOperation` carries
two cycle scalars plus `MemoryEstimate`; `RearrangementCost` carries the same
cycle pair and exchange-row estimate in another shape. `ExchangeFootprint` is
then converted into row bytes and copied into memory estimates.

Use nested shared records rather than copied scalars:

```text
CostEstimate { cycles, exchange_cycles, exchange_footprint }
PlanMetrics { cost, memory }
ParetoKey { metrics, compatibility }
```

GEMM pruning and global beam pruning may select different projections, but
they should project from the same metrics and use one dominance helper.

## Structural targets

### Typed tensor and address regions

Semantic tensor intervals are repeatedly encoded as `(u32, u32)`, including
deferred transforms, conversion estimates, replica-group keys, and attention
panel arguments. Physical SRAM ranges use the same tuple representation in
placement, host planning, ELF link options, and exchange diagnostics.

Use a `TensorRegion` built from `ShardExtent` for logical/physical tensor
regions, and a distinct `AddressRange` for SRAM intervals. Deferred panel
helpers should accept one region or panel-slice object instead of five scalar
coordinates. `ByteSpan { offset, bytes }` remains a distinct storage-level
concept.

### One physical exchange-transfer core

One physical transfer is successively restated by
`ExchangeScheduleTransfer`, `PendingTransfer`, `ScheduledTransfer`, and several
test-only transfer structs. Destination endpoints alternate between
`ExchangeScheduleDestination` and `(u16, u32)`. Conversion between these forms
is manual.

Define shared `TransferEndpoint` and `PhysicalTransfer` records. A pending
transfer should wrap the physical transfer with low-IR provenance, memory
elements, and paired-resource reservations. The schedule snapshot should
serialize the core record directly.

Timing has a related duplication: `ipu_target::PhaseTransferTiming`,
codegen's `ScheduledTransferTiming` and `MaterializedTiming`, diagnostics, and
profile activities hold overlapping event intervals. The encoder's timing
result should be authoritative; scheduler memory windows and dependency links
should extend it rather than copying its fields.

`ipu_target::Plan` is only the one-receiver form of `MulticastPlan` and is
immediately converted by callers. `point_to_point` should return the general
form and the wrapper should be removed.

### One package build result

`CompiledPackage`, `DiagnosticPackage`, and internal `BuiltApplication` repeat
the application, physical exchange phases, schedule snapshot, exchange-table
base, precision map, and input metadata. Ordinary and diagnostic builds then
have parallel construction paths.

Use one `CompiledPackage` containing common build artifacts and optional
checkpoint metadata. Factor exchange artifacts and tensor metadata into named
nested records. `DiagnosticTensor` and serialized `Binding` should be derived
from one compiled tensor-storage description so names, shapes, precision, and
physical slices cannot diverge.

### One profile vocabulary and schema fragment

Profile step and exchange-activity enums are duplicated in
`application.capnp`, `profile.capnp`, `ipu-package`, `ipu-profile`, and CLI
adapter enums. `ipu-package` contains separate hand-written encode/decode
matches for application profile plans and standalone profile reports.

Move the common Cap'n Proto enums and step metadata into an imported schema
fragment. Use `ipu_package::ProfileStepKind` directly in profile queries rather
than defining `ipu_profile::StepKind`. CLI parsing can use `FromStr` or small
parsers without introducing three more public enum vocabularies. Cycle samples
and package step plans remain different containers around the shared step
description.

### Diagnostic and stress-test transfer records

The exchange stress tests define `Transfer`, `TransferSpec`, `Payload`,
`ReplayTransfer`, and `ReplayEvent`, repeating production endpoints, word
counts, offsets, and timing. This makes diagnostics capable of testing their
own interpretation rather than the production representation.

Construct stress and replay cases from the shared physical-transfer record.
Keep expected payload contents and readback samples as test-only data attached
to that record.

## Small removals

These wrappers add little or no type safety and should be removed when their
callers are next touched:

- one-variant `HardwareTarget` and `SchedulingPolicy`;
- `ProfilingConfig`, which wraps one Boolean;
- one-variant `TileKernel` around `TileKernelSpec`;
- one-variant `RuntimeError` around `DriverError`;
- one-variant `MemoryRelation` if distinct-element policy moves into allocation
  constraints; and
- CLI-only `AttentionMode`, `ProfileGroup`, `ProfileSort`, and `ProfileKind`
  where the underlying types can provide parsing without acquiring a CLI
  dependency.

## Boundaries to preserve

Similarity alone is not a reason to merge representations. These boundaries
carry useful guarantees:

- `Operation` to `MidOperation` to low tile work to address-resolved
  `TileStep`;
- `ValueId`, `MidValueId`, and `LowShardId`;
- symbolic tensor regions versus physical byte spans;
- logical exchanges versus address-resolved physical transfers versus encoded
  exchange rows;
- graph, mid-level, low-level, and finalized Repeat records; and
- `LinkedSegment`, which views a linked image, versus package `Segment`, which
  owns serialized bytes and permissions.

Shared nested value types should cross these boundaries where they describe an
unchanged decision. The enclosing records should remain distinct when they add
resolution, placement, ownership, or serialization.

## Refactoring order

1. Normalize GEMM grid, block, distribution, and diagnostic selector types.
2. Remove `Plan`/`OperatorPlan` and duplicated operator identity.
3. Share allocation constraints between estimation and placement.
4. Normalize attention plan geometry and lowering signatures.
5. Introduce shared plan metrics and Pareto comparison.
6. Type tensor regions and SRAM address ranges.
7. Normalize physical exchange transfers and timings.
8. Consolidate package build results and tensor metadata.
9. Share profile schema vocabulary and remove CLI/query mirror enums.
10. Migrate stress diagnostics to production transfer types and remove small
    wrappers opportunistically.

Each structural change should preserve the canonical package byte-for-byte
unless it deliberately fixes an identified disagreement. Randomized tests
should compare the shared representation's derived estimates, lowering, and
serialization rather than reconstructing an independent expected algorithm.
