# Compiler and placement audit

## Current pipeline

1. Model builders append fully expanded `Phase`, `KernelCommand`, `Transfer`,
   and `Allocation` records to a `Schedule`.
2. Exchange lowering groups transfers, assigns non-conflicting tile roles to
   static offsets, and emits one plan row per participating tile.
3. Tile lowering expands every global phase into every tile program, including
   explicit idle-compute steps.
4. Runtime packaging deduplicates exchange rows, discovers required kernel
   symbols, measures generated code, links support images, and independently
   places generated code, support code, host plans, and template records.
5. Static templates compress repeated code and data only after the expanded
   schedule and per-tile programs already exist.

For the one-layer batch-8 SigLIP graph this currently means 603 phases,
537,017 compute commands, 885,329 transfers, and 1,076,111 allocations.
Compression at packaging cannot reduce the cost of constructing or lowering
those records.

## Hardware constraints

These limits belong in target properties and kernel operand constraints:

- Exchange packets encode a finite word count. A larger logical transfer must
  be statically split into packets.
- Exchange staging has a 32-KiB per-tile address window. Independent operations
  can be partitioned into any number of static passes; this is not a tensor-size
  limit.
- Instruction fetch and data access cannot use the same 16-KiB tile-memory
  element concurrently.
- AMP kernels require specific row, inner, output-panel, and coefficient-memory
  layouts. A kernel operand may require a contiguous panel even when the
  enclosing tensor is segmented.
- Kernel-call register operands are finite. Extra operands belong in a
  statically placed argument record, not in an arbitrary graph-size cap.

## Incidental limits and coupling

- Ordinary-low tensor arenas grow downward from the interleaved boundary while
  exchange plans and executable objects grow upward. Packaging measures support
  and generated images together, then relocates conflicting transient homes to
  ordinary-high SRAM and relowers once. This replaces the former fixed
  eight-element reservation with exact instruction-element demand.
- Exchange rows and compute-run tables occupy one contiguous `PLAN_BASE..end`
  interval. Rows are independently addressed and can instead be placed as
  relocatable executable objects; run tables are ordinary data objects.
- Tensors receive final addresses while operations are appended, before code
  size is known. Packaging can diagnose a code/data collision but cannot move a
  legal tensor to another allowed arena.
- Row-shard choices have historically been local to each operator. Residuals,
  Q/K/V, and other simultaneously live values impose a shared edge-layout
  contract. The placement query now models equal-layout copies, but the layout
  itself still needs a first-class IR type.
- Static templates are discovered too late. Repeated encoder layers should be
  represented as one symbolic phase block with per-instance weight and tensor
  relocations before transfer/allocation expansion.

## Standard placement model

Placement should consume objects with these properties:

```text
MemoryObject {
  tile
  size, alignment
  lifetime
  access: data | executable
  allowed arenas in preference order
  contiguity group
  relocations
}
```

One per-tile placement pass should place permanent weights, transient buffers,
exchange rows, run tables, host state, generated code, and linked support
segments together. Executable objects reserve complete instruction elements;
ordinary data reserves only its byte range. Tensor layouts remain segmented,
while individual kernel operands declare any contiguous panel they require.

The runtime now uses one `AddressSpace` implementation to normalize
reservations and calculate gaps for all static object classes. The remaining
work is to carry explicit allowed memory classes from compiler allocations into
that pass, then remove the contiguous plan region.

## Compile-time priorities

1. Keep exchange role assignment incremental. The original DSATUR
   implementation repeatedly scanned every uncolored group. A priority-queue
   implementation retains the compact schedule without that quadratic scan.
   The eleven additional launches seen in one experiment came from reducing
   the row-block dimension from 36 to 18, not from exchange coloring.
2. Store exchange staging addresses with transfers instead of duplicating one
   full `Allocation` per transfer. This removes a large fraction of the million
   allocation records and simplifies source/destination lookup.
3. Introduce symbolic repeated phase blocks before lowering. Encoder instances
   share code and exchange structure while weight addresses and selected tensor
   addresses remain relocation fields.
4. Replace explicit idle-compute steps with phase ranges or a sparse per-tile
   step stream. Profiling can synthesize idle intervals from global phase
   boundaries.
5. Maintain an incremental lifetime-aware allocation index. Rebuilding and
   sorting occupied intervals from the flat allocation vector should be a
   compatibility path, not the primary allocator.

## Worker launch boundary

The FP16 AMP supervisor already loops over all output and inner microblocks in a
kernel call. Each microblock loads a new 16x16 coefficient block, launches the
six workers, and waits before replacing shared coefficients. Poplibs follows
the same basic synchronization pattern. A persistent worker loop cannot simply
remove these barriers.

The existing kernel has a valid 128-bit coefficient-load path for weights in
interleaved SRAM, but FP16 kernel specialization does not yet derive that choice
from placement. Longer-term launch reduction requires a proven dual-coefficient
bank or worker-coordinated loading scheme; it should be driven by cycle profiles
and hardware tests rather than assumed from the execution model.
