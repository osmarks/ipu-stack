# Host descriptors and fragmented SRAM placement

Baseline: `0783a20`. Changes: `ed3978e`, `27a4be2`, `69c8915`.
Artifacts: `artifacts/placement-20260910/`.

## Descriptor storage is part of the reservation

Previously final host-command data was allocated only from addresses unused by
any tensor over the entire program. Two batch-four finalists placed tensors and
scheduled their exchanges successfully, then failed on a 12-byte descriptor.
That reported the next allocation, not the total missing space.

Host planning now counts packet and descriptor bytes per tile before packet
cache deduplication. Transfer slicing and phase membership depend on binding
sizes and tiles, not tensor addresses, so this bounds final storage even when
relocation changes packet equality. Package assembly reserves the maximum tile
requirement, aligned to four bytes, before placing tensors. Final host planning
can use that reservation and any otherwise unused ranges. This is a conservative
common reservation across tiles, consistent with the current common support map.

Provisional sizing uses a temporary descriptor address range: it no longer needs
holes in provisional tensor placement merely to determine the required size.
Generated host instructions and descriptors still come from the same emitter.
The relocation test starts with shared packets, changes their source addresses,
and checks that the resulting unshared packets fit the original bound.

## Retry allocation order before rejecting a tile

The ordinary allocator remains lifetime-ordered and unchanged on success.
On an out-of-memory result, that tile retries once, ordering by decreasing
alignment, size and lifetime duration. This protects large constrained buffers
from holes created by earlier small allocations.

The retry reconstructs free spans against already placed allocations whose
lifetimes overlap the new request. This permits arbitrary allocation order
without treating disjoint lifetimes as simultaneous. Both paths use the same
address-class, element-separation, access-tail, Repeat stride and loader-boundary
rules. Alias groups and Repeat constraints are analyzed once and retained.
No hardware requirements were relaxed. Failed tiles alone incur the retry;
there is no additional exchange-scheduling candidate.

A targeted case demonstrates recovery when a persistent small buffer separates
two free spans needed by a later large buffer. Randomized checks exercise both
allocation modes with mixed memory classes, element rounding, interleaved
offsets and overlapping lifetimes. Existing graph-level alias/Repeat placement
tests also pass. The final full codegen suite passes 210 tests with five ignored.
Clippy passes with the existing complexity allowances.

## Exposed packing alignment bug

The first new batch-four package completes assembly (finalist 4, 525,210 ms
selection), but 14 physical tiles fault in the first 64-bit load of
`rearrange_row_major_to_amp_f16_o0_r31_p31_c72_p80`. Disassembly confirms offset
`+0x8c` is `ld64step`; the diagnostic reports `InvalidMemoryAddress` there in
all six workers. Rearrangement contracts still declared two-byte alignment,
left over from the scalar F16 implementation. New descriptor placement exposes
that invalid assumption. Rearrangement now uses the normal eight-byte buffer
alignment required by its assembly fast paths.

The failed package and disassembly are retained under `vit-b4/`. It is not a
successful hardware result. The corrected build is under `vit-b4-aligned/`.

## Model validation

The first batch-two run with descriptor reservation and allocator retry passes
reference and hardware validation: 923,148 cropped cycles, 675,708 encoder
cycles, maximum absolute error 0.101074. The previous scheduler baseline was
918,432 / 670,560 cycles. Profile: `vit-b2/profile.html`.

The corrected batch-four package passes both reference and hardware validation,
with maximum absolute error **0.084717**. Finalist 4 is selected in 497,548 ms
(8.29 minutes); final tensor placement takes 378 ms. It reserves 2,516 bytes of
host data per tile. Its cropped profile spans **1,622,310 cycles (1.08154 ms)**,
with **1,247,040 cycles** in the encoder interval. The profile uses the renderer's
initial-entry cutoff, and the encoder interval is operation 23's final offset
minus operation 3's first offset. Profile: `vit-b4-aligned/profile.html`.

This establishes feasibility for the existing one-layer benchmark at batch four,
not the full 27-layer model or every other finalist. The originally selected
candidate's late descriptor failure is fixed; the allocator retry does not
establish that all previously fragmented alternatives are feasible.

The final batch-two build also passes reference and hardware validation, with
unchanged maximum absolute error 0.101074. It takes **923,100 cropped cycles
(0.6154 ms)** and **675,666 encoder cycles**: 0.51% / 0.76% slower than the
previous 918,432 / 670,560 baseline. Profile: `vit-b2-aligned/profile.html`.
Each distinct package was executed once; there were no repeated timing samples.

The failed and successful model builds used the same descriptor bound and
allocation retry as the final tree. Provisional descriptor sizing was decoupled
from tensor holes between the first and corrected builds; neither sizing's
virtual addresses nor its packets are emitted in the final package.


