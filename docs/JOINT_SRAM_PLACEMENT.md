# Joint SRAM placement (2026-09-07)

Standard and interleaved buffers now share a lifetime-ordered allocator. Region 0
accepts ordinary storage; region 1 accepts both classes. The old allocator placed
interleaved buffers first and permanently excluded their high-water range from
ordinary storage, even after those buffers died.

Requests are ordered by first use, with interleaved requests first at a shared
start event, then alignment/size. Standard requests prefer region 0. Within each
region, low addresses are preferred to retain large contiguous free spans. Dead
allocations return to the same free-space pool regardless of their access class.
Aliases and Repeat carried values retain their combined lifetimes. Iterated
parameters retain contiguous, uniformly strided storage, with a region-dependent
stride when bank separation requires 16 KiB elements or 32 KiB element pairs.
No runtime relocation or address-mode configuration is introduced.

Package support is reserved before final placement. Auxiliary allocations only
use addresses never occupied by any planned buffer, not merely addresses free
at the last event. Bank-separated ordinary buffers placed in region 1 reserve
whole 32 KiB pairs too.

The planner screens maximum simultaneous live bytes plus estimated package
support, and checks the interleaved-region capacity separately. It no longer
rejects a candidate solely because independent class peaks sum above capacity.
The contiguous-allocation screen likewise permits standard storage to use all
of region 1 at a different time. These remain estimates: actual placement on
expanded finalists checks alignment, lifetime interference and concrete support
reservations. No extra tile expansion or placement is added inside the beam.
Placement remains a greedy heuristic, not a proof that rejected layouts cannot
fit under another allocation order.

## Cost and validation

Existing batch-one logs measured final placement at 49 ms for MLP and 60 ms for
fused attention. New runs measured 31 ms and 47 ms respectively, across all 1472
tiles. These are observations under differing concurrent build load, not a claim
of allocator speedup. Provisional placements generally take tens of milliseconds;
some simultaneous attention finalists took roughly 0.1 seconds. The repeated
batch-two MLP provisional placement took 63 ms. Placement partitions requests
by tile and runs those tiles in parallel; it sorts requests and scans active/free
ranges, without exchange scheduling. More placements of already-expanded
finalists are inexpensive compared with compilation, but expanding every beam
branch just to place it would still be costly.

Hardware tests use F143 scale -4, Gaussian reference checking, default layout
search and 16 Rayon threads per concurrent build. All cycle counts use the
renderer-cropped profile span. Results live under `artifacts/joint-placement/`:

| Workload | Artifact directory | Cropped cycles | Maximum absolute error |
|---|---|---:|---:|
| MLP batch 1, one block | mlp-single-b1 | 128,202 | 0.006378 |
| MLP batch 4, one block | mlp-single-b4 | 400,842 | 0.007324 |
| Attention batch 1, separate QKV | attention-b1 | 170,958 | 0.000056 |
| Attention batch 1, fused QKV | attention-fused-b1 | 172,206 | 0.000054 |
| MLP batch 2, three blocks | mlp-b2-n3 | 1,031,874 | 0.015625 |

The batch-four log embeds its original output directory name `mlp-b1`: the CLI
batch default was four; the completed directory was renamed to match the actual
workload. This is not a batch-one result. The latest host-FP8 changes had not been
swept before this refactor, so higher-batch success alone does not isolate the
allocator's contribution from those earlier changes.

Unit coverage includes alternating class peaks that fit 320 KiB jointly instead
of a 512 KiB fixed partition, randomized mixed-class lifetime non-overlap,
persistent auxiliary-space exclusion, region/element restrictions, Repeat
strides and existing end-to-end placement tests. The full release tests passed
(147 codegen, four CLI, one doctest; three ignored), as did Clippy with the
repository's argument-count/type-complexity allowances.

A smaller three-block Repeat (129 tokens, 128 channels, hidden width 256, 64
active compute tiles) also passed hardware reference checking, maximum absolute
error 0.015625, under `mlp-small-repeat`. The Repeat stride regression explicitly
checks both ordinary-region placement and forced region-1 placement.

For the full batch-two three-block MLP, schedule selection took 351.658 seconds,
while final storage placement took 52 ms. The existing exchange-placement search
then tried seven address offsets; each allocation took 39–49 ms. Exact exchange
validation/rescheduling is separate and can take minutes on this workload. A
10-second perf sample of initial scheduling attributed about 70% of sampled CPU
time to `BinaryHeap<RepairReady>::pop` and 21% to `repair_ready`, rather than the
allocator. The refactor does not change the limits on address challengers or
exact scheduling attempts. Testing many placements for *memory feasibility* is
cheap; exact exchange scoring of every placement would not be.
