# Early exchange geometry and the missing B2 ViT plan

The late footprint model recognizes the historical fitting B2 plan. The more
immediate problem is the early ranking of conversions and weight broadcasts.

## Historical control

Reconstructed the full B2 FP8 ViT finalists at `8197fe1` in an isolated checkout.
The recorded finalist estimates reproduce exactly. Captured finalist 0's 61
exchange phases without scheduling or building a package and passed them through
today's address-independent footprint estimator:

| Measurement | Per-tile table bytes |
| --- | ---: |
| Historical encoded package | 43,820 |
| Current estimate of reconstructed historical geometry | 46,008 |
| Same estimate without cross-phase sharing | 51,552 |
| Budget | 65,536 |

The estimate is 5.0% above the recorded encoded table, comfortably within budget.
This does not prove the current compiler would produce byte-identical rows, but
it rules out the new late footprint estimate rejecting that captured geometry.

## First layout divergence

Compared expanded mid plans from the historical revision and `ea407b3`. The first
layout difference in finalist 0 is operation 4, the Q projection. Input projection
and its immediate pointwise operations keep their previous layouts; the input
cast's cycle estimate has changed with the faster cast kernel.

Historical Q preparation rearranges F16 into its consumer layout, then casts.
The recent plan casts one row per tile first, then redistributes FP8 into taller
AMP-left panels. It also changes the GEMM from nine K partitions to six.

The recent retile has 1,458 source owners and 1,440 consumer tiles (24 replicas of
60 unique panels). A largest consumer shard is `[2, 74, 192]`, or 28,416 bytes.
The old estimate divides that by 256 and charges 111 fragments. Its physical
geometry instead has 148 rows times six 32-byte panels: **888 fragments**.
This count agrees with expanded physical byte spans. Replicas add multicast
receivers, not additional source sends.

This is a demonstrated ranking error and a first divergence between complete
plans, not a trace of the precise beam step at which every historical branch
was discarded. GEMM candidate-generator files have not changed between these
revisions; cast ordering, caching, and frontier accounting have.

## Implementation

`estimate/movement.rs` factors compatible grid layouts into physical axes and
counts runs at the innermost differing ownership boundary. Equal trailing
partitions form contiguous runs. An ordered interval walk counts intersections
without constructing the Cartesian product of tiles or expanding byte spans.
It also accounts for the hardware's maximum transfer size.

Supported cases are row-major grids, aligned AMP-left/output grids, and
whole-shard broadcasts in any encoding. The latter corrects the opposite error:
unchanged packed weights are contiguous, rather than one fragment per 256 bytes.
Local intersections remain included conservatively; replicas do not multiply
multicast sends. Views, mixed formats, linear ownership, and unsupported packed
partition changes retain the prior estimate. Existing cycles/storage objectives
consume the new fragment count; no beam dimension or hard estimated-byte cutoff
was added. The early row-byte price itself remains coarse and does not model
cross-phase sharing.

The first implementation's nested axis scan dominated a sampled planning run
(69% self samples). That run was stopped. The retained implementation uses a
linear interval walk. A separate run with an incorrect AMP grain was also stopped
after the span-comparison test exposed it; neither run supplies finalist results.

## Validation and artifacts

Tests compare against actual physical-span expansion across 243 uniform grid
pairs, uneven grids in FP8/FP16/FP32, the B2 projection's 888-fragment retile, and
whole packed-shard replication. These are geometry tests, not hardware reruns.

Artifacts live under `artifacts/vit/early-geometry-b2-fp8/`: historical and recent
mid dumps, comparison script/output, historical exchange snapshot and footprint
log, and temporary diagnostic patches. The patches stop before scheduling and
are not part of production code. The snapshots and dumps allow subsequent
cost-model comparisons without another historical hardware run.

## First complete shortlist comparison

The first complete geometry run (fragment counts used by mid primitive costing
and conversion storage refresh, before updating explicit conversion cycle
scores) expanded 62 finalists. **Three** were below 64 KiB:

| Finalist | Expanded footprint estimate |
| --- | ---: |
| 33 | 46,456 B |
| 34 | 46,560 B |
| 32 | 56,296 B |

The previous 62-finalist run's minimum was 95,696 B and none fit. This is an
expanded-geometry result, not an encoded-package or hardware result. Finalist
33 is not a byte-identical resurrection of the historical plan.

From diagnostic file timestamps, lowering through completed mid finalists took
304.4 seconds, versus 239.7 seconds for the unmodified recent baseline. The new
run's slowest concurrent expansion/footprint task took 44.0 seconds. These runs
shared the host with development builds, so treat the wall-time comparison as
indicative. A sample of the linear implementation no longer showed the fragment
counter above 2% self time; layout-bound construction dominated.

The comparison exposed a second estimate path: explicit conversion operations
received bandwidth-only cycle scores from `CostModel::rearrangement_cost`, while
`refresh_exchange_rows` updated their storage estimate through mid costing.
Compatible direct retiles now use the same geometry and fragment-cycle price in
both paths. That also affects the memoized conversion scores used before beam
pruning. Other conversion strategies retain their existing cost behavior.
