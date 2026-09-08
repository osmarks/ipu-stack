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

## Full build with unified cycle scores

At `2d08e93`, ran the full B2 FP8 ViT build with default automatic planning,
reference checking requested, and detailed profiling requested. Results:

- Mid search: **243,500 ms**, versus **239,708 ms** in the earlier wider-shortlist
  build (about 1.6% longer, not a controlled isolated microbenchmark).
- All **64** expanded finalists predict tables below 65,536 bytes; minimum
  **36,808 bytes**. Previously none of 62 did, with minimum 95,696 bytes.
- Four finalists were placed/modelled; two admitted finalists were scheduled.
- Finalist 2: estimate **36,856 B**, actual encoded table **35,592 B**.
- Finalist 16: estimate **36,808 B**, actual encoded table **35,436 B**.

Both encoded tables satisfy the budget. Both packages then fail *post-link
support placement*: tile 506 cannot allocate **98,304 bytes of interleaved SRAM**
after reserving the linked code and package support. The attempt ended after
427,617 ms of selection, with no exchange-budget penalty retry. It produced no
finished package, hardware result, or profile. The other finalists have not been
shown infeasible by this test.

Thus early exchange ranking is substantially improved and the table-size failure
is removed for the two attempted plans. **Automatic B2 ViT building is still not
fixed overall**: the remaining demonstrated failure is package-aware SRAM
placement, not exchange-row encoding. Do not treat the historical hardware
profile as validation of these new selections.

Full command, log and machine-readable result are in
`artifacts/vit/early-geometry-full-b2-fp8/`.
Validation: 191 codegen tests and one doctest passed (four ignored); Clippy passed
for `ipu-codegen` and `ipu-tests`, including all targets, with the repository's
existing `too_many_arguments`/`type_complexity` allowances.

## Package fallback and the final partial memory element

The historical up-projection partial-output buffer was 52,480 payload bytes,
rounded to a 64 KiB interleaved reservation (nine row partitions, 27 column
partitions, six K partitions). The rejected newer choice has 71,040 payload
bytes, rounded to 96 KiB (ten row partitions, 18 column partitions, eight K
partitions). These are the same logical intermediate, not an added buffer.

At `95fa345`, scheduling retains the remaining already-placed candidates after
its preferred fast/compact choices. A successful package stops further attempts;
failed attempts remain bounded by the placement shortlist. This preserves search,
expansion, placement and compatible schedule cache entries. The B2 test in
`artifacts/vit/placement-fallback-b2-fp8/` tried indices 2, 16, 22 and 28. All four
failed package-aware SRAM placement. Candidate 28 got past up-projection but
failed on a 64 KiB down-projection partial-output reservation. This is not proof
that all 64 expanded candidates are infeasible.

The detailed free ranges exposed a narrower allocator restriction. The SDK loader
stops at `0xe7bb0`, 1,104 bytes before the architectural memory boundary
`0xe8000`. The last interleaved element was therefore unavailable to any request
requiring a whole-element reservation, even when its actual payload and access
tail ended below the loading limit. Runtime stack state is reserved elsewhere.

Placement now keeps payload size separate from reservation rounding. A standalone
distinct-element allocation may reserve the available portion of the last element
if its complete payload/access tail remains loadable. It keeps element alignment
and exclusivity, never clips at ordinary free gaps, and leaves repeat-group
strides unchanged. This does not introduce scattered buffers, change kernels, or
allow a package segment beyond the SDK loader's range.

Regression tests check the loading boundary, exclusive ownership while live,
reuse after death, rejection of even one payload byte beyond the limit, and
rejection of equivalent clipping at an ordinary free gap. All 193 codegen tests
and the doctest passed (four ignored); Clippy passed.
