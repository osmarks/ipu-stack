# Result grids and tile mapping in normal planning

The planner now generates feasible full reduce-scatter factor pairs, including
mixed row/column grids. Exact constrained grids remain available for experiments.
Candidate costing includes the mid reduction implementation and downstream
layout boundaries. Shortlist diversity distinguishes K splits and scatter
directions (collapsed, rows, columns, mixed), rather than giving every exact
result grid its own diversity slot. The existing shortlist-size bound and tied
memory-alternative retention remain unchanged.

Package planning models at least four complete mid finalists, expanding them in
parallel. This stage prepares physical transfer fragments but does not schedule
them or compile kernels. It evaluates mappings derived from the selected grids'
tile strides, including transposes within repeated groups and across the active
device. No particular MLP shape or experimentally successful mapping is built
into the search.

The resource model charges:

* Independent transmit-lane payload per C600 tile, with explicit partner-lane
  occupancy for paired transfers.
* Independent per-tile receive payload.
* Combined send/receive service on each SRAM element.
* Ordinary and eligible paired-transfer alternatives, choosing the lower modeled
  phase bottleneck. Pair eligibility checks payload size/alignment, complete
  destination pairs, matching receive addresses and the borrowed source lane.

Each phase estimate is its maximum resource load plus the existing phase cost.
The ordinary whole-program timeline estimator combines those phase prices with
compute, copy and repeated-region work. A load-weighted mean endpoint pressure
breaks ties between mapping candidates, but a mapping must improve the modeled
bottleneck before it earns an exact-scheduling evaluation. Repetition counts
weight mapping scores; this is not hardware autotuning.

After that model ranks complete plans, only `exchange_schedule_finalists` are
physically scheduled (default one). Each may have one mapping challenger,
compared with its identity mapping. Exact schedules decide whether to accept
the challenger. Final SRAM placement and its existing bounded refinement occur
after runtime/kernel support storage is known. Explicit `--tile-mapping` input
overrides automatic mapping search.

## Limits

This model captures resource load, not the exact ordering of multicast
hyperedges or the route-specific feasibility of every paired transfer. It uses
provisional SRAM addresses and representative iteration traffic. Final schedules
still check full encoding and repeat-address constraints.

In particular, the known 92-tile-group mapping's small MLP improvement does not
reduce this model's maximum resource load. The model therefore does not select
it automatically. A first experiment allowing tiny balance-only gains selected
a worse mapping, which exact scheduling rejected. That is why balance alone
does not justify another scheduling pass. Recovering such small wins needs a
better ordering/contention model or the later autotuning facility.

This change introduces no hardware-based search, new tensor layout layer, or
per-tile details in mid. The placement resource model lives beside physical
exchange preparation and shares its fragmentation code.

## Corrected transmit-lane assumption

The previous compact model incorrectly summed adjacent tiles' ordinary sends as
one shared four-byte-per-cycle resource. The physical scheduler only reserves a
neighbor's transmit lane for paired transfers. The hardware-validated control
profile provides a concrete counterexample: physical tiles 0 and 2 both send
ordinary multicast traffic in phase 0, over intervals [25, 4173) and [24, 4172).
Their 4,147-cycle overlap rules out that serialization assumption. Both compact
traffic accounting and the new placement model now charge ordinary lanes
independently. This also removes false mapping gains from merely separating
ordinary senders that already can transmit concurrently.
The entire measured exchange after the last barrier arrival lasts 7,332 cycles,
less than the 8,296 cycles those two sends alone would require if serialized.

## Validation

The automatic SigLIP MLP runs in **172,236 renderer cycles**, versus 178,782
before these changes (3.66% faster). It is within 78 cycles of the manually
swept 172,158-cycle combination. The chosen compute grids are `2x92x8` up and
`3x18x27` down; it keeps the identity tile mapping. The compact model ranks it
fourth, while the expanded traffic model ranks it first. Final SRAM placement
removes another 1,014 scheduled exchange cycles.

The MLP numerical check passes with maximum absolute error 0.015625. Attention
smoke also passes (maximum error 0.000113). The codegen suite passes 110 tests
with one ignored, and workspace Clippy passes with the existing complexity
exceptions. The independent-lane regression also schedules two ordinary sends
from adjacent tiles and checks that they overlap.
The build-only verification after correcting lane accounting produced the same
package SHA-256 as the measured build:
`abc4356557b5d0747f27b40e6c07049d416ae0332ba968686f7108738d93501e`.
No additional hardware run was needed for that identical package.

Logs, the exchange snapshot and the rendered MLP profile are under
`artifacts/layout-sweep/modelled-planning/`. The measured profile is
`final/profile.html`. Earlier exploratory model versions and their results are
retained separately there; they are not the final planner's performance.

## Exchange table budget

Two independent limits apply: `exchange_transfer_limit_per_tile` defaults to
16,384 static endpoint fragments per tile before scheduling;
`exchange_table_budget_bytes` defaults to 64 KiB of actual encoded tables per
tile. Neither converts a heuristic byte estimate into a transfer count.
The mid estimator's 256-byte payload assumption is only a cycle/ranking
heuristic; it never rejects a plan against either limit. Operator and beam
shortlists preserve low-exchange candidates and refresh prefix estimates before
pruning, including conversions and cached operator fragments.

After tile expansion and before placement/mapping search or scheduling, the
screen matches concrete source/destination byte spans and splits contiguous
spans at the ISA transfer-size limit. It accumulates TX and RX fragments per
tile across static phases, then takes the maximum. It does not sum maxima on
unrelated tiles or combine neighboring tiles' transmit lanes for storage.
Multicast counts the sender once and each receiver separately. Repeat execution
counts do not multiply the stored body. For example, an aligned contiguous
8 KiB transfer counts once, not as 32 synthetic 256-byte fragments.

The pre-scheduling count is conservative: coalescing and paired/bidirectional
encoding can combine fragments. It is a complexity limit, not a byte estimate.
In particular, the old 36-byte uncompressed row allowance does not participate
in acceptance. Both provisional and final compact encoded tables must fit the
separate byte budget.
This is separate from the total tile SRAM budget and is not a global
transfer-count cutoff that penalizes distributing work across more tiles.

If package selection exhausts the exchange budget, it retries compact planning
with search penalties of 16, then 256 cycles per heuristic table byte. The
penalty changes prefix retention, not reported execution cycles or the hard
cap. Retries share cached operator implementations. They now respond to
geometry/encoded-size rejection, not the mid heuristic. This remains a bounded
search and cannot prove no feasible layout exists.

The benchmark exposes `--exchange-table-budget-kib N` and
`--exchange-transfer-limit-per-tile N`. API callers can set either limit to
`u64::MAX` to disable it, and `exchange_table_cost_per_byte`
to choose an initial ranking penalty. Capture examines the requested finalist
without package selection's automatic retries. Logs report heuristic and
geometry footprints separately.

Validation: full FP8 B2 ViT now completes mid planning, where it previously
stopped at operation 19. Its captured first finalist has approximately 12,500
geometry-derived endpoint fragments on the busiest tile, above the former
8,192-fragment default but below the increased 16,384-fragment default. This run did not schedule exchanges or execute hardware;
it does not establish that another finalist or retry fits both limits.
Regression coverage includes a contiguous 8 KiB span, disjoint phase owners,
Repeat reuse, fragment-limit boundaries, and independent encoded-byte checks.

## Cast ordering and packing alternatives

Eligible FP16-to-FP8 operator inputs offer both orders of casting and
redistribution before the existing region-beam prune. Choices are independent
between inputs and operations; automatic host bindings avoid duplicate choices
when the host can supply the requested physical layout directly. Prescribed
Repeat boundary conversions use redistribution followed by casting. Available
reusable quantizations participate in beam equivalence; reuse itself remains
unchanged. No new global policy or retry was added for cast ordering.

The distributed packing pass now constructs the four supported panel-size
variants (32, 64, 128, 256 rows) and retains their non-dominated whole-program
cycle/memory tradeoffs using the same `PlanMetrics` as the beam. Each variant
uses its chosen size for eligible packing operations; the original program is
also retained. The old local winner selection and cycles-only acceptance gate
are removed. This bounds the additions at four programs per original rather
than enumerating a Cartesian product across every packing operation.
