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

`PipelineConfig::exchange_table_budget_bytes` defaults to 64 KiB per tile.
The beam rejects partial programs above this budget before retaining the next
beam. Exchange storage remains a Pareto dimension, so lower-exchange alternatives
compete alongside fast ones. Forced GEMM layouts do not bypass this limit.

This is deliberately a conservative complexity policy, not a proof of memory
infeasibility: the existing estimate sums phase maxima and charges 36 bytes per
transfer fragment, plus phase headers. Thus the default permits roughly 1,820
worst-tile fragments across a static program, fewer after headers. It can reject
programs whose compact encoded tables would fit. It does not impose a global
transfer-count cutoff that penalizes distributing work over more tiles.

After tile expansion, the same footprint calculation uses concrete byte-span
fragmentation and rejects excess before placement/mapping search and scheduling.
Both provisional and final compact encoded tables must also fit the budget.
Repeated execution counts the stored body once, not once per iteration.

The benchmark exposes `--exchange-table-budget-kib N`; API callers can set the
field to `u64::MAX` to disable the policy. This is separate from the total tile
SRAM budget. Raising it allows more expensive exchange layouts and may bring
back long scheduling attempts followed by package-placement failure.

Validation (2026-09-08): the full FP8 batch-2 ViT capture command now rejects at
mid operation 15: smallest surviving estimate 69,828 bytes versus the 65,536-byte
budget, before any exchange scheduling. This prevents the previous 105,852- and
119,756-byte table-placement failures, but does not establish a feasible B2 plan
under the new default. Regression coverage checks a lower-exchange layernorm
alternative, late fragmentation rejection, budget boundaries/override, and
unchanged static footprint when increasing Repeat execution count.
