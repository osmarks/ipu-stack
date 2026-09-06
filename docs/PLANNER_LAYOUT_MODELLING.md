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

* Shared transmit-bus payload for adjacent C600 execution tiles.
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
