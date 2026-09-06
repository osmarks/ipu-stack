# MLP layout alternatives measured after the sweep

These experiments use the upgraded GELU/reduction kernels and the historical
`4x92x4` up / `4x24x15` down compute grids. Down weights use standard input
storage, materialized into interleaved compute storage. Timings are the profile
renderer's cropped interval. The control reproduces **175,140 cycles**.

## Rectangular result ownership

Result factors multiply the compute grid's row and column partitions; they do
not change GEMM compute partitioning. Explicit constraints now admit mixed
factors and partial use of the K-way result distribution. Automatic candidate
generation is unchanged during these experiments.

| Change from control | Cycles |
| --- | ---: |
| Down result `5x3` instead of `15x1` | **173442** |
| Down result `4x3` | 176652 |
| Down result `7x2` | 177558 |
| Down result `3x3` | 179214 |
| Up result `2x2` instead of `4x1` | 179274 |
| Both up `2x2` and down `5x3` | 177570 |

The best mixed result reproduces 173,442 cycles in two further builds/runs:
a **1,698-cycle (0.97%) improvement**. It changes the down result grid from
`60x24` to `20x72`. The final reduction exchange falls from 7,991 to 5,941
scheduled cycles; the other three exchange horizons are unchanged. Its
endpoint lower bound only falls from 4,704 to 4,560: most of the exchange
saving is a smaller gap above that bound, rather than reduced endpoint work.

The up `2x2` result does improve the large gather, from 31,382 to 28,065
scheduled cycles. However, its preceding reduction exchange worsens from
3,779 to 8,944, and the whole program loses. Optimizing that gather alone
would choose the wrong result ownership.

The compact estimate prefers the winning down result to control (258,236
versus 259,590). This experiment establishes a useful candidate, not that
unconstrained shortlisting will retain or select it. Finer K ownership and
additional grid orders were not tested here.

## Physical tile mapping

Package construction accepts a bijection from planned tile indices to execution
tile indices. It is applied to expanded shard ownership and work before
per-tile projection, SRAM placement, and physical exchange scheduling. Tensor
coordinates and local storage orders are preserved. Mid remains a description
of whole-device operations. Invalid mappings are rejected before mutation.

Eight global transpose embeddings were tested. All lost: the best was
178,116 cycles, and the worst 189,816. Follow-up experiments transpose within
up/down compute-row groups and combine mappings with the mixed result.
One smaller mapping wins: transpose a `46x2` index grid within each consecutive
92-tile group. Precisely, planned tile `t` maps to
`92*(t/92) + 46*(t%2) + (t%92)/2`, using integer division. This runs at
174,132 cycles on the original result layout and **172,158 cycles** with the
mixed down result, a combined **1.70% improvement** over control.
Two confirmation runs both reproduce 172,158 renderer cycles. Their full
counters differ by six cycles (178,962 and 178,968), which the crop removes.

For the combined winner, the four scheduled exchange horizons are
`[7048, 3755, 30716, 5860]`, versus control's `[7048, 3779, 31382, 7991]`.
The mapping improves the large gather without increasing its endpoint lower
bound. The other tested within-group permutations and global/mixed combinations
lose to the unmapped mixed result. These are fixed permutation experiments,
not an implemented traffic-driven topology optimizer.

One instructive global mapping (transpose width 92) leaves the first exchange
almost unchanged, 7,048 to 7,018 cycles, but increases the large gather from
31,382 to 39,504. Its endpoint lower bound rises from 23,544 to 33,408.
The regression is therefore not merely a larger unexplained scheduling gap;
the mapping creates a worse bottleneck in the scheduler's endpoint model.
These experiments do not rule out a traffic-aware, nonuniform permutation.

## Bounded reduction groups

`ReductionStaging::Batched(N)` receives up to N remote partials per epoch,
reusing the packed buffer and alternating accumulators. `N` is nonzero; the
last group can be shorter. Both mid costing and tile expansion use the same
batch limit, and costing charges the shorter final kernel separately.

| Down remote partials per epoch | Cycles |
| --- | ---: |
| 14 (complete/control) | **175140** |
| 7 | 179382 |
| 4 | 190800 |
| 2 | 207654 |
| 1 | 235032 |

For seven partials per epoch, the final exchange becomes two exchanges of
5,132 and 4,979 cycles, versus one of 7,991. Extra kernel launches and barriers
also contribute. Up batches of two and one run at 182,070 and 188,214 cycles.
Bounded staging is useful as a scratch-memory tradeoff, but does not improve
this memory-feasible MLP. Trees and pipelined panel reductions were **not**
implemented or measured; the batching result does not rule them out.

## Validation and reproduction

All executable cases pass the full MLP numerical check (maximum absolute
error 0.011719). The initial sweep has 22 passes and three no-candidate
rejections: requested column factors exceed the available local column groups.
The follow-up has 20 passes and confirmation adds two: **44 numerical passes
and three build rejections across 47 cases**, with no hardware failures.
Unit coverage exercises unequal contributor counts, a short final batch,
single-contributor groups, mapping preservation, and invalid mappings.
The codegen suite passes 109 tests with one ignored; workspace Clippy passes
with the existing complexity exceptions.

```sh
python3 scripts/mlp-layout-alternatives.py --sdk "$SDK_PATH" \
  --output artifacts/layout-sweep/alternatives-v1 --jobs 16
python3 scripts/mlp-layout-alternatives.py --sdk "$SDK_PATH" \
  --output artifacts/layout-sweep/alternatives-followup --followup --jobs 16
python3 scripts/mlp-layout-alternatives.py --sdk "$SDK_PATH" \
  --output artifacts/layout-sweep/alternatives-confirm --confirm --jobs 16
```

Builds run concurrently; hardware ownership uses the shared sweep device lock.
Each cohort preserves its executable, binary hash, constraints, mappings,
profiles, numerical results and per-phase exchange horizons. The initial
control and winner have rendered `profile.html` files in their case folders.
The combined winner's rendered profile is
`artifacts/layout-sweep/alternatives-followup/mapping-block-92-2-mixed/profile.html`.
Completed cohorts resume without recompilation or hardware execution; the
driver rejects changed binary identities or case definitions.

The practical priority is mixed result factors and narrowly scoped tile mapping,
with complete reductions retained for this workload. Both winners fit the
existing tensor representation. Simple global topology changes and smaller
receive batches have not justified automatic search expansion or a broader IR
rewrite.
