# Copies between exchanges

## Dependency audit

The canonical batch-1, 27-layer FP8 ViT with fused QKV was expanded without
placement or exchange scheduling. The audit compares local-copy byte ranges
against exchange source/destination traversals, resolving allocation aliases.
It retains strided-copy rows rather than treating their enclosing span as written.
This is the canonical graph, not a reconstruction of the optimized runtime
profile; its phase IDs and counts differ from that profile.

| Operation | Exchange boundaries | Local copies | Movable before preceding exchange |
|---|---:|---:|---:|
| Input projection (0) | 1 → 2 | 4 | 4 |
| QKV projection (4) | 10 → 11 | 3 | 3 |
| Attention output projection (14) | 31 → 32 | 3 | 3 |
| MLP up projection (18) | 40 → 41 | 1 | 1 |
| MLP down projection (21) | 47 → 48 | 22 | 22 |
| MAP KV projection (26) | 56 → 57 | 1 | 1 |

These copies finish a weight-layout materialization. The preceding exchange
writes remote portions of its destination; the local copies write disjoint
portions. The following exchange reads those local portions to replicate the
materialized weights. Thus they cannot move later, but can move earlier. Whole
allocation dependency checks incorrectly report a preceding write/write hazard.
For example the MLP-up copy transfers seven 512-byte rows into rows spaced by
3,072 bytes. It can move earlier despite being strided.

The existing `append_exchange_phase` only tries pulling the new exchange ahead
of intervening copies, requires whole-allocation disjointness between exchanges,
and restricts fusion to matching operator provenance. A suitable change is:

1. Check intervening copies for movement in either direction, preserving copy
   order and all byte-range read/write hazards, including aliases.
2. Merge the transfer lists in original order. The exchange scheduler already
   computes overlapping memory dependencies and can order receive-then-forward
   traffic. Do not require the two exchange lists to be independent.
3. Preserve that order through transfer preparation/coalescing, and check the
   resulting schedule and placement. Merging phases can extend simultaneous
   allocation lifetimes even when moving the copies themselves does not.

This removes the intervening *compute phase*, not the necessary byte movement.
Eliminating movement itself would mean composing the weight-layout conversion
with replication, or retaining a compatible local alias. Mid copy composition
currently deliberately preserves conversions before increasing replication to
avoid repeating packing on every receiver. Removing that rule indiscriminately
would trade these few local copies for potentially much more packing work.

Not every boundary permits moving all copies together. The head's 71 → 72 and
85 → 86 groups contain copies that actually read preceding exchange results;
these require splitting the group or a different transformation. The audit is
feasibility evidence, not a measured speedup or an implemented fusion pass.

Artifacts: `artifacts/copy-boundaries-20260911/precise-expansion.log` contains
`COPY_BOUNDARY` JSON records, including per-copy hazards; `boundary-audit.patch`
contains the temporary instrumentation. `precise-expansion.json` records the
baseline expansion. No diagnostic instrumentation remains in production code.

## Paired self-only delivery

The previous failed self-only probes used ordinary mode. This investigation
also tried paired mode, temporarily bypassing the incomplete-pair and self-only
software guards. Source was `0x65000`, destination `0x60000` (standard) or
`0x98000` (interleaved), on logical sender 0 and sender 1, with 2, 64 and 512 words.
The paired encoder used both send directions (`sctl=7`).

None of the twelve self-only probes passed. Standard destinations on both
senders and interleaved destinations on sender 1 retained their initial values;
sender 0's interleaved cases stopped with an address exception. The scheduler's
modeled receive events therefore do not establish that the hardware received.

As controls, four 512-word probes added the sender's partner as a real receiver,
with an independent destination pointer. Both source lanes and both destination
memory classes passed, checking 2,048 source/destination words per probe. Their
scheduled event horizon was 315 cycles. This reconfirms that paired own-pair
loopback works; isolated paired self-delivery needs more than lifting guards.

One plausible missing piece is partner-side mux setup: paired receive code gives
one member the XPIC source-selection stream. Testing a partner that executes
only these controls, without a memory receive, remains an experiment; it was
not established by these probes. A dummy full receive would require real scratch
storage and must not overwrite the partner's live data.

The ordinary and paired restrictions remain unchanged. Probe snapshots, logs,
and `paired-self-probe.patch` are under `artifacts/copy-boundaries-20260911/`.

## Implementation

`low/expand/exchange_grouping.rs` now checks both legal movements for a whole
copy group: leave it after the combined exchange, or move it before the preceding
exchange. The check indexes relevant allocation roots, distinguishes reads from
writes, and compares actual byte spans with affine copy rows. It does not expand
strided copies into word operations. Copies retain their relative order; kernels,
checkpoints, Repeat boundaries and the existing operator-provenance boundary
remain barriers to this transformation.

Merged transfer lists retain their original order. The existing physical
scheduler supplies byte-range dependencies, including receive-then-forward
chains. Transfer coalescing now checks every Repeat source binding before joining
contiguous messages, so it cannot erase a loopback memory dependency.

The canonical 27-layer graph drops from 92 exchange phases to 86, with unchanged
local-copy and logical-transfer counts (50,051 and 621,754). Its new expansion
measurement is 6.46 seconds; this was not an isolated compiler-speed benchmark.
Codegen validation passed 234 tests (four ignored), plus focused copy-motion
tests after extending shared-read coverage. The workspace/all-targets check passed.

A B1024 hardware replay passed a six-transfer receive-then-forward chain over
four tiles, checking all 8,192 touched words. The captured inputs and log are
`artifacts/copy-fusion-20260911/forwarding.json` and `forwarding.log`.

### Scheduling and Repeat corrections

The first full fused build was numerically correct (FP32-reference cosine
0.994168165 on all three resident invocations), but took 14,458,956 cropped cycles
(9.639304 ms), slower than the preceding 13,291,890-cycle run. Two interactions
needed correction before treating fusion as a performance improvement:

- Compact stream ordering could place a newly ready forwarder before the input
  traffic of other forwarders. Its late payload then occupied receiver-row
  timelines far into the future. Compact ordering now prioritizes dependency
  depth before stream-wave rank; actual exchange events may still overlap.
  This reuses the scheduler's memory-dependency DAG. Independent phases retain
  their original ordering.
- `_BASE` selection rejected an entire phase when any sender mixed stationary
  and moving sources. Arithmetic patching grew from 5,817 to 317,457 words per
  layer, with the largest list growing from 24 to 236. Selection is now per tile,
  choosing a common changing displacement and encoding exception patches
  relative to that base. Stationary sends receive inverse-displacement patches;
  the many weight sends keep constant encoded offsets. The base remains fixed
  during each phase. Every iteration must have nonnegative representable offsets;
  unsuitable tiles retain ordinary absolute-address patching.

At the earlier capture's fixed placement, the four relevant phase pairs have
these B1024 schedule horizons (excluding barrier/setup/patch time):

| Pair | Separate, summed | Naively fused | With dependency depth |
|---|---:|---:|---:|
| QKV, 9 → 10 | 18,977 | 30,549 | 15,948 |
| Attention output, 21 → 22 | 6,789 | 13,801 | 7,584 |
| MLP up, 29 → 30 | 15,641 | 23,456 | 16,042 |
| MLP down, 36 → 37 | 11,371 | 23,017 | 14,751 |

These comparisons preserve all captured Repeat addresses. They are scheduler
measurements, not predictions of complete runtime savings. Removing a barrier
also changes setup and overlap with compute. Giving every fragment of a stream
its maximum dependency depth made no further difference on these pairs and was
not retained.

With a common moving base, the captured MLP-down phase requires only 486
exceptional sends instead of 302,920 changing sends, with at most two exceptions
per tile. This is a send count, not a packaged patch-word count. Tile-local
selection alone would not solve the critical-path problem; exception patching
is the essential part.

The corrected two-layer ViT passed three resident hardware invocations with FP32
reference checking. This also exercises the table-patching path used for two
iterations. The updated codegen suite passes 235 tests (four ignored).

The unoptimized two-layer package still has a maximum 169-word Repeat table
patch list, near the MLP-down reduction/residual boundary. The fixed-placement
two-exception result above therefore must not be generalized to arbitrary
layouts or to every phase. The optimized 27-layer package needs its own audit
of the maximum per-tile patch list, as well as an end-to-end timing comparison.

### Full-model validation and base-selection correction

The next optimized 27-layer run (`artifacts/copy-fusion-20260911/final27/`)
passed all three resident invocations with the same FP32 cosine, 0.994168165.
Its cropped runtime was 13,343,658 cycles (8.895772 ms), versus 13,291,890
cycles (8.861260 ms) before fusion: still 0.39% slower. Exchange-table reservation
was 40,880 bytes per tile, versus 40,824. The rendered profile is `model.html`.

Repeat arithmetic patches totaled 9,707 words per profiled layer, with a maximum
107 on one tile/phase; row-sharing patches remained 1,828 words with maximum two.
The maximum was near the QKV reduction/bias boundary, not the fused weight
distribution. This exposed a selection error: choosing the most common *moving*
base ignored the stationary pattern. A few changing bias sends could cause many
previously stationary addresses to require inverse patches.

Selection now compares against zero base and requires strictly fewer patched
address words. It uses the actual encoded sender-address groups, including paired
restarts, rather than transfer counts. The groups are parsed once and reused for
patch generation. This retains mixed-base relocation where it saves work without
applying it to stationary-dominated phases. The 27-layer timing above predates
this final selection correction and must not be presented as its performance.

The corrected unoptimized two-layer run
(`artifacts/copy-fusion-20260911/repeat2-patch-selection/`) passed all three
resident invocations with FP32 cosine 0.997122946. Its maximum Repeat patch list
fell from 169 words to two; total patched words fell from 8,814 to 1,665.
Both packages have 1,256 Repeat patch helper calls in the profiled layer.
Cropped runtime fell from 1,627,302 to 1,614,030 cycles (1.084868 to 1.076020 ms).
This validates the final selection correction on hardware; an optimized
27-layer timing with that correction has not yet been measured.

### Grouping across operator labels and zero fills

The next extension removes the same-operator restriction. Exchange provenance
becomes neutral when phases from different operations are joined; individual
kernel provenance is retained. Repeat regions and checkpoints still delimit
motion. Explicit copies and `FillZero` kernels now share one affine read/write
range representation for dependency checks, including allocation aliases.
Other kernels remain motion barriers. Groups still retain their internal order.

The unoptimized two-layer hardware test in
`artifacts/exchange-boundaries-20260911/grouped2/` passed three resident
invocations with unchanged FP32 cosine 0.997122946. Its profiled layer has 37
exchange barriers, down from 42 with only the corrected base selection. Five
internal single-tile copy phases disappear; the Repeat entry/exit copies remain.
The maximum Repeat patch list remains two words. Runtime increased from
1,614,030 to 1,624,134 cycles (1.076020 to 1.082756 ms): several merged schedules
are longer than their separate predecessors. Fewer barriers alone do not
establish a speedup. Full-model comparisons are recorded separately below.

The codegen suite passes 235 tests (four ignored), including cross-provenance
copy/fill motion, aliases, shared reads and actual receive-write-send hazards.
The workspace/all-targets check also passes.

### Full-model comparison and retained implementation

Both optimized 27-layer builds passed three resident hardware invocations, with
FP32-reference cosine 0.994168165 and maximum absolute error 0.424608. They accepted
the same sequence of operator/boundary improvements. Timings use the renderer's
cropped range, not the entire profiling interval.

| Build | Cropped cycles | Time | Reserved exchange bytes/tile | Repeat patch words, maximum list | Profiled-layer barriers |
|---|---:|---:|---:|---:|---:|
| Corrected base selection | 13,281,234 | 8.854156 ms | 40,880 | 2,052 total / 18 maximum | 30 |
| Cross-operator copy/fill grouping | 13,391,934 | 8.927956 ms | 42,336 | 2,042 total / 16 maximum | 27 |

Artifacts and rendered profiles are respectively
`artifacts/exchange-boundaries-20260911/base27/model.html` and
`artifacts/exchange-boundaries-20260911/grouped27/model.html`.

The grouped version removes all three remaining internal single-tile copy
phases. Repeat entry/exit copies remain, as do the mixed padding/cast and
padding/packing preparation groups. Row-sharing patches fall from 1,828 words
to 14 words per profiled layer, but this is outweighed by longer exchanges.
The principal merged schedule horizons are:

| Boundary | Separate horizons, summed | Merged horizon |
|---|---:|---:|
| Q/K preparation | 4,464 + 6,856 = 11,320 | 11,704 |
| MLP-up preparation | 3,558 + 8,747 = 12,305 | 14,011 |
| MLP-down preparation | 5,448 + 10,141 = 15,589 | 17,928 |

These are the actually selected placed schedules, so the comparison includes
placement effects and is not an isolated scheduler benchmark. Their enlarged
horizons and changed per-tile completion times outweigh the barriers removed.
The full grouped run is 110,700 cycles (0.83%) slower and reserves 1,456 more
exchange bytes per tile. The two-layer run also regressed.

Consequently **the cross-operator/fill grouping experiment was removed from the
active tree**; its implementation remains in commit `9f04299`. The retained
implementation is the same-operator copy-motion pass plus dependency-aware
compact ordering and corrected mixed-base selection. Its validated profile is
`base27/model.html`. It is marginally faster than the pre-fusion 8.861260 ms run,
and fixes the 107-word patch regression of `copy-fusion-20260911/final27`.
Further fusion needs a way to select it using actual schedule cost (including
completion skew and setup), or an ordering algorithm that handles these merged
streams better; removing more dependency-check restrictions alone is insufficient.
