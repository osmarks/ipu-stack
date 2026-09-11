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
