# Exchange slack and larger batches

Baseline: `bf20e00`. Scheduler implementation: `7661483`; interval cursor: `1c63f98`.
Artifacts: `artifacts/exchange-slack-20260909/`.
The full model is the one-layer So400m/14 ViT with fused QKV and FP8 GEMMs,
378×378 input, 729 tokens, width 1152, MLP width 4304, and MAP.

## What the profile's idle time means

`barriers-b2.json` separates entry imbalance from the scheduled exchange itself.
These measurements precede the new scheduling policy:

| Physical phase | Scheduled cycles | Entry spread | Cycles after last entry |
|---|---:|---:|---:|
| 3 | 17,939 | 1,584 | 18,168 |
| 8 | 19,126 | 19,776 | 19,374 |
| 13 | 7,573 | 1,878 | 7,848 |
| 21 | 17,799 | 33,504 | 18,066 |
| 23 | 36,105 | 714 | 36,378 |

Thus phase 23 is predominantly a schedule problem. Phases 8 and 21 also have
large entry imbalance: improving their schedule cannot remove that component.
Phase 23 belongs to operation 21, the MLP down projection, between its GEMM
compute and partial-result reduction. It is not the GELU input redistribution.
The endpoint-word lower bound excludes control instructions and is not assumed
attainable. All new schedules still use the production encoder, timing rules,
SRAM hazards, Repeat dependencies, paired-lane reservations and validation.

## The ready queue was prioritizing completed work

The old queue used fixed endpoint word totals for point-to-point phases, and
combined each tile's send and receive traffic into one pressure value. Dynamic
remaining work was enabled only when multicast was present. As a phase drained,
the queue could still prioritize endpoints whose originally large workload had
already completed, while send and receive work also inflated each other's
pressure despite using independent resources.

The production policy now uses **remaining, directional pressure for entirely
point-to-point phases**. It keeps the existing remaining combined pressure for
phases containing multicast. The distinction is structural, without a transfer
count threshold, fitted coefficient, or additional candidate materializations.
Paired source-lane reservations count as send work in the directional model.
The existing matching and critical-neighborhood passes are unchanged.

This is intentionally narrower than using directional pressure everywhere.
That broader alternative regresses several multicast-heavy attention/ViT
phases. The captured attention's 21 phase horizons sum to 89,200 cycles before,
88,810 with directional pressure everywhere, and 87,566 with the selected
point-to-point-only policy. This sum is a capture comparison, not a measured
whole-model runtime.

## Controlled ordering experiments

The replay entry point accepts explicit combined, directional, remaining
combined, remaining directional, and address-ordered stream-wave policies.
`--select-widths --write-selected-snapshot PATH` first saves production-selected
ordinary/paired widths so subsequent ordering comparisons hold those fixed.
The two MLP phases below already select ordinary widths under current rules.
`vit-selected.json` similarly fixes widths for the current ViT candidate.

| Capture/phase | Combined baseline | Static directional | Remaining combined | Remaining directional |
|---|---:|---:|---:|---:|
| MLP 1 | 9,600 | 9,392 | 9,082 | 9,343 |
| MLP 3 | 15,700 | 15,660 | 11,565 | 9,993 |
| Older ViT B2 / 17 | 12,911 | 14,521 | 10,743 | 9,812 |
| Older ViT B2 / 31 | 5,641 | 13,972 | 5,641 | 5,223 |
| Older ViT B2 / 34 | 36,105 | 35,934 | 18,614 | 14,357 |
| Current ViT / 3 (multicast) | 17,939 | — | 17,939 | 17,234 |
| Current ViT / 8 (mixed) | 19,126 | — | 19,126 | 19,929 |
| Current ViT / 13 (multicast) | 7,573 | — | 7,573 | 8,165 |
| Current ViT / 21 (multicast) | 17,799 | — | 17,799 | 17,780 |
| Current ViT / 23 (point-to-point) | 36,105 | — | 18,614 | 14,357 |

Both updating remaining work and separating directions contribute. Static
directional pressure alone is not a general improvement.

The endpoint diagnostic makes the gap concrete. In the current phase 23's
baseline, a late receiver has 12,672 payload cycles and 23,351 internal gap
cycles (largest gap 7,465). A late sender has a 21,623-cycle gap. In the new
schedule, the critical receiver has 12,960 payload cycles and 1,337 internal
gap cycles (largest gap 32). It has *more* address discontinuities, 661 versus
285 on the old critical receiver. Reducing pointer changes is not the primary
cause of this speedup; selecting work for the appropriate congested resources
keeps the endpoints occupied.

## Stream waves: useful storage tradeoff, unsuitable universal default

The separate stream-wave experiment groups transfers by source and destination
set, orders each stream by address, and visits bounded payload waves in a
dependency-respecting order. It does not insert global barriers; the same row
builder overlaps independent transfers. Explicit 256- and 1,024-word waves were
tested. This is a different ordering algorithm, not a change to encoding rules.

| Current ViT phase 23 | Cycles | Maximum row bytes | Scheduling seconds |
|---|---:|---:|---:|
| Baseline | 36,105 | 8,228 | 34.24 |
| Remaining directional | 14,357 | 9,032 | 43.66 |
| 256-word stream waves | 18,955 | 5,260 | 18.50 |
| 1,024-word stream waves | 30,901 | 4,960 | 8.90 |

The stream alternative reduces row storage and compilation work, but loses
against remaining pressure in device cycles and regresses the captured MLP.
It remains an explicit offline option. It could be useful when choosing a
compact schedule for a memory-constrained plan; that selection is not integrated
into package placement in this change.

On the 1,172,736-transfer B4 phase 33, remaining directional pressure changes
73,044 to 28,668 cycles. Maximum row storage grows from 16,072 to 18,016 bytes.
CPU scheduling time changes from 79.87 to 131.54 seconds. These CPU times are
single samples from concurrent independent replays; modest differences are
not robust per-component speedup measurements. Every reported candidate passed
full schedule validation. No hardware timing constraints were relaxed.

## Batch 4 feasibility

The initial rerun with the baseline scheduler exhausted five admitted finalists.
Selection took 955,481 ms (15.9 minutes). Rejections were:

- Finalists 4 and 8: no room for a 12-byte host-data allocation after support.
- Finalist 50: a 49,152-byte interleaved allocation did not fit.
- Finalist 20: an 88,320-byte interleaved allocation did not fit.
- Finalist 16: a 70,288-byte standard allocation did not fit.

Other concurrent tile failures include the padded MLP partial allocation
`[5,4,729,4304]` and attention storage `[64,729,768]`. These are late support or
contiguous/class-constrained placement failures, not a demonstrated device
batch-size limit. Log: `vit-b4/run.log`.

The new scheduling policy also exhausts all five admitted finalists; batch 4
still does not build. Selection takes 1,257,935 ms (21.0 minutes):

- Finalists 4 and 8: the same 12-byte host-data failure.
- Finalist 50: a 122,880-byte standard allocation on tile 103.
- Finalist 20: a 70,288-byte standard allocation on tile 207.
- Finalist 16: an 88,320-byte interleaved allocation on tile 0.

Log: `vit-new-b4/run.log`. Both full batch-four attempts precede the interval
cursor optimization below. The improved ordering can increase exchange-row
storage, and faster device schedules alone do not resolve these placement
failures. No batch-four hardware execution or profile was produced.

## Full-model validation

Times use the renderer's initial-entry cutoff, at 1.5 GHz. Encoder span is
operation 23's last offset minus operation 3's first offset.

| Batch | Previous cropped cycles | New cropped cycles | Previous encoder span | New encoder span |
|---|---:|---:|---:|---:|
| 1 | 650,028 | 626,850 | 459,030 | 438,366 |
| 2 | 965,118 | 918,432 | 713,736 | 670,560 |

Both pass reference validation, with maximum absolute errors 0.065186 and
0.101074. Batch one improves 3.6% overall / 4.5% over the encoder interval;
batch two improves 4.8% overall / 6.0% over the encoder interval. The batch-two
profile confirms phase 23 at 14,357 scheduled cycles, with unchanged 714-cycle
entry spread and 14,634 cycles after the last entry.

Rendered profiles are `vit-new-b1/profile.html` and `vit-new-b2/profile.html`.
The exact final batch-two transfer capture is `vit-new-b2/exchange.json`. Export
mode exits before hardware execution; a subsequent normal reference build
produced a byte-identical package (see `export-package.sha256`) and performed
the hardware run. No identical hardware invocation was repeated.

The full codegen suite passed 208 tests, with five ignored. Tests compare the
lazy/grouped heap with eager priority under all queue policies and validate
randomized stream orders, Repeat source addresses, semantic dependencies and
encoded rows. Clippy passed with the repository's existing complexity allowances.

## CPU cost: monotone interval search

Profiling the remaining-directional B4 replay (`remaining.perf.data`, 3,757
samples) attributes 46.3% of samples to earliest-offset search, 10.3% to the
ready heap, and 6.9% to the greedy scheduling body. Instruction-level annotation
identifies repeated binary searches through the sender history as the hot loop.
After each occupied interval, the old search jumped to its end and binary-
searched the entire history again. The cursor implementation performs the
initial search once and advances monotonically through subsequent intervals.
Receive-control collision checks and all offsets remain unchanged.

The B2 phase-23 replay retains row fingerprint `f21f91b7851e6013`, all row sizes
and its 14,357-cycle horizon. Its single measured CPU sample changes from 43.66
to 34.96 seconds. On the B4 million-transfer phase, the cursor changes 131.54
to 62.09 seconds, preserving fingerprint `d98ab6f46600e322`, all rows, and
the 28,668-cycle horizon. This is also below the original static-policy
79.87-second sample. The new exhaustive-slot oracle checks minimal offsets across
random histories, controls, starting offsets and reserved sender lanes. All
51 exchange tests pass, with two ignored. The cursor does not change the
hardware schedule and is validated separately from the ordering policy.
All 21 phases of the captured attention also preserve their row fingerprints
and horizons exactly under the cursor change.

The diagnostic now also reports initial endpoint delay and out-of-order
payload insertion. Neither the tested MLP phase 3 nor ViT phase 23 inserted
backfilled sends/receives; their measured gaps are not an artifact of sorting
the diagnostic output.
