# Switching exchange bases without an extra synchronization

Superseded by [one schedule with timed outgoing-base changes](EXCHANGE_BASE_AWARE_2026_09_13.md).
The measurements below describe the earlier split-schedule implementation.

That scheduler could put fixed-address transfers and moving Repeat
sources into two timed sections of one exchange row. Both sections execute under
the original global synchronization. There is no intervening worker launch,
profile sample, or global barrier.

The first section uses OUTGOING_BASE = 0. Its rows finish their receive teardown
and memory hazards, then advance to a common boundary. The boundary loads the
second section's INCOMING_BASE and switches OUTGOING_BASE to the Repeat pointer
retained in supervisor m6. Independent scheduling can choose a different incoming
base for a tile, so changing just the outgoing register would be incorrect.

The existing memory-dependency analysis rejects separation when fixed-first
ordering would reverse a forwarding, overwrite, or read-before-write dependency,
including dependencies involving later Repeat source addresses. The same
scheduler, width selection, relocation code, and placement-replay cache serve
both sections. The combined schedule remains available: currently separation
must improve the scheduled horizon, including switch overhead, and reduce the
number of Repeat patch words. Patch execution time is not needed to justify
selection. This deliberately does not yet accept a slightly slower schedule in
exchange for larger patch savings.

## Timing validation

Hardware delay-substitution probes establish 13 timed cycles for an isolated
OUTGOING_BASE write, and 27 for SETZI followed by INCOMING_BASE and OUTGOING_BASE
writes. Treating the boundary as three ordinary six-cycle instructions would
be wrong. The exchange decoder now recognizes these instructions and includes
those costs. Including entry/alignment instructions, the two-section join adds
46 cycles to the sum of the separately scheduled horizons.

`ipu-e2e-test --workload exchange-stress --exchange-pattern base` exercises
moving -> fixed -> moving sections over three Repeat iterations, standard and
interleaved SRAM, point-to-point, multicast, paired transfers, and self-receive.
It also changes the point receivers' incoming base between sections. Each
iteration checks exact payloads; the final host readback checks 7,168 words.
Receive-only multicast tiles substitute 27 delay cycles for the sender's three
scalar instructions, checking timing as well as data relocation.

## BS1, 27-layer SigLIP

Same saved early-cast recipe, B1024 scheduling, random reference tensors, FP8 PV.
The control was built after the profiling-allocator change, avoiding conflating
that placement change with this optimization. Replaying its existing package
against the new build's saved inputs produced bit-identical embeddings in both
resident inference calls.

| Metric | Combined control | Timed sections |
|---|---:|---:|
| Runtime, cropped at the renderer's initial-sync boundary | 10,744,152 cycles | 10,528,050 cycles |
| Time at 1.5 GHz | 7.162768 ms | 7.018700 ms |
| Repeat patches per layer, across all tiles | 11,322 words | 365 words |
| Largest Repeat patch list on one tile/phase | 82 words | 8 words |
| Cross-phase sharing patches per layer | 14 words | 14 words |
| Largest sharing patch list inside Repeat | 1 word | 1 word |

Runtime improves **2.01%**, saving 216,102 cycles over 27 layers. The new build
passes both resident reference checks with minimum FP32-reference cosine
0.994352454. This is a randomized benchmark comparison, not a new pretrained
accuracy evaluation.

Final-address schedule comparison, including section switching:

| Phase | Combined cycles | Section cycles | Combined max row bytes | Section max row bytes |
|---|---:|---:|---:|---:|
| QKV weights/activations (8) | 16,972 | 13,874 | 2,604 | 2,552 |
| MLP upprojection (23) | 17,676 | 17,602 | 5,280 | 5,288 |
| MLP downprojection (27) | 15,957 | 14,424 | 6,492 | 6,612 |

All Repeat patches in these three phases are eliminated. Remaining patches are
in phases 10, 18, 20, 22, 25, and 29. No additional paired transfers were selected
in this layout. The package-support sizing pass reserves 43,732 exchange-table
bytes versus 43,684 in the control; phase maxima above are measured per tile,
not total traffic across the accelerator.

For the two MLP exchanges, measured time after the last tile enters the phase,
minus its scheduled horizon, falls from 1,572 to 278 cycles and from 2,049 to
294 cycles. These residuals include setup and synchronization as well as patching;
they are not isolated patch-loop measurements.

There is a compiler-time cost: the BS1 package-planning pass increased from
68.8 to 86.5 seconds at 16 Rayon threads. Mixed phases now evaluate the combined
schedule and two section schedules; their recipes are cached across placement,
but replay still rebuilds and validates each physical schedule.

## BS2 validation

The saved BS2 FP8-PV recipe also fits and passes two resident reference calls,
with unchanged minimum FP32-reference cosine 0.994151272.

| Metric | Combined control | Timed sections |
|---|---:|---:|
| Cropped runtime | 17,462,346 cycles | 17,198,316 cycles |
| Batch time at 1.5 GHz | 11.641564 ms | 11.465544 ms |
| Repeat patches per layer | 13,854 words | 1,049 words |
| Largest Repeat patch list | 100 words | 15 words |
| Cross-phase sharing patches per layer | 2,916 words | 2,916 words |
| Largest sharing patch list inside Repeat | 1 word | 1 word |

The runtime reduction is **1.51%**. Sections are selected for phases 16, 32,
and 34. Their maximum row-size changes are +92, -68, and +28 bytes respectively.
Package planning increased from 264.0 to 366.8 seconds at 16 Rayon threads.
The control is `artifacts/profile-allocation-20260913/bs2-pv/model.ipuexe`;
new package, reference outputs, patch audit, exact memory profile, and incremental
runtime profile are in `artifacts/exchange-base-sections-20260913/bs2/`.

## Artifacts and checks

- New package, resident tensors, raw and incremental HTML profile:
  `artifacts/exchange-base-sections-20260913/` (`model.html` and `model.data/`).
- Exact memory profile: `memory/placement-570361.html` and its data directory.
- Matched control package: `artifacts/exchange-overlap-20260913/model.ipuexe`.
  Its new hardware profile is `control.capnp` in the new artifact directory.
- `patches.json` and `control-patches.json` audit emitted helper calls and their
  descriptor tables. They include one-time head patching separately from Repeat.
- Unit tests check section dependency ordering, all decoded event horizons,
  transfer hazards, active/inactive tiles, row alignment, transfer indices, and
  relocated patch-word offsets. The full exchange crate tests also pass.

Final checks: 289 codegen tests passed (5 ignored), 52 exchange tests passed
(2 ignored), and `cargo check --workspace` passed.

## Forcing the rejected BS1 splits

A diagnostic build disabled only the horizon-improvement gate, retaining the
mixed-source, dependency, and patch-reduction checks. All six additional phases
were legal and eliminated their Repeat patches. Both resident checks passed
with cosine 0.994352454. Production source and the release test executable were
restored; this experiment does not change selection policy.

Measured exchange durations below run from the last tile's profile entry to the
last exit in the first profiled transformer layer. They include setup/patching
and synchronization, unlike the scheduled event horizon.

| Phase | Current measured cycles | Forced split cycles | Change | Scheduled change |
|---|---:|---:|---:|---:|
| 10, before QKV bias add | 2,526 | 2,550 | +24 | +21 |
| 18, attention output-projection GEMM preparation | 5,712 | 5,598 | -114 | +1 |
| 20, before output-projection bias add | 1,098 | 1,266 | +168 | +160 |
| 22, before MLP layernorm | 2,814 | 2,940 | +126 | +156 |
| 25, before BiasGeLU | 2,970 | 3,210 | +240 | +240 |
| 29, before downprojection bias add | 1,104 | 1,284 | +180 | +182 |

Five splits lose time even after accounting for patch removal. The combined
schedule overlaps small parameter traffic with activation redistribution; forcing
sections gives up some of this overlap. Particularly in phase 25, removing the
patches yields no visible reduction in the time beyond the scheduled horizon,
consistent with patch work already being hidden by other tiles' arrival waits.
The larger losses cannot be explained by the removable 16-cycle entry prefix.

Conversely, phase 18 shows the current gate is too conservative: a one-cycle
increase in its scheduled horizon accompanies a 114-cycle measured improvement.

Forcing all six makes the full cropped run 10,544,568 cycles / 7.029712 ms,
versus 10,528,050 / 7.018700 ms with current selection: **0.157% slower**.
It removes all remaining 365 Repeat patch words and 14 sharing patch words
per layer. Thus always separating these phases has a small total runtime cost
in this model, although separation is not uniformly beneficial.

Artifacts: `artifacts/exchange-separation-rejections-20260913/` contains the
captured combined/fixed/moving schedule comparisons, diagnostic executable,
original source copy, forced package and resident outputs, emitted patch audit,
and `hardware-comparison.json`. The source change for the diagnostic build was
`if false && ...` on the horizon comparison in `exchange/sections.rs`.
