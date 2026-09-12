# Gather, packing and multicast — 12 September 2026

Explicit local packing is expensive if concentrated on a few tiles. A better
variant for these FP8 activations is to receive directly into packed relay
buffers, then multicast those buffers. It needs neither local packing kernels
nor mid-exchange instruction patching. The ordinary scheduler can retain the
receive-to-send dependencies within one exchange phase.

## Controlled comparison

Input: the final batch-two SigLIP capacity-baseline capture from
`artifacts/exchange-packing-20260912/siglip-traffic.json`. GEMM grids, original
source addresses (including all 27 Repeat iterations), final receiver tiles,
destination addresses and payload bytes remain fixed. All replays use ordinary
32-bit transfers and balanced B1024 stream scheduling. No paired-width search
or topology optimization was added.

The experiment splits each operand replica group across a configurable number
of distinct relay tiles. A relay is neither an original sender nor a final
receiver of its own segment, avoiding any reliance on loopback. Relay buffers
avoid the captured reads and writes. This establishes independence within the
captured phase; it does **not** establish coexistence with every other live
allocation in the full model.

The first exchange gathers one copy of the operand onto the relays. Two variants:

- **Local pack:** receive contiguous source runs into standard scratch; use the
  existing `copy_strided_u64` helper to rearrange into interleaved scratch;
  multicast the packed result in a second exchange.
- **Direct receive:** receive fragments at their final packed offsets in relay
  scratch; multicast contiguous covered intervals. This retains more small
  unicast transfers in the gather but eliminates the local copy. Padding holes
  are preserved, not overwritten by uninitialized staging bytes.

In the second variant, gathering and forwarding can be one physical exchange.
Concatenating their transfer lists expresses the receive-before-send dependence;
production dependency analysis and scheduler validation check the resulting row.

## Local packing cost

Packing was executed on hardware using the actual affine task lists and the
production strided-copy helper, with standard input and interleaved output.
The fixture uses dedicated addresses clear of its host/runtime support; these
are not full-model placements. Timing spans all helper calls on each tile,
including launch and loop overhead. Every measured output passed bytewise
comparison. Each distinct task configuration was run once.

| Relay segments per replica group | QKV packing cycles | MLP-up packing cycles | MLP-down packing cycles |
|---:|---:|---:|---:|
| 1 | 23,316 | 33,144 | 18,960 |
| 2 | 11,706 | 16,620 | 9,528 |
| 4 | 7,170 | 8,358 | 4,812 |
| 6 | — | — | 3,240 |
| 8 | 4,230 | 4,224 | — |
| 16 | 2,520 | 2,460 | — |
| 32 | 1,170 | 1,422 | — |

MLP-down uses 240 replica groups; six relays per group already occupy 1,440
tiles. QKV and MLP-up have 40 and 27 groups, allowing more subdivision.

Adding separately scheduled gather and multicast horizons to measured packing
gives the following **stage-sum estimates**, before extra barrier/setup cost:

| Activation preparation | Existing exchange | One relay/group | Best tested local-pack variant |
|---|---:|---:|---:|
| QKV | 11,426 | 44,963 | 13,716 (32 segments) |
| MLP-up | 16,894 | 65,219 | 19,166 (32 segments) |
| MLP-down | 8,729 | 34,338 | 14,774 (6 segments) |

These sums are not measured end-to-end execution. Per-tile completion differences
can overlap part of the packing with the exchange tail, while the additional
exchange boundary adds synchronization/setup cost. In particular, the original
QKV and MLP-up exchanges also contain weights; the isolated activation figures
above must not be presented as the cost of those full phases.

## Direct packed reception and forwarding

At natural panel boundaries, receiving one batch plane of a column panel places
the data directly in the order needed for the subsequent multicast. QKV uses
18 relay segments per group; MLP-up uses 24. Down uses six larger segments with
strided reception. The latter also avoids copying, but has a less favorable
gather schedule.

For **activation traffic alone**, forwarding in one exchange gives:

| Movement | Existing cycles | Forwarding cycles | Change | Maximum timed-row bytes, old → new |
|---|---:|---:|---:|---:|
| QKV | 11,426 | 11,756 | +330 (+2.9%) | 12,008 → 888 |
| MLP-up | 16,894 | 17,039 | +145 (+0.9%) | 17,176 → 1,060 |
| MLP-down | 8,729 | 11,290 | +2,561 (+29.3%) | 9,684 → 3,428 |

All figures in this section are **scheduler/codegen results**, not hardware
timings. There is no added compute phase or global barrier in the merged variant.
For down, two separate direct-receive/forward phases total 9,992 scheduled cycles
before their extra synchronization/setup; the merged stream ordering is worse.

Including the **unchanged weight traffic** in the actual QKV and MLP-up phases
changes the result favorably:

| Complete captured phase | Existing cycles | Forwarding cycles | Change | Maximum timed-row bytes, old → new |
|---|---:|---:|---:|---:|
| QKV input + weights, phase 16 | 27,245 | 23,913 | −3,332 (−12.2%) | 11,992 → 944 |
| MLP-up input + weights, phase 50 | 28,376 | 26,420 | −1,956 (−6.9%) | 17,788 → 1,584 |

The original activation and weight addresses remain unchanged in this comparison.
Relay scratch avoids both sets of accesses, including every parameter iteration.
All schedule invariants pass. These are per-phase timed-row sizes, **not** final
whole-package table allocations after sharing, patch tables and placement.

| Movement | Relay tiles | Maximum additional relay buffer |
|---|---:|---:|
| QKV | 720 | 2,368 B |
| MLP-up | 648 | 2,624 B |
| MLP-down | 1,440 | 4,864 B |

The direct variant needs one buffer per relay; there is no separate gather and
pack buffer. It adds one unreplicated operand's worth of receive traffic. Small
strided receives now happen on the relays once, rather than independently on all
36/54/6 replicas. The final receivers consume larger contiguous transfers. This
explains the table reduction without relying on instruction livepatching.

## Limits and next decision

This is a promising transport alternative for QKV and MLP-up, not a blanket rule
to insert local packing. It must remain an explicit choice against direct
multicast, with scratch, exchange rows and execution cost included. An ordinary
identity-copy composition pass would otherwise erase the relay and recreate the
original exchange.

The full model has not been rebuilt or run with these relays. Whole-model live
allocation compatibility, final row sharing/patching, and hardware timing of
the forwarded exchange remain to be validated before claiming a model speedup.
Relay selection here is deterministic and simple; it is not a search optimum.

## Reproduction

`scripts/exchange-gather.py` generates the alternative snapshots and packing
tasks, then uses the existing exchange benchmark for replay. For example:

```sh
python3 scripts/exchange-gather.py \
  artifacts/exchange-packing-20260912/siglip-traffic.json \
  --output artifacts/exchange-gather-20260912/full24 \
  --phase 50 --shards 24 --direct-receive --include-repeated \
  --scheduler target/release/ipu-exchange-schedule-bench
```

Omit `--direct-receive` for local packing. `copy_check` consumes the resulting
`copies.json`, measures every nonempty packing case, and checks the bytes. It
accepts at most 1,472 cases per invocation; the 32-segment QKV and MLP-up cases
were measured separately. Direct relays have no packing kernel to benchmark.

Results are under `artifacts/exchange-gather-20260912/`: `s1` through `s32` for
local packing, `direct18`/`direct24`/`direct6` for activation-only forwarding, and
`full18`/`full24` for the complete phases. `hardware*/cycles.json` contains the
measured packing cycles. The original root-level preliminary phase-50 broadcast
used an aliasing scratch address; it is superseded by the checked `s1` result
and is excluded from every table above.

Validation: nine Python reconstruction/analysis tests pass, including symbolic
source-word reconstruction through gather, packing and multicast with padding
holes. Rust fixture compilation and Python lint pass. No production compiler or
runtime behavior was changed by this experiment.
