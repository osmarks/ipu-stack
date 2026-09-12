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

## Production integration

Implemented in `low/expand/relay.rs`, after ordinary tile expansion and before
costing/placement. There is one alternative to direct multicast, derived from
contiguous native panels and individual outer-axis coordinates. There is no
relay-count knob, new mid operation, kernel, scheduling mode, or package retry.
The existing cast-order and local-packing choices address different conversions
and remain unchanged.

The pass requires complete, nonoverlapping coordinate-preserving gathers, equal
replica extents, and enough distinct relay tiles. Existing receive/forward
dependencies and unsupported or incomplete panels stay direct. It never adds
padding initialization. Relay buffers are ordinary `ExchangeStaging` shards,
so allocation, liveness, Repeat handling and address patching see them normally.

Selection reuses the same geometry, cycle calibration and per-tile row accounting
as complete-plan costing. It accepts the alternative only if modeled cycles do
not increase and the worst tile's estimated row storage plus relay scratch is
smaller. This is a ranking rule, not a placement guarantee; final package
validation still accounts for the other live tensors and exact encoded rows.

The batch-two capacity baseline selects relays in phases 16, 34, 50 and 61.
Low expansion took roughly 10–13 seconds; the detailed whole-program row estimate
fell from 142,912 to 88,048 bytes and the fragment maximum from 15,017 to 10,445.

Hardware validation:

| SigLIP batch-two run | Encoded table maximum | Minimum cosine against FP32 |
|---|---:|---:|
| One layer, profiled | 61,148 B | 0.997085087 |
| Two layers through Repeat, unprofiled | 61,116 B | 0.996864709 |

Both run under the default 80 KiB exchange-table budget. The preceding one-layer
build needed 99,228 bytes and a diagnostic 128 KiB budget. The final placed
one-layer cycle estimate changes only slightly, from 1,389,300 to 1,385,584;
the principal verified improvement is table storage. The new profiled run spans
1,388,640 cycles (0.92576 ms) using the renderer's normal leading-interval cutoff.
There is no matching old profiled run here from which to claim a measured speedup.

The [rendered profile](../artifacts/relay-integration-20260912/profile.html),
package, profile capture and hardware logs are in
`artifacts/relay-integration-20260912/`. The complete codegen suite passes with
244 enabled tests and four ignored tests. New tests reconstruct bytes through
the selected relay graph and reject missing coverage, overlaps and dependencies.

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

Full batch-two builds and resident hardware validation now pass for both models,
as recorded below.
Relay selection here is deterministic and simple; it is not a search optimum.

## Full-model revalidation

Built from `5e02c88` with the normal 80 KiB/tile exchange-table and 16,384-fragment
limits. Both use the capacity baseline, zero local optimization steps, fused QKV,
FP8 F143 dense operands at scale −4, FP16 attention, and B1024 scheduling.
Profiling is disabled for these capacity tests; exact placement profiles are saved.
Host builds ran concurrently with 12 Rayon threads each; device access was serialized.

| Batch-two model | Result | Maximum encoded exchange storage | FP32-reference minimum cosine | Estimated device cycles | Measured resident batch latency |
|---|---|---:|---:|---:|---:|
| SigLIP So400m, 27 layers | PASS | 61,316 B/tile | 0.993968791 | 25,281,807 | 18.455 ms |
| PE L/14 capacity probe, 24 layers | PASS | 50,652 B/tile | 0.993687455 | 22,689,775 | 15.991 ms |

Each reference test uploads all layer parameters once and checks two consecutive
inference calls. Both calls pass with the same cosine; maximum absolute errors
are 0.411731 and 0.394768 respectively. These are randomized-weight numerical
tests, not pretrained-model accuracy tests. The PE probe still omits RoPE and
uses layernorm epsilon 1e−6 rather than 1e−5.

Resident latency is a separate replay of the saved input/weight/output fixture:
image upload, inference, and embedding download, excluding initial weight upload
and device attachment. Both returned outputs match the saved hardware output
exactly. The two SigLIP calls take 18.280 and 18.627 ms; PE takes 16.043 and
15.937 ms, with 100 µs host polling. Aggregate throughput is approximately
108.4 and 125.1 images/s respectively. These short host-timed samples are not
cycle-profile measurements.

SigLIP previously failed the exchange-storage limit; the complete model now
places and runs without increasing it. PE's previous full capacity build used
74,628 B/tile, versus 50,652 now (32.1% less). Its previous cycle estimate was
19,758,901, versus 22,689,775 now (14.8% higher): the storage improvement is
not a demonstrated speedup. There is no matched previous resident timing in
this comparison. Package planning took 132.3 s for SigLIP and 88.8 s for PE
while the two builds shared the host.

Artifacts: `artifacts/relay-big-20260912/{siglip-b2,pe-b2}/` contains each package,
`build-run.log`, `latency.log`, saved `resident/` fixture and memory profiles.
Exact placement: [SigLIP](../artifacts/relay-big-20260912/siglip-b2/memory/placement-466697.html)
and [PE](../artifacts/relay-big-20260912/pe-b2/memory/placement-466699.html).

Reproduce each build after sourcing `.env` (use `LAYERS=27`, `MODEL=()` for
SigLIP; `LAYERS=24`, `MODEL=(--vit-model pe-core-l14-capacity)` for PE):

```bash
out=artifacts/relay-big-recheck/siglip-b2
mkdir -p "$out"
RAYON_NUM_THREADS=12 RUST_LOG=info target/release/ipu-trivial-test "$IPU_CONFIG" \
  --sdk "$POPLAR_SDK_ENABLED" --runtime-source device/static_runtime.S \
  --workload siglip-vit-benchmark "${MODEL[@]}" --vit-layers "$LAYERS" \
  --vit-batch 2 --fuse-qkv --fp8-scale=-4 --capacity-baseline \
  --optimization-steps 0 --exchange-stream-words 1024 --no-profile \
  --reference-run --reference-fp32 --reference-inferences 2 \
  --device-lock artifacts/layout-sweep/device.lock \
  --memory-profile-directory "$out/memory" --save-reference-inputs "$out/resident" \
  --package "$out/model.ipuexe" > "$out/build-run.log" 2>&1
target/release/ipu-host-exchange-bench "$out/model.ipuexe" "$IPU_CONFIG" \
  --data "$out/resident" --sdk "$POPLAR_SDK_ENABLED" \
  --device-lock artifacts/layout-sweep/device.lock > "$out/latency.log" 2>&1
```

## Reproduction

### Full SigLIP local-optimization check

The full-model results above have local optimization disabled. A subsequent
27-layer batch-two run with the same configuration and `--optimization-steps 8`
does **not** pass hardware validation. It must not replace the validated baseline.

The optimizer rejects its proposed tile mapping at placement, then accepts one
layout change: encoder attention (operation 12) uses materialized scores instead
of 64-column streaming blocks, and the Q view boundary (value 350) stays in its
native packed format. Estimated cycles fall from 25,281,807 to 21,976,602 (13.1%),
with 68,388 B/tile encoded exchange storage. Subsequent proposals fail placement
of an 82,944-byte standard-memory allocation. Planning takes 493.1 s with 32
Rayon threads. The first hardware inference returns NaNs and fails the FP32
comparison; there is no valid measured speedup.

A control with `--optimization-steps 0 --attention-strategy materialized` passes
both resident calls (minimum cosine 0.994043410, maximum absolute error 0.454393).
It uses 68,044 B/tile of exchange storage and estimates 22,197,354 cycles (12.2%
below the streaming baseline). Materialized attention therefore fits and works
for at least this full-model plan. The control has different QK/PV grids as well
as retaining the canonical Q boundary, so the failed optimized plan's exact
numerical defect is not isolated by this comparison. No kernel or planner code
was changed during these checks.

Logs, packages and memory profiles are under
`artifacts/relay-big-20260912/siglip-b2-optimized/` and
`artifacts/relay-big-20260912/siglip-b2-materialized/`. Both reuse the reproduction
command above with the indicated flags; the materialized control does not save
a resident replay fixture. These checks used code from `721bb30`.

### Offline gather experiments

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

Offline validation: nine Python reconstruction/analysis tests pass, including symbolic
source-word reconstruction through gather, packing and multicast with padding
holes. Rust fixture compilation and Python lint pass. The offline experiment
itself did not change production behavior; integration is described above.
