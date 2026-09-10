# Two-layer ViT Repeat evaluation

Benchmark support: `08c2c1d`; search-effort CLI: `a01de7a`.
Artifacts: `artifacts/vit-repeat-20260910/b1-n2-beam8/`.

## Model and build

Batch one, 378×378 pixels, 729 tokens, So400m width 1152, hidden width 4304,
16 heads, fused QKV, FP8 GEMMs at scale -4. The embedding, position addition,
final encoder normalization and MAP head run once. One structured Repeat runs
the encoder twice with distinct weights, biases and normalization parameters.
The encoder body is built once and imported using the checked operation API;
its parameter inputs become iterated sequences, with the residual state carried
between iterations. `--vit-layers` defaults to one, preserving the old benchmark.

The normal 64-wide search was stopped after approximately ten minutes in
high-level planning. `lower_operations` separately invokes `lower_repeat` for
each outer beam branch, and each invocation plans the whole body using the same
beam width. A CPU sample spent about 80% in resolving layout shard extents.
This is nested layout search, not exchange scheduling or patch generation.

For this evaluation, the existing configurable beam was exposed through the
benchmark CLI and set to eight. The planner default is unchanged. High-level
planning took 39,471 ms; finalist selection, including scheduling/assembly, took
139,989 ms. Finalist zero was selected from 24 complete candidates. Final tensor
placement took 342 ms. Reserved exchange tables were 23,984 bytes per tile;
host descriptors were 2,236 bytes per tile.

```sh
source .env
RAYON_NUM_THREADS=32 target/release/ipu-trivial-test "$IPU_CONFIG" \
  --sdk "$POPLAR_SDK_ENABLED" --workload siglip-vit-benchmark \
  --vit-batch 1 --vit-layers 2 --planning-beam-width 8 \
  --fuse-qkv --fp8-scale=-4 --reference-run \
  --diagnostic-atol 0.2 --diagnostic-rtol 0.05 \
  --device-lock artifacts/layout-sweep/device.lock \
  --package artifacts/vit-repeat-20260910/b1-n2-beam8/model.ipuexe \
  --profile-output artifacts/vit-repeat-20260910/b1-n2-beam8/profile.ipuprof
```

## Hardware result

Reference and hardware validation pass, maximum absolute error **0.075684**.
The cropped profile spans **1,189,716 cycles, 0.793144 ms at 1.5 GHz**. Detailed
samples cover the first encoder iteration; the `repeat-remainder` sample covers
the second. This is a constrained-search result, not an optimized two-layer
runtime comparison against the earlier single-layer plan. The package ran once.
Rendered profile: `profile.html`; operation summaries: `operations.json`.

Topology tests cover fused and unfused projections, distinct parameter sequences,
a single repeated body, and one-time bookends. Both ViT tests and Clippy pass.
A supplementary small FP8 case was rejected at the initial projection. A small
FP16 run using the default wide search was stopped along with the full-size wide
search; neither supplementary run reached hardware.

## Actual patch and source-displacement structure

The audit reads the final package's generated calls and replacement-word tables,
then decodes the *called* exchange rows. Debug ranges also describe patch data,
so ranges merely named `exchange row` must not all be decoded as instructions.
The script is `artifacts/repeat-address-audit/audit.rs`; output is
`patch-audit.txt`. Counts below are per tile-row in the emitted loop body,
not counts summed over both dynamic iterations.

- 1,395 Repeat-patched instruction words across 912 rows with changing sources.
- 908 rows have one uniform iteration-dependent source displacement.
- Three rows have two displacement runs; one has three runs.
- All mixed rows combine stationary sources with one nonzero displacement.
- Repeat patches per affected row: 437 rows patch one word, 473 patch two,
  one patches five and one patches seven.

Thus 99.56% of affected rows are structurally suitable for a single outgoing-base
relocation, subject to validating the hardware's nonzero-base semantics for all
used send forms. The four mixed rows do not exhibit tight alternation. No new
placement constraints appear necessary for the large majority of these rows.
This audit does not establish timing or correctness of mid-row base changes.

Cross-phase sharing is a separate, larger patch workload in this model. There
are 3,377 sharing-patcher call sites inside Repeat; their list lengths have
maximum 78 words. Across the entire program there are 9,966 such sites, also
with maximum 78. These calls recur on each loop
iteration. Unlike the mostly one/two-word Repeat patches, the 78-word batches
are plausible worker-patcher candidates. No isolated patcher timing or worker
speedup was measured in this evaluation.

## Phase-level eligibility and worst-tile patch work

The phase audit pairs generated row calls with profile phase metadata, including
inactive tiles' synchronization records. It checks that every physical tile has
exactly one corresponding row call for each recorded exchange phase. Code:
`artifacts/repeat-address-audit/phases.rs`; results: `phase-patch-audit.txt`.

The Repeat body has 25 exchange phases (physical IDs 3–27). Eighteen have no
iteration-dependent sender addresses. Four changing-source phases have a uniform
displacement on **every** sender in the phase:

| Phase | Operation | Moving senders | Maximum Repeat-patched words on a tile |
|---|---|---:|---:|
| 5 | QKV projection | 292 | 1 |
| 16 | Attention output projection | 144 | 1 |
| 21 | MLP up projection | 184 | 2 |
| 25 | MLP down projection | 288 | 2 |

These four phases are structurally eligible for one constant outgoing base per
sender for the entire phase. Base registers are per tile: different senders do
not need equal displacements. Stationary senders use zero displacement, and
Repeat's receiving buffers retain their addresses.

Three other phases fail the constant-base criterion:

| Phase | Operation | Mixed physical tiles | Displacement runs | Maximum Repeat-patched words on a tile |
|---|---|---|---:|---:|
| 3 | Attention normalization preparation | 223 | 2 | 1 |
| 18 | Output projection bias/add preparation | 524, 526 | 2 each | 7 |
| 19 | MLP normalization preparation | 524 | 3 | 2 |

Phase 3's mixed sender has displacements `[2304, 0]`; phase 18's two mixed
senders each have one transfer displaced by 1152 bytes followed by 38 stationary
transfers; phase 19's mixed sender has `[2304, 0, 2304]`. Thus mid-phase base
changes would require one, one, and two internal transitions respectively,
plus whatever initial/final base setup is required. This is structural evidence,
not validation of the hardware semantics or timing of those base changes.

For cross-phase row sharing, the **worst tile patches 78 words in phase 15**,
preparation for the attention output projection. Within Repeat, the only other
sharing-patch phases are 8 and 19, each with maximum one word per tile. The
largest Repeat-specific patch list is seven words in phase 18. These maxima,
rather than a percentile across unrelated rows, identify the worker-patching
and base-relocation experiments worth measuring.

## Worker bulk patching

The row-sharing helper now uses six workers for lists of at least 24 words.
Workers patch indices worker_id, worker_id + 6, etc. Generated offsets are
unique instruction positions, so writes are disjoint. The supervisor waits for
local worker completion before executing the exchange row. Smaller lists retain
the supervisor loop. Repeat's individual word patcher is unchanged.

Hardware checks used sparse destinations and guard words at list sizes 0, 1, 7,
23, 24, 25, 29, 30, 31, 77, 78, 79, 156 and 1024. All 28 worker/scalar cases pass
bitwise comparison. At 24 words the worker path takes 450 cycles versus 846 for
the old loop; at 78 words it takes **936 versus 2520 cycles**; at 1024 words,
9468 versus 31848. These include timing/wrapper overhead. The earlier estimate
of 3744 cycles for 78 words, based on assembly source instruction counting,
overstated the measured old-loop cost.

The full batch-one, two-layer FP8 ViT passes reference validation with unchanged
maximum absolute error 0.075684. Cropped runtime is **1,185,018 cycles
(0.790012 ms)** versus 1,189,716 previously, saving 4698 cycles (0.395%).
The second-iteration remainder spans 493224 cycles versus 494802. These are
single executions of each full package, not averaged timing samples.

Artifacts: `artifacts/vit-repeat-bulk-20260910/`, including `check.rs`,
`compare.log`, `run.log`, `operations.json` and rendered `profile.html`.
The focused check uses the production helper and a copy of the old scalar loop;
it covers non-multiples of six and the dispatch boundary. Its source can be
built temporarily as an ipu-tests binary, then run with --sdk and --output.

## Reusable region planning

Implementation: `d43c1ff`. The body search is now a separate, reusable candidate
frontier in `mid/region.rs`, with Repeat-specific construction still in the
planner. The search instance owns an immutable source/output/shape, graph,
configuration and cost-model context. Its cache key contains argument types,
ownership offsets, canonical storage groups, automatic-layout and parameter
flags, allocation multiplicity and required format equalities. It caches both
successful frontiers and infeasible contexts.

Body values use local IDs; attaching a candidate remaps values, storage groups,
deferred-input references and nested Repeat references into the enclosing state.
Operator implementation programs retain their independent local namespaces.
Unrelated outer prefixes are not part of the cache key. Instead, every attached
candidate is costed with the enclosing live values before the outer Pareto beam
prunes it. The usual beam limits bound combinations between parent branches.
Repeat no longer discards all but the first body candidate, and an infeasible
boundary rejects that branch rather than aborting other outer alternatives.
This introduces no new IR layer or fixed weight-layout policy.

The full-size batch-one, two-layer ViT now completes at the default beam width
of 64. Each of four planner configurations performs **8 body searches and 56
cache hits**, replacing 256 body searches with 32 in total. Different boundaries
still require different searches. High-level planning took **415414 ms**;
selection including high-level planning, scheduling and package construction
completed in **521188 ms**. The prior default-width run was stopped after about
ten minutes, so there is no completed before/after wall-time ratio. A CPU sample
still shows shard-layout resolution dominating the remaining work.

The build screens 48 complete candidates, admits four to placement and one to
scheduling, selecting finalist zero. Hardware and reference validation pass,
maximum absolute error **0.109497**. Cropped runtime is **1,040,742 cycles
(0.693828 ms)**; the second-iteration remainder spans 422622 cycles. This is
12.18% faster than the previous beam-eight worker-patching package (1185018
cycles), but both beam width and retention of body alternatives changed.
The package was executed once.

Artifacts: `artifacts/vit-repeat-region-20260910/`, including `run.log`,
`model.ipuexe`, `profile.ipuprof`, rendered `profile.html`, `operations.json`
and a CPU sample. Reproduce with the command above, omit --planning-beam-width,
and use this artifact directory for outputs.

Validation: 212 release codegen tests pass, five ignored. New tests cover
renumbered-equivalent boundaries, key distinctions, cached failures, retained
body alternatives, attachment into an existing state and tile expansion of the
attached candidates. Existing randomized Repeat lowering, sequence, placement
and low-expansion tests also pass. Clippy passes with the existing argument-count
and type-complexity allowances. Additional per-context start/end timing logs
make distinct slow searches visible before an entire region search completes.
