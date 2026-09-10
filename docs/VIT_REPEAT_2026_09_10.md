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
median 1, 95th percentile 78, and maximum 78 words. Across the entire program
there are 9,966 such sites, also with maximum 78. These calls recur on each loop
iteration. Unlike the mostly one/two-word Repeat patches, the 78-word batches
are plausible worker-patcher candidates. No isolated patcher timing or worker
speedup was measured in this evaluation.
