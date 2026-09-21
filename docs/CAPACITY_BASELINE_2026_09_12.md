# Capacity-first baseline — September 12, 2026

Later [full-model relay revalidation](EXCHANGE_GATHER_2026_09_12.md#full-model-revalidation)
supersedes the capacity results below: both SigLIP 27-layer and PE 24-layer
batch-two builds now fit and pass two resident hardware inference calls under
the normal exchange limits.

The experimental `--capacity-baseline` uses the existing operator catalogue,
conversion inserter, mid implementations and liveness estimator. It does not add
a retry planner or panel-streaming GEMM implementation. The established baseline
remains the default.

## Selection

- Canonical activations have one replica and use every available whole-row owner:
  577 owners for PE and 729 for SigLIP, rather than coarse 16 KiB chunks.
- Persistent parameter homes remain compact and unreplicated. Repeat retains all
  layers' parameters across host inference calls.
- Candidates are ranked on the actual input conversion, compute and output
  conversion sequence. Inputs with subsequent consumers remain live in that
  estimate. Automatic parameters use their compact homes instead of pretending
  they can reside directly in the compute layout.
- Both cast orders are evaluated before choosing an operator. The score minimizes
  total live tensor memory, then maximum standard allocation, estimated exchange
  rows, and cycles. Shortlisting also retains the smallest-buffer extreme.

There is no hard prohibition on transient activation replication. Existing GEMMs
trade replication against partial-result storage and weight staging. Nor is this
a global placement-aware scratch budget: the isolated candidate estimate omits
unrelated live values and the rest of the resident parameter set. Complete
package construction remains the feasibility check. The final PE MLP input still
has 16 transient FP8 replicas, and SigLIP's has eight. Early quantization avoids
materializing their larger FP16 counterparts. Trials with fewer replicas and
smaller standard buffers needed more interleaved partial-result storage.

Minimizing maximum standard allocation first was insufficient: it selected
larger interleaved partial results. Comparing both classes' simultaneous total
and allowing early casts gave the better capacity result.

## Hardware and capacity results

These batch-two tests use randomized parameters and FP8 F143 dense operands at
scale −4, FP16 attention, fused QKV, B1024 exchange streams and zero local
optimization steps. They do not validate pretrained PE accuracy; the PE probe
still omits RoPE and uses layernorm epsilon 1e-6.

| Case | Result |
|---|---|
| PE, 24 layers, batch 2 | Hardware/reference PASS; cosine 0.993717251, maximum absolute error 0.397096. Two resident inference calls, each processing two images. |
| SigLIP, 27 layers, batch 2 | Early exchange guard: 16,968 fragments on the busiest tile, limit 16,384. |
| SigLIP batch 2, diagnostic limit 17,000 | An earlier capacity trial reached 104,044 encoded exchange bytes on the busiest tile, above the 81,920-byte package budget. Raising the fragment guard did not make it feasible. |
| Pretrained SigLIP, batch 1, capacity policy | Selected a sub-word exchange in MAP preparation; rejected during exchange lowering. This policy is not the default. |

PE's estimated tensor-only per-tile peak fell from 456,744 to 370,472 bytes;
including the fixed support reserve, the reported estimate fell from 505,896 to
419,624 bytes. The successful trial used at most 74,628 bytes of exchange table
and repeat-patch storage per tile. Its estimated execution cost was 19,758,901
cycles per batch, **not a measured runtime**. Profiling was disabled. Concurrent
package construction took 174 seconds in that trial.

The unchanged default baseline was rebuilt with the actual pretrained SigLIP
checkpoint: all six image cases pass again, with the same cosine values
(0.994614293–0.997865842) as the earlier validation. Its estimated cost also
remains 15,706,340 cycles. This run is under `siglip-pretrained-default/`.

## Correctness fixes exposed by the new layouts

**Shared exchange rows:** patch offsets were inferred separately for each row
from its difference against normalized addresses. A first invocation using zero
address offsets could omit the patch table; a later invocation then panicked.
Even without the panic, a subsequent zero-address invocation could retain stale
addresses. All invocations now restore the same union of varying words. A test
reconstructs every invocation in both zero/nonzero orderings.

**Blocked attention:** the merge kernel stored its running maximum and denominator
after each FP32 value row, but the mid tensor did not reserve those two elements.
This overlapped the next row's values and wrote beyond the final row. Mid now
represents the state explicitly in the accumulator's width, and both FP32 and
final FP16 merge paths use the padded accumulator stride. Returned tensors keep
their original width. A one-layer PE diagnostic passes all 41 checkpoints; the
full batch-two reference test passes after this fix.

Layout-preserving pointwise candidates also now allow input precision conversion
instead of rejecting an otherwise usable layout. The ordinary conversion path
supplies the kernel's required precision.

## Reproduction and artifacts

Artifacts are under `artifacts/capacity-baseline-20260912/`. `pe-b2-state/` records
the successful full-model trial; `pe-diagnostic-state/` records the checkpoint
check. `pe-b2-final/` repeats the full validation through the final explicit
CLI option. `siglip-b2-final/` records the default guard failure, and `siglip-b2-limit/`
the diagnostic larger-limit experiment. The earlier directories preserve failed
experiments rather than being overwritten.

```sh
source .env
RAYON_NUM_THREADS=12 RUST_LOG=info target/release/ipu-e2e-test "$IPU_CONFIG" \
  --sdk "$POPLAR_SDK_ENABLED" --runtime-source device/static_runtime.S \
  --workload siglip-vit-benchmark --vit-model pe-core-l14-capacity \
  --vit-layers 24 --vit-batch 2 --fuse-qkv --fp8-scale=-4 \
  --capacity-baseline --optimization-steps 0 --exchange-stream-words 1024 \
  --no-profile --reference-run --reference-fp32 --reference-inferences 2 \
  --device-lock artifacts/layout-sweep/device.lock \
  --memory-profile-directory artifacts/capacity-baseline-20260912/pe-b2-final/memory \
  --package artifacts/capacity-baseline-20260912/pe-b2-final/model.ipuexe
```

For SigLIP, omit `--vit-model` and use `--vit-layers 27`. Leave the default
exchange limits in place; the larger-limit experiment was diagnostic only.

Validation: 240 enabled codegen tests pass (four ignored), plus 12 runner tests.
Clippy completes with existing workspace warnings. Regression tests cover the
capacity boundary ownership, compact parameters through Repeat, zero-address row
sharing, and space for online attention state.
