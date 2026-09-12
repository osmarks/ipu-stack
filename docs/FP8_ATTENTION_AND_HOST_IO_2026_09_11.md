# FP8 attention and resident image/embedding transfers

Experiments use native F143 operands and FP16 AMP accumulation for selected
attention products. Softmax, score storage, attention outputs, and existing
FP32 merge state retain their previous precision. The default remains FP16
attention with FP32 AMP accumulation. The experimental controls are
`--attention-qk-fp8-scale=-4` and `--attention-pv-fp8-scale=-4`, independently.
Both operands of a selected product use that scale. This evaluates materialized
attention with independent product grids, not the streaming Flash implementation.

The implementation reuses the existing distributed product generator and cast
kernels. Inner partitions use 32-element FP8 groups, rather than the 16-element
FP16 groups. Casts are explicit mid operations after F16 operand distribution;
they are priced and included in the profiles. These results do not establish
how fast an implementation with earlier quantization or fused producer casts
could run.

## Projected attention

Batch one, fused QKV, FP8 projections at scale -4, materialized attention,
four local optimization steps, B1024 exchange streams. Single deterministic
hardware invocation for each variant. Gaussian inputs and weights; comparison
against the existing device-precision host reference. No attention quantization
is simulated by that reference.

| Attention products | Cropped cycles | Time | Change | Maximum absolute output error |
|---|---:|---:|---:|---:|
| FP16 | 216,444 | 144.296 us | baseline | 0.000061 |
| FP8 QK | 222,630 | 148.420 us | +2.86% | 0.000145 |
| FP8 PV | 221,076 | 147.384 us | +2.14% | 0.001511 |
| FP8 QK and PV | 229,644 | 153.096 us | +6.10% | 0.001562 |

The PV comparison retains the same product grid: the maximum GEMM kernel time
falls from 15,228 to 7,872 cycles. However, the compute round containing its
preparation and GEMM grows from 19,080 to 23,712 cycles. Both operands must be
cast, with the larger cast reaching 6,888 cycles. The existing arithmetic is
substantially accelerated; preparing FP8 operands erases the benefit.

QK uses a different selected grid: 145/146 query rows and 32/48 output columns
per tile, versus 81 rows and 64/80 columns. Its maximum GEMM takes 7,242 cycles,
but its preparation/GEMM round grows from 23,010 to 28,770 cycles. Softmax is
unchanged at 23,700 maximum kernel cycles. Whole-plan changes also alter the PV
grid in the QK-only run, so this is not a fixed-layout arithmetic comparison.

Artifacts: `artifacts/fp8-attention-20260911/{qk,pv,both}/` contain packages,
raw profiles, rendered `model.html`, kernel/phase JSON and run logs. The FP16
comparison is `artifacts/attention-grids-20260911/dense/`.

## Full 27-layer ViT

Batch one, 378 x 378 input, fused QKV, FP8 projection/MLP scale -4, eight local
optimization steps and B1024. Each package passes three successive resident
invocations with identical logical outputs, compared to an unquantized FP32
reference. These are randomized-model checks, not accuracy on trained SigLIP.

| Attention products | Cropped cycles | Time | Change | FP32-reference cosine |
|---|---:|---:|---:|---:|
| FP16 | 12,836,406 | 8.557604 ms | baseline | 0.994102280 |
| FP8 QK | 12,948,846 | 8.632564 ms | +0.88% | 0.994187413 |
| FP8 PV | 12,961,548 | 8.641032 ms | +0.97% | 0.994595915 |
| FP8 QK and PV | 13,101,936 | 8.734624 ms | +2.07% | 0.994329794 |

All clear the required cosine > 0.99. Slightly higher cosine in a quantized
variant is not evidence that quantization improves general accuracy; rounding
errors can cancel for this particular model/input. The combined profile contains
no FP16 GEMM kernels, confirming that both encoder and MAP attention products
actually switched. All three plans fit with resident parameters.

Thus FP8 attention arithmetic works in the tested numerical regime, but the
current explicit operand casts make it a performance regression. Leave the
production default unchanged. A profitable variant would need to avoid or
combine these conversions—for example by consuming already-quantized operands
or having a producer write them—rather than merely changing AMP instructions.

Packages, rendered profiles, phase/kernel queries and logs are under
`artifacts/fp8-attention-20260911/full27-{qk,pv,both}/`. The FP16 baseline is
`artifacts/attention-residual-20260911/full27/`.

## Host transfer mechanism

This section excludes parameter upload, loading, initialization and reference
computation. The resident interface receives host-packed FP16 image patches and
returns an FP16 embedding. Image normalization and conversion from an external
image format are outside this interface.

For batch one, 378 x 378 RGB pixels contain 857,304 bytes. The 1,152-element
embedding contains 2,304 bytes. The tested input distribution introduces no
replication or padding bytes in the upload. There are three upload batches and
one output batch. Each batch allows at most one 4 KiB slice per participating
tile, reusing the same pinned host storage across batches.

The driver pins/maps a command page and per-tile data pages once per session.
It copies each input batch into those pages and acknowledges the host-sync
register. Selected controller tiles emit exchange requests, and target tiles
issue host-read packets. Payloads are split at host-page boundaries, with
at most 1 KiB per incoming long packet. Each tile receives into its reserved
host aperture and then copies the bytes to the tensor allocation using the
supervisor host runner. Output tiles send directly from their tensor allocation,
with at most 256 bytes per packet; a closing read orders completion. The driver
then copies the embedding out of pinned storage.

The current implementation is sequential: upload, model computation, download.
The aperture can serve as scratch during computation, so this interface does not
prefetch the next image into it while the current inference runs. Host packet
encoding and routing are in `ipu-exchange`; host batching/placement is in
`ipu-codegen/src/host.rs`; handshakes and pinned-buffer copies are in `ipu-driver`.

The default register-poll loop sleeps 100 us, which often becomes about 155 us
on this host. `HostSession::set_poll_interval(Duration::ZERO)` provides an
explicit busy-poll alternative without changing the default. It consumes a
host CPU during waits, including model computation.

## Resident inference loop

`ipu-host-exchange-bench` loads a saved package and its weights once, then
repeatedly uploads the image, executes inference, and downloads the embedding.
Every returned embedding must exactly match the output of a validated reference
run. The timer covers the whole loop, including host copies, handshakes,
execution, output collection and byte comparison. Loading, parameter upload,
reference computation and reporting are outside the timer. Per-call latency is
also recorded, with printing deferred until the loop ends.

Build an unprofiled ViT with `--no-profile --reference-run --reference-fp32
--reference-inferences 20 --save-reference-inputs DATA`. This saves packed input,
weights and validated output. The package fixes the number of resident calls;
the fixture executes all of them. Replay it with:

```sh
RUST_LOG=warn target/release/ipu-host-exchange-bench MODEL.ipuexe "$IPU_CONFIG" \
  --sdk "$POPLAR_SDK_ENABLED" --device-lock artifacts/layout-sweep/device.lock \
  --data DATA --poll-us 0
```

Use `--poll-us 100` to measure the existing default. Full profile packages are
rejected to avoid including profiler readback in the measurement. The fixture
reuses the same image each time; it tests steady resident execution and transfer
cost, not host image decoding/preprocessing. Host wall times vary with CPU cache
state and scheduling, despite deterministic IPU computation.

### Full-model measurements

Batch-one 27-layer ViT including MAP, FP8 projections/MLP and default FP16
attention. Fused QKV, eight optimization steps, B1024. Profiling is disabled.
Twenty resident invocations per polling mode; no warmup calls are discarded.

| Host polling | Whole-loop time | Mean per image | Images/s |
|---|---:|---:|---:|
| Default 100 us sleep | 192.700 ms | 9.634986 ms | 103.788 |
| Busy polling | 174.396 ms | 8.719821 ms | 114.681 |

Both loops pass exact embedding comparisons on every invocation. The preceding
20-call reference run also passes, with FP32-reference cosine 0.994102280 and
maximum absolute error 0.394307. All parameters stay resident through the loop.
The package, validated data and timing logs are in
`artifacts/fp8-attention-20260911/resident27/`; the timing logs are `poll100.log`
and `busy.log`. Replay does not rerun the planner.

Busy polling removes about 0.915 ms per image, or 9.5% of the default loop time,
at the cost of occupying a host CPU during waits. Its 8.720 ms complete-loop
latency is close to the earlier 8.558 ms cropped device profile. These are
separate profiled/unprofiled builds, so their difference is not an isolated
measurement of PCIe transfer time.

### Output completion

`invoke_streaming_deferred` intentionally returns before the final output
transfer. For this three-input-batch, one-output-batch program, phases 0–5
transfer the image; phase 6 releases the final input completion and includes its
staging copy, model computation and arrival at the output transfer barrier.
`HostSession::finish` then releases the output transfer, waits for completion,
and collects the embedding from pinned storage.

The main inference runner previously released that final transfer explicitly.
The convenience `invoke`/`invoke_prepared` paths collected too soon; they now
finish the output transfer before returning. Both the reference runner and the
new fixture use this completion path. Preliminary `host-counters/` experiments
collected before releasing the last transfer and are invalid for timing; none
of that counter instrumentation remains in the code.

## Validation

- All three FP8 projected-attention variants pass the device-precision reference.
- All three FP8 27-layer variants pass three resident calls with FP32-reference
  cosine greater than 0.99.
- The new FP8 lowering test covers QK-only, PV-only and combined conversion,
  including odd key/channel tails and kernel compilation.
- Workspace/all-target checks and the driver unit suite pass. The codegen suite
  passed before the final FP8 tail test was added; that test also passes.
- Both host polling modes pass exact embedding comparisons for three resident
  one-layer invocations and twenty full 27-layer invocations through the
  convenience `HostSession::invoke` API.

Implementation commits: `82d0a84` (FP8 product experiments) and `68ddd71`
(resident loop fixture and host output completion).

## Matched execution and memory profiles for the blog

`artifacts/blog-profile-20260912/full27/model.html` and `memory.html` are a
paired execution timeline and exact per-tile allocation map. The executable
rebuilt with memory diagnostics is byte-identical to the original profiled
`artifacts/attention-residual-20260911/full27/model.ipuexe` (SHA-256
`62fb94a8754c5e03bc1f90696f44c78eb8396f30b55e739df037be504cf74069`).
The fresh capture measures 12,836,412 cropped cycles / 8.557608 ms, six cycles
longer than the original capture. All three resident checks pass at cosine
0.994102280. The allocation map includes profiling reservations; it describes
the profiled executable, not the unprofiled host-loop package. Both HTML files
are standalone. The artifact directory includes the raw profile, memory JSON,
a checked Chromium screenshot and reproduction instructions in `README.md`.

## Batch-two follow-up (September 12)

The 27-layer batch-two configuration still fails before local optimization, both
with profiling (56.327 s) and without it (52.342 s). Settings match the preceding
full-model checks, with batch changed to two. Neither reaches hardware execution.
Logs and the initial memory estimates are in
`artifacts/batch2-check-20260912/{full27,unprofiled27}/`.

In the unprofiled run, logical tile 506 first fails to allocate a 62,984-byte
replicated FP8 activation in lifetime order. Its size-ordered retry fails on an
82,944-byte persistent QKV-weight sequence: 27 layer shards of 3,072 bytes,
with one weight replica. Persistent-eligible free ranges total 83,632 bytes,
but the largest hole is 42,408 bytes. An additional 4,344 bytes in the host
aperture cannot hold persistent weights. This is a failure of this partial
placement, not proof that a different complete placement or layout cannot fit.
The optimizer requires a successfully packaged initial plan before searching.

The optimized batch-one map does contain real headroom: counting address ranges
never occupied at any phase, the minimum is 62,160 bytes, the median 97,068,
and the maximum 469,732. Stacked reuse rows do not add capacity. They are packed
by address overlap, not globally ordered by execution time. For example, tile
0's end-of-execution host aperture reservation occupies row 26, while rows
27–32 contain earlier allocations at different addresses, used at steps 237–301.
