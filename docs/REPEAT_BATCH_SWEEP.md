# Repeated SigLIP batch sweep (2026-09-06)

The sweep uses one structured `Repeat` with three sequential blocks and distinct
weight tensors for each iteration. It tries every integer batch size from 1 to 8
on the C600. `flash` means the streaming/blocked implementation; `materialized`
means the QK-transpose, softmax, PV implementation.

- MLP: 729 tokens, 1152 model channels, 4304 intermediate channels; F16 state.
- Attention: 729 tokens, 16 heads of 72 channels, three Q/K/V projections per
  block. Heads are joined before the next block's projections. The carried
  attention state is F32, its natural output precision; projections use F16.
- Every iteration has its own parameters, all resident in SRAM. This is not
  weight streaming or repeated execution with one shared parameter set.
- Runtime is the renderer-cropped profile span for the entire three-iteration
  program. Repeat currently has aggregate tile samples, not separately rendered
  samples for each operation inside each iteration.

## Fixes found while checking Repeat

1. Automatic operand layout selection could retarget a carried body argument to
   a GEMM's replicated input layout. The loop then replicated every yielded
   activation back to that layout (92 replicas for the batch-1 MLP). Carried
   layouts now stay fixed at the loop boundary; ordinary mid copies prepare
   each consumer. Retargeted invariant layouts also propagate to outer inputs.
2. The GELU polynomial overflowed for large finite FP16 inputs before `tanh`
   saturated. With strict floating-point exceptions, workers stopped and the
   program never reached sync. Clamp the polynomial input to [-8,8], retaining
   the original value for the final multiplication. Hardware validation covers
   all 63,488 finite FP16 bit patterns, in-place buffers, tails and canaries.
   Kernel costs were recalibrated: this fix increases isolated GELU cost.
3. A generic view fallback assigned a whole batch to one tile. It now distributes
   matrix rows, retaining contiguous rows. Factor views also have an explicit
   inverse: exchanging the split and merge axes is not the inverse permutation.
   Joined heads are copied as contiguous regions, not individual interleaved
   columns. Coordinate tests cover ranks 2–5, all axis pairs and cropped windows.
4. F32-to-F16 conversion was representable and costed, but had no device kernel.
   Sequential attention needs it before each subsequent projection. A worker
   codelet now converts complete word pairs, with one worker handling an odd
   final element. Its startup and loop work are included in costing.
5. Attention output validation now uses logical storage maps, including F32,
   rather than assuming binding order equals tile ownership. Diagnostic parameter
   generation uses standard deviation `1/sqrt(fan_in)` so deep random networks
   retain bounded variance. Numerical tolerances were not relaxed.

The isolated GELU/reduction checker now includes kernel source in its build-cache
key and runs under strict FP exceptions. The hardware result is in
`artifacts/repeat-sweep/gelu-finite-stack`. Maximum errors were 0.00201314 for GELU
and 0.00464249 for reduction, across 204 cases. Worker exception diagnostics are
available through `exchange-live-state --workers`.

## Reproduction and artifacts

Build `ipu-trivial-test` and the `ipu-stack` profile CLI in release mode, then run:

```sh
python3 scripts/repeat-batch-sweep.py --output artifacts/repeat-sweep
```

The runner uses eight concurrent builds with six Rayon threads each and serializes
hardware access with `artifacts/layout-sweep/device.lock`. It writes commands,
logs, result JSON, packages, raw profiles and rendered HTML. Use a new output
directory after changing code: existing result files are resumed.

Final MLP results are in `artifacts/repeat-sweep-carried`; final attention results
are in `artifacts/repeat-sweep-cast`. Earlier `repeat-sweep` and
`repeat-sweep-final` directories contain diagnostic attempts and superseded
results, not the final comparison. `repeat-sweep-joined` isolates the missing
cast implementation and is also superseded.

## Results and remaining limits

| Batch | Three-block MLP | Three-block streaming attention | Three-block materialized attention |
|---|---|---|---|
| 1 | PASS, 979,482 cropped cycles | PASS, 1,788,726 cropped cycles | Gaussian diagnostic PASS; profiled build fails loopback placement validation |
| 2–8 (each tested) | Planner SRAM rejection | Planner SRAM rejection | Planner SRAM rejection |

Streaming and materialized attention Gaussian diagnostics compare 256 samples of
final loop output, with maximum absolute errors 0.000031 and 0.000105 respectively.
The streaming constant-output run checks all 839,808 logical elements. MLP checks
all 839,808 Gaussian-reference output elements, with maximum error 0.000381.
These results do not claim that the profiled materialized case is fixed: its
post-placement exchange lowering rejects source/receiver memory-element overlap
in a multicast loopback. Diagnostic instrumentation changes placement enough
that its separate hardware package passes. No materialized cycle result is
available from this sweep.

For batch-2 MLP, the smallest rejected candidate reports 333,728 standard bytes
(including estimated exchange rows), 241,664 interleaved bytes, 49,152 package
support bytes, and 517,024 bytes of maximum simultaneous tensor/table usage.
Interleaved reservation rounds up to 262,144 bytes. The sum of independent class
maxima and support is therefore 645,024 bytes, against 586,960 planned-data bytes.
There is also a 1,104-byte contiguous-standard-allocation overflow. This is a
planner rejection, not a hardware allocation failure or proof that no layout
could fit. The fixed arena split and distinct class peaks matter even when the
simultaneous total alone looks small enough.

Attention's late preparation exchange has 182,652 transfers for streaming and
182,216 for materialized. Counts are whole-device totals, not per-tile table
sizes. In the streaming final placement, the largest row in that phase has 2,186
32-bit words (8,744 bytes); the materialized preliminary schedule has 6,035 words
(24,140 bytes). Other phases, participating receivers, and repeat address data add
further storage. Less fragmented layouts are a useful next direction; scheduling
speed alone would not remove the table-size cost.

Validation: 131 release codegen tests and four CLI tests pass, plus the doctest;
one pre-existing codegen test remains ignored. Clippy passes with the existing
argument-count and type-complexity allowances. No hardware program was rerun
merely to obtain another timing sample.
