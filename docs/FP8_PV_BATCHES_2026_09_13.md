# FP8 PV at batch two and extended local search

Compiler: `cfb658f`. Full 27-layer SigLIP, fused QKV, FP8 projection/MLP
weights at scale −4, B1024 exchange rows. Randomized weights and inputs;
FP32 reference and two resident inferences. These are not trained-model
accuracy measurements.

## Batch two

Replayed `artifacts/bs2-search-20260912/state.json` (98 completed search
attempts). Its encoder attention is materialized, with QK grid 6×7×1 and
PV grid 7×1×6. The FP8 experiment changes only encoder PV scale to −4;
MAP remains FP16. Unlike the BS1 experiment, this PV grid already has
nonempty shards at FP8's 32-element grain.

| Encoder PV | Resident FP32 cosine | Host-loop batch latency | Images/s | Placed estimated cycles |
|---|---:|---:|---:|---:|
| FP16 | 0.993905451 | 12.432461 ms | 160.87 | 17,567,158 |
| FP8 | 0.994151272 | 12.210071 ms | 163.80 | 17,079,727 |

Both pass two resident numerical checks and exact saved-output host replay.
Host timings include image upload and embedding download, excluding weight
initialization. Each timing contains only two invocations, with busy polling;
the apparent 1.8% improvement is approximate, not a precise device-cycle
comparison. Both binaries are unprofiled.

The 80 KiB exchange-table guard rejects both replays: 88,176 bytes/tile
for FP16 and 86,904 for FP8. Raising the guard to 96 KiB allows both
unprofiled packages to fit. This changes only the experiment's limit, not
the compiler default. The fully profiled variants still fail placement of
a 27,648-byte standard allocation on tiles 60 and 368 respectively.
Profiling adds 896 bytes of samples per tile, more generated code, and
splits an available address range at 524,288. These runs therefore do not
provide detailed BS2 execution timelines. Exact memory profiles are available.

Artifacts under `artifacts/fp8-pv-batches-20260913/`:

- `bs2-unprofiled/`: FP16 control package, resident fixture, exact memory map.
- `bs2-pv-unprofiled/`: FP8 PV package, resident fixture, exact memory map.
- Corresponding `.log`, `.sh`, `-state.json`, and `-host.log` files.
- `bs2-control.log` and `bs2-pv.log`: failed profiled placement attempts.

The copied old checkpoint context removes obsolete default alignment/tail/
distinct-element fields. Profiling and the exchange guard are updated explicitly
for each invocation; normal checkpoint compatibility validation remains enabled.
Original checkpoints and packages are preserved.

## More BS1 search

Started 64 additional local-search steps from
`artifacts/producer-fp8-20260913/pv-state.json`, using 16 Rayon threads.
This retains the existing FP8 PV recipe and explores surrounding boundaries,
cast ordering, and saved operator alternatives. Those alternatives largely
predate FP8 PV, so this is not exhaustive FP8 attention layout enumeration.

The first accepted change enables early casting for operation 21, the MLP
down-projection. Detailed estimated cycles improve from 10,906,290 to
10,868,841, but hardware performance regresses:

| BS1 recipe | Renderer-cropped cycles | Time | FP32 cosine |
|---|---:|---:|---:|
| Previous FP8 PV best | 10,666,374 | 7.110916 ms | 0.994352454 |
| Search checkpoint at attempt 64 | 10,798,722 | 7.199148 ms | 0.994352454 |

The candidate is 1.24% slower on hardware. Operation 21's profiled exchange
spans increase from 40,308 to 50,922 cycles, and padding appears in its
preparation. Phase spans overlap other work and must not be added as an
independent critical-path decomposition. Both resident checks pass.

The previous FP8 PV package remains the measured BS1 best. The checkpoint
profile is `artifacts/fp8-pv-batches-20260913/bs1-step64/model.html`.
Its state, package, query output and reference logs are retained alongside it.

At 12:30 UTC the search is still running (PID 554031), with another accepted
boundary change at value 344 and estimated cycles 10,851,912. That later
checkpoint has not been hardware measured. Progress is atomically saved to
`artifacts/fp8-pv-batches-20260913/bs1-search-state.json`; command and log are
`bs1-search.sh` and `bs1-search.log`. Further estimated wins need hardware
validation before replacing the measured best.
