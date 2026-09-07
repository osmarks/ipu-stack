# Compiler CPU profile, 2026-09-07

The batch-one SigLIP attention build with automatic attention/product planning
took **217.8 seconds before and 103.8 seconds after** these changes (2.10× faster).
All 24 screened candidates retained identical costs, both exact scheduling
candidates retained identical costs, and the resulting package is byte-identical:

```
0e261d4745147827c90f30c4316d175e9814f017eabd3637a7624941381d8130
```

These are individual profiled runs on the 56-core host with
`RAYON_NUM_THREADS=16`, release binaries, and a warm kernel compiler cache.
Times cover entry to mid planning through tile-image construction. The final
whole command, including package writing/inspection and perf overhead, took
105.0 seconds and peaked at 2.92 GiB RSS. CPU timings vary; these are not medians.
The unchanged package also matches the previously hardware-tested attention
program, so no duplicate device timing was needed.

| Build phase | Before | Indexed allocation/traffic | Also parallel scheduling |
|---|---:|---:|---:|
| Mid planning | 12.20 s | 12.43 s | 12.34 s |
| Finalist screening and scheduling | 170.66 s | 130.41 s | 60.94 s |
| Final storage placement | 0.93 s | 0.05 s | 0.06 s |
| Final exchange lowering | 25.43 s | 25.97 s | 27.49 s |
| Placement improvement search | 7.16 s | 0.69 s | 1.17 s |
| Complete package construction | 217.77 s | 170.93 s | 103.78 s |

## Findings and changes

The original CPU profile attributed 35.5% of samples to `allocate_tile_class`
and 25.9% to mapping resource scoring. The expensive `select_finalist` phase
includes storage allocation and topology screening, not only exact scheduling.

* Allocation scanned the entire device's alias groups for every tile and memory
  class. Groups and Repeat constraints are now partitioned by tile once, preserving
  their allocation order, lifetimes, and address choices.
* Mapping scoring rebuilt SRAM-element maps and ordinary traffic loads for every
  candidate mapping (up to 171 mappings per finalist here). Bijective tile mappings
  merely rename these resources. Their loads are now indexed once, and each mapping
  adjusts only potentially paired transfers, including the borrowed sender lane.
  All 24 four-tile permutations are checked against reference resource scores.
* Independent exchange phases and the ordinary/remapped versions of a finalist
  now schedule concurrently in the existing Rayon pool. Collection order and
  score tie-breaking remain deterministic. Each phase retains its own relocation
  recipe; the shortlist size and infeasibility handling are unchanged.
* Encoding a `PhaseProgramBuilder` now borrows it, avoiding complete schedule clones
  solely to validate or normalize generated rows.

The new profile puts `byte_spans` at 15.8% of samples, its byte-span sorting at
another 9.6%, and `TileProgramSchedule::finish` at 14.2%. Their larger shares reflect
removal of other work, not a comparable increase in absolute cost. Next candidates
are generating physical spans without scalar enumeration and incrementally
encoding the unaffected prefix of a tile's exchange stream. The latter must preserve
receive cutovers, combined controls, and SENDPICP alignment: simply skipping
validation would change correctness. Final exchange lowering remains expensive
despite reuse of scheduling choices.

Validation: 133 codegen tests, 42 exchange tests, and the codegen doctest pass;
one preexisting codegen test remains ignored. Clippy passes with the repository's
existing allowances for argument count and type complexity.

## Reproduction

Artifacts are in `artifacts/compiler-perf/{baseline,indexed,parallel}/`, with phase
comparisons in `artifacts/compiler-perf/comparison.json`. `baseline/self.txt` and
`parallel/self.txt` contain the CPU sample summaries. Keep the profiled executable
available unchanged until perf has collected its build ID; rebuilding it during
recording can prevent symbol resolution (this happened to the intermediate run).

After enabling the local Poplar SDK and building `ipu-trivial-test` in release mode:

```bash
mkdir -p artifacts/compiler-perf/new
cp target/release/ipu-trivial-test artifacts/compiler-perf/new/compiler
RAYON_NUM_THREADS=16 perf record -F 99 -g --call-graph dwarf,8192 \
  -o artifacts/compiler-perf/new/perf.data -- \
  artifacts/compiler-perf/new/compiler c600-init.ipucfg \
  --device-lock artifacts/layout-sweep/device.lock \
  --package artifacts/compiler-perf/new/model.ipuexe \
  --workload siglip-attention-benchmark --attention-strategy auto \
  --attention-batch 1 --attention-products auto --inspect-exchanges \
  > artifacts/compiler-perf/new/run.log 2>&1
perf report -i artifacts/compiler-perf/new/perf.data --stdio \
  --no-children -g none --sort symbol
```

`--inspect-exchanges` exits after package inspection without running the IPU.
The default release build supplies function symbols for self-sample attribution;
inlined caller attribution needs a build with debug information.

## Byte-span radix sorting

Benchmarked `radsort` 0.1.1 on the actual eight-byte `ByteSpan` records, with
`span.offset` as the key, including AMP-output and block-major spans generated
by the compiler. The benchmark alternates algorithms, excludes input copying,
and reports the median of five batches. Its source and invocation are in
`crates/ipu-codegen/src/storage/sort_bench.rs`; raw results are in
`artifacts/compiler-perf/radix/sort-benchmark.csv`.

| Input | Entries | Comparison sort | Radix sort | Speedup |
|---|---:|---:|---:|---:|
| AMP output | 256 | 3.34 µs | 2.63 µs | 1.27× |
| AMP output | 1,024 | 14.98 µs | 8.75 µs | 1.71× |
| AMP output | 16,384 | 441.54 µs | 212.43 µs | 2.08× |
| Block major | 300 | 4.16 µs | 2.33 µs | 1.79× |
| Block major | 1,024 | 15.46 µs | 6.92 µs | 2.24× |
| Already sorted | 1,024 | 0.61 µs | 6.76 µs | 0.09× |

The integration uses radix sorting at 256 entries and above, but first returns
if the spans are already sorted. Small lists retain comparison sorting. The
[upstream benchmark](https://github.com/JakubValtar/radsort/wiki/Benchmarks)
is a useful starting point; it measures random scalar values, so these local
measurements also cover field keying and the compiler's structured offsets.

The isolated full build retained the identical package hash. Sorting's CPU
sample share fell from about 9.6% to 5.3%; total sampled cycles fell about 3%.
Whole-command wall time was 108 seconds versus the preceding 105-second run,
so this single full-build measurement does not establish a wall-time gain.
