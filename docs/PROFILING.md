# Profiling

`ipu-package` retains the Cap'n Proto cycle-profile format and `ipu-profile`
retains aggregation and filtering.

Low-level codegen supports explicit sample addresses:

- `CodegenOptions::initial_profile_address`
- `StepProfile::{before, after}`
- `CodegenOptions::final_profile_address`

The caller owns the profile buffer layout and converts captured counter words
into `ProfileReport`. The report contains the samples and metadata explicitly
recorded by code generation.

Inspect or query a report with:

```sh
ipu-stack profile-inspect profile.capnp
ipu-stack profile-render profile.capnp -o profile.html
ipu-stack profile-query profile.capnp --group-by kernel
ipu-stack profile-query profile.capnp --kind exchange --group-by phase
```

The profile schema supports operation names, phases, epochs, kernel symbols,
and metadata for the query layer.

## Kernel cycle calibration

Generate a source-identified database of per-kernel hardware measurements with:

```sh
scripts/calibrate-ipu21-costs.sh c600-init.ipucfg
```

The database is written to `profiles/ipu21-kernel-costs.json` and remains
untracked. It is intended for kernel development, regression comparisons, and
future autotuning; the planner continues to use its established empirical cost
model.

## Repeat iterations

Repeated bodies are instrumented per kernel/exchange, with a distinct profile
`epoch` for each iteration. The emitted loop keeps a profile-buffer cursor in its
stack frame and advances it by one body's timestamp stride. Executable code and
exchange rows remain shared; only timestamp storage and host profile metadata
scale with iteration count. Inactive tiles retain an aggregate idle interval.

The detailed batch-two, three-block FP8 MLP profile is
`artifacts/grouped-repair/mlp-b2-n3/profile.html` (raw `profile.ipuprof`). It contains
90,372 samples and spans 1,038,666 cropped cycles, about 346,222 per block. The
previous aggregate-only run spanned 1,031,874 cycles; detailed instrumentation adds
about 0.7%. Both pass numerical validation (maximum absolute error 0.015625).
The HTML was checked in headless Chromium and displays all three iterations.

A smaller three-block workload also passes. Regression coverage checks timestamp
count, iteration epochs, the emitted body's single-iteration address range and
retention of structured Repeat. The full test run passes 149 codegen tests, four
CLI tests and the doctest, with four manual/preexisting tests ignored; Clippy
passes with the existing argument-count/type-complexity allowances.

## Packed-copy batching

The subsequent profile at `artifacts/strided-copy/mlp-b2-n3/profile.html`
uses the same batch-two, three-block FP8 MLP and detailed instrumentation.
It spans 856,230 cropped cycles (285,410 per block), down 17.6% from 1,038,666.
Hardware reference validation passes with maximum absolute error 0.015625.

Physical copy spans can be traversed in destination order when source and
destination are distinct buffers and destination spans do not overlap. For the
regular `[164, 6, 32-byte]` transpose, this replaces 164 short strided helper
invocations with six longer ones. The 64-bit helper also uses hardware repetition
and stepping loads/stores, reducing its inner loop to two instruction bundles
per word. This preserves the copy mapping; it does not eliminate the conversion.
The final placement is rebuilt, so the overall speedup includes any resulting
exchange changes.

Aggregate tile cycles attributed to `copy_strided_u64` fall from 271,441,452 to
86,044,044, and the longest attributed interval falls from 62,748 to 16,590
cycles. The set of intervals changes because longer strided copies now use this
helper too; these aggregates are not matched-kernel microbenchmarks.

SDK codelets also batch irregular regions using offsets or compact worklists
(for example Poplibs `MultiSlice.cpp` and `BroadcastVectorInner2D.cpp`). A general
descriptor-list helper could extend batching to irregular copies, but is not
implemented here: this regular transpose fits the existing strided-copy ABI.
Regression coverage verifies byte mappings and preserves same-buffer ordering.
The updated suite passes 150 codegen tests, four CLI tests and the doctest, with
four ignored tests; Clippy passes with the existing allowances.
