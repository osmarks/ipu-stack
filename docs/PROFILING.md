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
