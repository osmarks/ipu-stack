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
