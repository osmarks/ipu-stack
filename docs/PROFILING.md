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

Run the hardware calibration suite with the device configuration used by the
other tests:

```sh
scripts/calibrate-ipu21-costs.sh c600-init.ipucfg
```

This writes `profiles/ipu21-kernel-costs.json`. Pass it to `ipu-tests` with
`--kernel-calibration profiles/ipu21-kernel-costs.json`, or keep the equivalent
`IPU_STACK_KERNEL_CALIBRATION` setting in `.env`. `--no-kernel-calibration`
temporarily disables an environment-provided database.

Measurements are keyed by the emitted tile-kernel specialization and its local
work size. Exact measurements replace analytical local-kernel estimates;
related GEMM specializations are conservatively interpolated from measured
input and output work, while other unmeasured kernels retain bandwidth or
arithmetic fallbacks. The database
contains a digest of the kernel source tree and is ignored after those sources
change, so stale measurements cannot silently guide planning. Exchange timing
and documented target bandwidths remain target properties rather than kernel
calibration data.
