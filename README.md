# ipu-stack

`ipu-stack` is a small collection of runtime, packaging, code-generation, and
early graph-lowering components for Graphcore IPU21 devices. It does not yet
contain the allocator or complete graph-to-device compiler.

## Components

- `ipu-target` generates and encodes device and host exchange programs.
- `ipu-codegen` emits straight-line supervisor code from caller-resolved tile
  programs.
- `ipu-elf` compiles Graphcore tile sources and links Colossus ELF objects.
- `ipu-package` reads, writes, and validates `.ipuexe` application packages and
  cycle profiles.
- `ipu-profile` queries cycle profiles.
- `ipu-driver` initializes hardware, loads packages, and drives host exchange.
- `ipu-runtime` is a thin device/load/session wrapper.
- `ipu-tests` builds and runs the explicit hardware diagnostic package.
- `ipu-cli` exposes generic compile, link, inspect, profile, load, and host-run
  operations.

The `device/` directory retains the static runtime support and generic FP16 and
FP32 GEMM kernels. Workload-specific kernels and planners are intentionally out
of scope.

## Design boundary

`ipu-codegen::build_package` accepts a `ComputeGraph` and `PackageConfig`. The
graph is shaped structured SSA. Its separate mid-level lowering selects
precision and layout with a toy cost model and inserts explicit casts and
rearrangements. Mid-to-low lowering then produces logical per-tile shard work,
kernel runs, synchronized exchanges, and structured repeats. Package
construction produces completion-only tile programs until SRAM placement,
exchange encoding, and kernel-symbol selection are implemented. The config
uses one shared `PipelineConfig` for target, tile count, input formats, operator
catalog, scheduling, and profiling. `PackageConfig` adds the toolchain, static
runtime source, and build directory.

`TileProgram` remains the finalized lower representation. Its exchange rows,
addresses, kernel symbols, operands, arguments, and profile destinations are
all explicit rather than inferred by codegen.

## Build

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The hardware test builds and round-trips its own package before loading it:

```sh
IPU_CONFIG=config.bin \
POPLAR_SDK_ENABLED=/path/to/poplar \
scripts/hardware-e2e.sh
```

The test checks that every supervisor and worker context halts after the

## CLI

```sh
ipu-stack kernel-compile device/static_runtime.S /tmp/runtime \
  --sdk "$POPLAR_SDK_ENABLED"
ipu-stack object-inspect kernel.o
ipu-stack package-inspect application.ipuexe --bindings
ipu-stack profile-render profile.capnp -o profile.html
ipu-stack profile-query profile.capnp --group-by kernel
ipu-stack host-run application.ipuexe bootloader.elf config.bin graph
```

The CLI intentionally excludes graph construction, allocation, model commands,
diagnostic workload generation, and format-conversion experiments.
