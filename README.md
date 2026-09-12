# ipu-stack

`ipu-stack` is an experimental graph compiler, runtime, and profiling toolkit for
Graphcore IPU21 devices. It lowers tensor graphs to executable `.ipuexe`
packages, including layout selection, SRAM allocation, kernel generation,
exchange scheduling, and repeated execution with resident parameters.

The full pretrained SigLIP So400m/14 384 vision tower (27 encoder layers and MAP
head, batch size 1) runs on hardware. On six real photographs, its embeddings
achieved **0.994614–0.997866 cosine similarity** against an independent FP32
Hugging Face reference. Weights were uploaded once and retained across all six
inferences. This is a small numerical validation, not a task-accuracy evaluation
or a timing benchmark; see the [validation report](docs/SIGLIP_PRETRAINED_VALIDATION_2026_09_12.md).

## Components

- `ipu-codegen`: tensor graph lowering, layout planning, allocation, device
  kernels, and package construction.
- `ipu-exchange`: device and host exchange scheduling and encoding.
- `ipu-elf`: Graphcore tile compilation and ELF linking.
- `ipu-package`: `.ipuexe` packages and cycle-profile serialization.
- `ipu-profile`: cycle-profile queries and interactive HTML rendering.
- `ipu-driver`: hardware initialization, package loading, and host exchange.
- `ipu-runtime`: device, load, and session interfaces.
- `ipu-tests`: hardware diagnostics, model benchmarks, and reference validation.
- `ipu-cli`: compile, link, inspect, profile, load, and host-run commands.

`device/` contains assembly runtime support and kernels. Generated kernels and
operator implementations live in `ipu-codegen`.

## Compilation

`ipu-codegen::build_package` accepts a `ComputeGraph` and `PackageConfig`.
High-level graphs describe tensor operations and structured repeats. Mid-level
plans select whole-device implementations, precisions, and layouts, with
explicit conversions and reductions. Low-level expansion produces per-tile
kernel work and exchanges. Allocation, scheduling, linking, and encoding then
produce a complete executable package.

The planner first builds a baseline with canonical layout boundaries and compact
parameter storage, then evaluates local improvements. Accepted changes retain a
complete feasible package. `--optimization-steps 0` selects the baseline alone.
See [baseline and local planning](docs/BASELINE_LOCAL_PLANNING.md) and the
[compiler data flow](docs/COMPILER_DATA_FLOW.md).

## Build and hardware setup

Host tools use Rust 2024 and the Cap'n Proto compiler (`capnp`). Device compilation
requires the Poplar SDK toolchain; execution requires an accessible IPU device
and an `IPUCFG1` configuration capture. Host Rust compilation uses the native CPU
ISA through `.cargo/config.toml`.

```sh
cargo build --release --workspace
cargo test --workspace
```

Set the SDK and configuration paths, then run the hardware diagnostic:

```sh
export IPU_CONFIG=/path/to/config.bin
export POPLAR_SDK_ENABLED=/path/to/poplar
scripts/hardware-e2e.sh
```

The diagnostic builds and round-trips a package, loads it, and checks supervisor
completion and inactive workers. See [hardware bring-up](docs/BRINGUP.md) for
numerical GEMM smoke tests and configuration details.

## Pretrained SigLIP validation

The validated precision policy uses an FP16 input projection and F143 FP8
operands with FP16 accumulation/results for encoder and MAP dense GEMMs.
Calibration selects one fixed scale per GEMM position, shared by both operands
and across all encoder layers. The run uses nearest weight rounding, without
GPTQ or per-block scales. A single global scale of −4 clips learned activations
and is unsuitable for this checkpoint. See [FP8 support](docs/FP8.md).

Create a Python environment with PyTorch, Transformers, SafeTensors,
`huggingface_hub`, Pillow, and NumPy, then export the pinned checkpoint and demo
images and calibrate the scales:

```sh
python scripts/siglip-pretrained-fixture.py artifacts/pretrained/fixture
python tools/calibrate_siglip_fixture.py artifacts/pretrained/fixture \
  --fp16-embedding --scale-sharing repeat-role --shared-operand-scale \
  --nearest-only --report artifacts/pretrained/calibration.json
```

Calibration uses CUDA by default and the first three images; the other three
are held out. The fixture applies the checkpoint's RGB bicubic resize to
384×384 and normalization, then packs the top-left 378×378 pixels into patches.
Resizing directly to 378×378 would change the input.

Build and run all six images with the weights resident:

```sh
RAYON_NUM_THREADS=16 RUST_LOG=info target/release/ipu-trivial-test "$IPU_CONFIG" \
  --sdk "$POPLAR_SDK_ENABLED" --runtime-source device/static_runtime.S \
  --workload siglip-vit-benchmark --vit-layers 27 --vit-batch 1 \
  --fuse-qkv --optimization-steps 8 --exchange-stream-words 1024 \
  --reference-run --reference-fp32 --reference-inferences 6 \
  --reference-fixture artifacts/pretrained/fixture \
  --reference-calibration artifacts/pretrained/calibration.json \
  --no-profile --package artifacts/pretrained/model.ipuexe
```

The runner fails if any image's cosine similarity does not exceed 0.99. The
[validation report](docs/SIGLIP_PRETRAINED_VALIDATION_2026_09_12.md) records the
checkpoint revision, per-image results, and original artifacts. Earlier
randomized-model timings use a different precision configuration and do not
measure this calibrated pretrained run.

## Inspection and profiling

```sh
target/release/ipu-stack package-inspect application.ipuexe --bindings
target/release/ipu-stack profile-render profile.capnp -o profile.html
target/release/ipu-stack profile-query profile.capnp --group-by kernel
```

The runtime profile viewer shows tile timelines, exchange modes, synchronization,
and [estimated useful kernel work](docs/PROFILE_USEFUL_WORK.md). The benchmark
runner also accepts `--memory-profile-directory PATH` for memory reports,
including exact per-tile placement and allocation reuse. See
[cycle profiling](docs/PROFILING.md) and [memory profiling](docs/MEMORY_PROFILING.md).
