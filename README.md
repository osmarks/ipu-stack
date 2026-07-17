# ipu-stack

`ipu-stack` is an experimental Rust compiler, package format, and direct host
runtime for Graphcore IPU21 devices. It does not link Poplar into the host
application. The Graphcore tile compiler remains an offline kernel compiler.

## Artifact model

There are three deliberately separate artifact levels:

1. A kernel artifact is a `.gp`, extracted Colossus ELF32 relocatable object,
   and compiler metadata JSON. It represents one specialized tile operation.
2. A tile program is linked machine code plus a fixed-width declarative command
   stream. Tiles may select different kernel specializations and code sections.
3. An `.ipuexe` is a Cap'n Proto whole-device application. It contains
   content-addressed blobs, final segments for every physical tile, tensor
   bindings, host-exchange pages, device-configuration writes, and named entry
   commands.

The independent linker performs reachability-based section collection and all
Colossus relocation types emitted by the current toolchain. No independent
instruction assembler is required.

## Crates

- `ipu-elf`: invokes `popc`, extracts ELF and metadata, and links tile images.
- `ipu-compiler`: fixed-shape graph IR, layouts, exchange phases, SRAM liveness,
  specialization keys, per-tile commands, and a CPU encoder reference.
- `ipu-package`: Cap'n Proto `.ipuexe` serialization, validation, compression,
  and semantic digests.
- `ipu-driver`: direct Linux device setup, bootloader framing, application load,
  HSP synchronization, and attached host exchange pages.
- `ipu-runtime`: graph-schedule packaging, dynamic kernel retention, per-tile
  command generation, and automated diagnostic execution.
- `ipu-cli`: build, inspect, plan, probe, and load commands.

## Build and inspect

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

cargo run -p ipu-cli -- kernel-compile device/runtime.S /tmp/runtime \
  --name runtime --sdk "$POPLAR_SDK_ENABLED"
cargo run -p ipu-cli -- encoder-plan -o /tmp/encoder.json --tiles 1472
```

The ignored end-to-end hardware test is `scripts/hardware-e2e.sh`. It requires
`POPLAR_SDK_ENABLED` and `IPU_CONFIG`, and accepts optional `IPU_BOOTLOADER` and
`IPU_DEVICE` overrides.

Logging uses `tracing`. Set `RUST_LOG`, for example
`RUST_LOG=ipu_driver=debug,ipu_elf=info`, to expose batch and linker details. Set
`IPU_LOG_FORMAT=json` for newline-delimited JSON events.

## Current hardware boundary

The Rust path has attached to a C600, reset and configured it, loaded a linked
application onto all 1472 discovered package tiles in 64-tile bootloader
batches, completed startup synchronization, and run a supervisor plus six
barrel workers without an IPU exception. SDK-generated application host
exchange is represented and driven by `HostSession`. Native packet headers,
XREQs, command reads, and D2H instruction plans are independently assembled,
but the native output prototype does not yet return payload bytes: its host
page remains zero and its final phase can remain in WAEX. It is diagnostic
code, not a passing production data path.

The exchange planner lowers one-to-one and fanout transfers to absolute,
single-send exchange rows. Direct hardware
acceptance includes a 1,024-word single-packet broadcast to all 1,471 other
tiles. The graph runtime executes generated per-tile plan tables and separately
linked compute kernels. The automated hardware graph checks a 1,472-value
reduction, an all-tile affine permutation, a 64-word multicast, and a dependent
64-word relay. It validates every exposed word through diagnostic SRAM reads,
without Poplar exchange code generation or host-side phase delays.

The scheduler treats the on-chip fabric as non-blocking. Tile-disjoint groups
run concurrently; role conflicts become statically timed slots in the same
launch, with one synchronization and a shared event horizon. Hardware testing
includes a multicast whose relay receiver sends the payload onward in a later
slot without an intermediate BSP barrier. Compute is a following graph phase:
the dispatcher branches to a separately compiled kernel symbol and exchange
commands perform no arithmetic. The remaining runtime work is native host
completion and host-phase retirement plus broader kernel dispatch.
