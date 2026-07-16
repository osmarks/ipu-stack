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
   bindings, host-exchange pages, and named entry commands.

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
- `ipu-cli`: build, inspect, plan, probe, and load commands.

## Build and inspect

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

cargo run -p ipu-cli -- kernel-compile device/runtime.S /tmp/runtime \
  --name runtime --sdk "$POPLAR_SDK_ENABLED"
cargo run -p ipu-cli -- encoder-plan -o /tmp/encoder.json --tiles 1472
```

The hardware fixture is `scripts/hardware-runtime.sh`; it requires explicit
`POPLAR_SDK_ENABLED`, `IPU_CONFIG`, and `IPU_TILE_COUNT` environment values.

Logging uses `tracing`. Set `RUST_LOG`, for example
`RUST_LOG=ipu_driver=debug,ipu_elf=info`, to expose batch and linker details. Set
`IPU_LOG_FORMAT=json` for newline-delimited JSON events.

## Current hardware boundary

The Rust path has attached to a C600, reset and configured it, loaded a linked
application onto all 1472 discovered package tiles in 64-tile bootloader
batches, completed startup synchronization, and run a supervisor plus six
barrel workers without an IPU exception. Arbitrary application host exchange is
represented and implemented by `HostSession`, but the standalone fixture has
not yet completed non-debugger result readback. TDI cannot retirement-break its
terminal supervisor loop, so that is not used as a production data path.

The exchange planner now lowers one-to-one transfers to direction-specific
point-to-point rows and fanout to single-send multicast rows. The direct loop
runtime executes generated per-tile plan tables across repeated globally
synchronized launches. Hardware acceptance includes a 1,472-value parallel
sum: 11 reduction rounds, 97 exchange epochs, and the exact result `1084128`
without Poplar exchange code generation or host-side phase delays.

The current scheduler conservatively caps an epoch at 16 independent exchange
groups. Wider favorable matchings have passed at 368 groups, but a route-aware
resource model is still needed before raising the general cap. The remaining
executable-lowering work is to dispatch specialized compute kernels from the
same per-tile program stream and expose terminal results through normal host
exchange rather than the TDI acceptance breakpoint.
