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
  command generation, direct host execution, and automated diagnostics.
- `ipu-cli`: build, inspect, plan, probe, and load commands.

## Build and inspect

```sh
# Offline encoder, planner, package, and driver tests only.
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

cargo run -p ipu-cli -- kernel-compile device/runtime.S /tmp/runtime \
  --name runtime --sdk "$POPLAR_SDK_ENABLED"
cargo run -p ipu-cli -- encoder-plan -o /tmp/encoder.json --tiles 1472
```

`cargo test --workspace` does not claim device-transfer coverage. The exclusive
hardware acceptance gate is `scripts/hardware-e2e.sh`; its tests are ignored by
the ordinary test command so that package builds cannot reset an attached IPU.
It requires
`POPLAR_SDK_ENABLED` and `IPU_CONFIG`, and accepts optional `IPU_BOOTLOADER` and
`IPU_DEVICE` overrides. The suite includes seeded randomized exchange graphs;
run one reproducible case directly with `IPU_RANDOM_SEED=0x1234 cargo run -p
ipu-runtime --bin ipu-randomized-e2e`.

Logging uses `tracing`. Set `RUST_LOG`, for example
`RUST_LOG=ipu_driver=debug,ipu_elf=info`, to expose batch and linker details. Set
`IPU_LOG_FORMAT=json` for newline-delimited JSON events.

## Current hardware boundary

The Rust path resets and configures a C600, loads linked code onto all 1472
tiles, attaches host pages, and drives HSP without Poplar. Native D2H is not
currently accepted as working: with input and output assigned disjoint host
regions, the output remains zero. Earlier round-trip tests reused the input
host offsets for output and could therefore pass on unchanged H2D input bytes;
that overlap is now forbidden by the layout and covered by a unit invariant.
The required hardware tests fail until the device-to-host protocol writes the
disjoint output region correctly.

Host transfers are split at the recovered short/long packet limits and 4 KiB
attachment boundaries. The runtime allocates one attached buffer per page,
places the command page after the data pages, and derives one self-contained
call's HSP phase count from its generated operations. Multi-page layouts and
packet boundaries are covered by unit tests and hardware runs.

Exchange Tx/Rx staging addresses are selected from explicit memory constraints:
tile, byte range, alignment, placement direction, and half-open phase lifetime.
The allocator rejects exhaustion and permits the same address on different
tiles or across disjoint lifetimes. The host H2D destination is constrained to
the protocol's directly encodable 16 KiB host-to-tile window; runtime control
storage within that window reduces the largest contiguous allocation currently
available to an application. Ordinary exchange receivers use the full 32 KiB
exchange window. No concrete staging address is specified by the host exchange
acceptance graph.

The exchange planner lowers one-to-one transfers to absolute exchange rows and
coalesces transfers sharing a source tensor into multicast groups. The same
single-packet multicast plan passes under Poplar orchestration, but the Rust
end-to-end randomized test cannot establish its result until native D2H works.
The graph runtime executes generated per-tile plan tables and separately linked
compute kernels. The older diagnostic graph contains a 1,472-value reduction,
an all-tile affine permutation, and a relay, but its launcher still needs
conversion to the per-epoch HSP protocol.

The scheduler treats the on-chip fabric as non-blocking. Tile-disjoint groups
run concurrently; role conflicts become statically timed slots in the same
launch, with one synchronization and a shared event horizon. Compute is a
following graph phase: the dispatcher branches to a separately compiled kernel
symbol and exchange commands perform no arithmetic. A randomized hardware
acceptance path attempts H2D to the controller tile, two generated tile-exchange
epochs via a relay tile, and D2H from the automatically allocated return range.
Command boundaries use the generated C600 GSP program before the next exchange
can begin. D2H lowering emits separate tile-0 transaction ownership and
source-tile payload programs in one host phase. Multi-packet D2H uses one XREQ
and close around 256-byte packet/payload pairs, matching the recovered SDK
schedule.

The seeded randomized hardware runner uses allocated addresses and generated
payloads. Each case performs H2D, generated D2D fanout to one through four
random destinations, a generated gather into distinct tile-0 ranges, and D2H
verification. Default cases cover 1, 2, 15, 16, 17, 31, 63, and 64 words;
`IPU_RANDOM_CASES` extends into larger boundaries.
