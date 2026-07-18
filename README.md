# ipu-stack

`ipu-stack` is an experimental Rust compiler, package format, and direct host
runtime for Graphcore IPU21 devices. It does not link Poplar into the host
application. The Graphcore tile compiler remains an offline kernel compiler.

## Artifact model

There are three deliberately separate artifact levels:

1. A kernel artifact is a `.gp`, extracted Colossus ELF32 relocatable object,
   and compiler metadata JSON. It represents one specialized tile operation.
2. A tile program is a distinct straight-line supervisor program linked with
   the specialized kernels and constants used by that tile.
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
  specialization keys, static per-tile programs, and a CPU encoder reference.
- `ipu-package`: Cap'n Proto `.ipuexe` serialization, validation, compression,
  and semantic digests.
- `ipu-driver`: direct Linux device setup, bootloader framing, application load,
  HSP synchronization, and attached host exchange pages.
- `ipu-runtime`: static tile code generation, graph packaging, direct host
  execution, and automated diagnostics.
- `ipu-cli`: build, inspect, plan, probe, and load commands.

## Build and inspect

```sh
# Offline tests plus required hardware acceptance.
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

cargo run -p ipu-cli -- kernel-compile device/runtime.S /tmp/runtime \
  --name runtime --sdk "$POPLAR_SDK_ENABLED"
cargo run -p ipu-cli -- encoder-plan -o /tmp/encoder.json --tiles 1472
```

`cargo test --workspace` includes exclusive hardware capability tests and
therefore requires an attached C600. Each transport capability is a separate
test, and an in-process lock serializes access to the device.
Crate-local `--lib` tests cover only encodings and structural invariants; they
are not a transport acceptance result.
`scripts/hardware-e2e.sh` runs those tests directly. It requires
`POPLAR_SDK_ENABLED` and `IPU_CONFIG`, and accepts optional `IPU_BOOTLOADER` and
`IPU_DEVICE` overrides. The suite includes seeded randomized exchange graphs;
run one reproducible case directly with `IPU_RANDOM_SEED=0x1234 cargo run -p
ipu-runtime --bin ipu-randomized-e2e`.

Logging uses `tracing`. Set `RUST_LOG`, for example
`RUST_LOG=ipu_driver=debug,ipu_elf=info`, to expose batch and linker details. Set
`IPU_LOG_FORMAT=json` for newline-delimited JSON events.

## Current hardware boundary

The Rust path resets and configures a C600, loads linked code onto all 1472
tiles, attaches host pages, and drives HSP without Poplar. Device-only point,
multicast, relay, permutation, and all-tile reduction graphs pass from
initialized SRAM.

Host/device composition passes with randomized payloads. Required hardware
tests cover local and remote H2D/D2H, distinct host slices on different tiles,
an 8-KiB remote round trip, H2D -> D2D relay -> D2H, and H2D -> compute -> D2D
-> compute -> D2H. The latter doubles a random `u32` on each endpoint and checks
the final wrapping result on the host. A 64-KiB round trip at tile address
`0x60000` also passes: H2D is automatically staged through the packet-addressable
exchange window and copied to ordinary SRAM, while D2H sends directly from the
high address.

Host transfers are split at the recovered short/long packet limits and 4 KiB
attachment boundaries. The runtime allocates one attached buffer per page,
places the command page after the data pages, and derives one self-contained
call's HSP phase count from its generated operations. Multi-page layouts and
packet boundaries pass direct hardware acceptance.

Large sparse host schedules do not duplicate one follower call per 4 KiB page
on every tile. The packager emits the tile's specialized active transfers and
compresses consecutive inactive phases into calls to a static counted loop.
For the 2048 GEMM this keeps the largest generated tile program below 13 KiB
despite 12,288 host attachment phases.

Host binding sizes and D2H source addresses must be word aligned. Direct H2D
destinations must be 32-byte aligned; aligned destinations outside the directly
encodable packet window are staged automatically.

Exchange Tx/Rx staging addresses are selected from explicit memory constraints:
tile, byte range, alignment, placement direction, and half-open phase lifetime.
The allocator rejects exhaustion and permits the same address on different
tiles or across disjoint lifetimes. The H2D packet field directly encodes a
16-KiB tile window. The packager allocates packet tables and reusable staging
space there, splits transfers at attached-page boundaries, and emits target-tile
copies when the requested destination lies elsewhere in SRAM. Ordinary
exchange receivers use the full 32-KiB exchange window.

The exchange planner lowers an independent one-to-one transfer with the
point-to-point scheduler and converts its receiver to the compiler-allocated
absolute exchange-window address. Transfers sharing a source tensor are
coalesced into multicast groups; dependent groups with a nonzero timing offset
also use the multicast scheduler. Both paths pass direct device-only hardware
acceptance without relying on native D2H.
The static runtime executes a distinct straight-line program per tile and calls
separately linked compute kernels. Hardware acceptance separately covers a
1,472-value reduction, an all-tile affine permutation, multicast, and a
dependent relay.

The scheduler treats the on-chip fabric as non-blocking. Tile-disjoint groups
run concurrently; local endpoint conflicts become statically timed slots in the same
launch, with one synchronization and a shared event horizon. Compute is a
following graph phase: each tile program calls a separately compiled kernel
symbol and exchange commands perform no arithmetic. The randomized hardware
acceptance path executes initialized multicast, sparse compute, and a second
random matching as one static program. D2D transitions use the SDK-derived
internal sync and ANS non-participation protocol without host intervention.
D2H and H2D target-operation encoders match SDK images for physical tiles 31
and 260. The XREQ owner is `target & 0x3d`, and its route word is
the bit at `2 * (target / 64) + ((target >> 1) & 1)` in the 46-bit XREQ
bitmap, split 24/22 bits across its two words; both formulas are checked
against extracted SDK code and direct hardware. Static host phases use
`sans 1; sync 1` followers, an XREQ
owner entering through sync 15, and a target entering and completing through
sync 7 around its sync-0 payload operation. No device command dispatcher is
used.

Offline unit tests verify encodings, allocation, lowering, and package structure;
they do not count as evidence that a transport capability works. The seeded
randomized hardware runner uses generated payloads and destinations. Each case
performs D2D fanout to one through six destinations, sparse compute, and a
second disjoint matching. Default cases sample 1, 15, 16, 17, 52, 64, 65, 127,
512, and 1,024 words. Diagnostic readback verifies every result.

## Blocked GEMM and profiling

`ipu-gemm-e2e` builds and runs a square FP32 GEMM whose matrices originate on
the host. Matrices use 64x64 blocks. Each output tile owns one A block, one B
block, and one C block; each K iteration multicasts the A row blocks, preserves
the received A block, multicasts the B column blocks through the reused receive
window, and invokes a six-worker specialized GEMM kernel. A 2048 square GEMM
uses 1,024 output tiles and 64 device exchange launches. Exact output checking
has passed on hardware at dimensions 64, 128, 1,024, 1,600, and 2,048.

```sh
IPU_GEMM_DIMENSION=2048 cargo run -p ipu-runtime --bin ipu-gemm-e2e

IPU_GEMM_DIMENSION=128 \
IPU_PROFILE_OUTPUT=/tmp/gemm-profile.capnp \
  cargo run -p ipu-runtime --bin ipu-gemm-e2e
capnp decode schemas/profile.capnp Profile </tmp/gemm-profile.capnp
```

Profiling is optional and absent from an ordinary package. A profiled package
samples the per-tile 32-bit cycle counter before and after each static exchange
or compute step, reads the samples back through the normal D2H path, and writes
the separate `schemas/profile.capnp` format. Durations must use wrapping
subtraction. Sampling dispatches a worker to read the counter and therefore
perturbs short steps; the records are intended for graph-level attribution,
not instruction-level benchmarking. The hardware acceptance suite runs a
profiled 128 GEMM, parses records for all 1,472 tiles, and requires exchange and
compute samples.
