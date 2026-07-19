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

Host-exchange hardware tests can delay every host acknowledgement by an
independent seeded interval. This verifies that device execution remains at the
host rendezvous until the write actually occurs:

```sh
IPU_TEST_HOST_WRITE_JITTER_MAX_US=2000 \
IPU_TEST_HOST_WRITE_JITTER_SEED=0x1234 \
  scripts/hardware-e2e.sh
```

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
the host. The planner selects row sharding from the matrix shape and tile count;
the K and N block dimensions are separately configurable. Each K iteration
multicasts A row blocks and B column blocks before invoking a six-worker
specialized GEMM kernel. Exact output checking has passed on hardware through
dimension 4,096. Use release builds for large graphs: most construction time is
compiler/runtime code rather than Cap'n Proto I/O.

```sh
IPU_GEMM_DIMENSION=2048 cargo run --release -p ipu-runtime --bin ipu-gemm-e2e

IPU_GEMM_DIMENSION=128 \
IPU_PROFILE_OUTPUT=/tmp/gemm-profile.capnp \
  cargo run --release -p ipu-runtime --bin ipu-gemm-e2e
capnp decode schemas/profile.capnp Profile </tmp/gemm-profile.capnp
cargo run --release -p ipu-cli -- profile-render /tmp/gemm-profile.capnp -o /tmp/gemm-profile.html

IPU_MEMORY_PROFILE_OUTPUT=/tmp/gemm-memory.capnp \
  IPU_GEMM_PACKAGE_ONLY=1 \
  cargo run --release -p ipu-runtime --bin ipu-gemm-e2e
cargo run --release -p ipu-cli -- memory-inspect /tmp/gemm-memory.capnp --tile 0

# Compare one completed output block directly in tile SRAM against D2H.
IPU_GEMM_LOAD_PACKAGE=/tmp/gemm.ipuexe \
IPU_GEMM_SRAM_CHECK_BLOCK=15,24 \
  cargo run --release -p ipu-runtime --bin ipu-gemm-e2e
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

For bounded table or JSON analysis, use `profile-query`; it groups by kernel,
operation, phase, tile, or kind and filters by timeline offset and semantic
metadata. See [docs/PROFILING.md](docs/PROFILING.md) for query examples and the
definitions of phase-critical versus aggregate work cycles.

Compute samples include the exact kernel symbol, specialization role and shape,
tensor and SRAM operands, scalar arguments, and planner-supplied semantic
metadata. The blocked GEMM planner adds output wave, output block coordinates,
inner block, row range, and byte count. Exchange samples identify the following
kernel and summarize the local sends and receives. `IPU_MEMORY_PROFILE_OUTPUT`
writes a separate all-tile allocator report containing every tensor region,
address, size, lifetime, allocation kind, and matching host binding name.

`IPU_PROFILE_GRANULARITY` controls cycle instrumentation:

- `graph` records one interval per tile for low-overhead whole-graph timing.
- `phase` records each static compute phase and separates every exchange epoch
  into synchronization wait and active exchange intervals. This is the default
  and the recommended semantic overview: exchanges, GEMM, GeLU, layout kernels,
  layers, and blocks remain distinct.
- `step` additionally preserves every lowered kernel invocation per tile.
  It provides the finest diagnostics and produces the largest reports.

The older `IPU_PROFILE_AGGREGATE` setting is retained as an alias for `graph`.
All modes instrument every tile; granularity changes time resolution, not tile
coverage. Sampling inserts device work and barriers, so use an unprofiled run
for final performance numbers.

Profile kinds are `synchronization`, `exchange`, `compute`, and `idle`. An idle
sample means that the tile has no kernel command in that compute phase; the
following synchronization sample accounts for time spent waiting for other
tiles to finish.

The GEMM verifier runs while the application remains loaded. On any mismatch it
bulk-reads the owning tile's complete 64x64 C block through an inactive worker
context and reports both SRAM-versus-D2H and SRAM-versus-expected difference
counts. `IPU_GEMM_SRAM_CHECK_BLOCK=ROW,COLUMN` forces that comparison on a
passing run.

## Composed MLP

`ipu-mlp-e2e` composes rectangular blocked GEMMs and GeLU activations into one
static graph. The default is an eight-layer FP32 network with batch 512 and
width 2,048. All weights and the input are uploaded before execution;
intermediate activations remain on-device, and only the final activation is
read back. Layer boundaries fuse GeLU with the required AMP-C16 to AMP-A8
layout conversion. Non-final layers forward their GEMM accumulators directly
into that transition instead of evacuating and copying an intermediate home
tensor. Diagonal validation weights keep host verification linear in activation
size while the IPU still executes dense GEMMs.

```sh
IPU_PROFILE_GRANULARITY=phase \
IPU_PROFILE_OUTPUT=profiles/mlp-512x2048x8-phase.capnp \
  cargo run --release -p ipu-runtime --bin ipu-mlp-e2e
```

`IPU_MLP_BATCH`, `IPU_MLP_WIDTH`, `IPU_MLP_LAYERS`, and
`IPU_MLP_ROW_BLOCK` override the shape and row specialization.
`IPU_MLP_PACKAGE_ONLY`, `IPU_MLP_PACKAGE`, and
`IPU_MEMORY_PROFILE_OUTPUT` provide package-only and allocator inspection
paths analogous to the GEMM executable.
