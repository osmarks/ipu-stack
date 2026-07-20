# Compiler architecture

The stack has three artifact levels:

1. A kernel artifact is one specialized tile operation compiled by `popc` to a
   relocatable Colossus ELF object plus metadata.
2. A tile program is a straight-line supervisor program linked with only the
   kernel sections and constants used by that tile.
3. An `.ipuexe` is a Cap'n Proto whole-device image containing all tile
   programs, SRAM initializers, tensor bindings, host calls, and device setup.

The graph compiler produces globally ordered exchange and compute phases. It
then lowers that schedule to one `LoweredTileProgram` per logical tile. A tile
program contains concrete exchange rows and concrete compute calls with final
SRAM addresses. It does not contain opcodes for a device-side interpreter.

The final code generator emits a distinct instruction stream for every tile:

- initialize the supervisor and six workers once;
- execute each required device synchronization;
- call an inline exchange row when the tile is a source or destination;
- call a specialized compute kernel when the tile owns that operation;
- perform the final completion synchronization and stop.

A D2D exchange has source and destination actions. Other tiles participate only
in synchronization required by the phase; they are not modeled as forwarding
tiles. Multiple transfers in one phase are placed on one static event timeline.

Host operations are lowered to straight-line calls before and after graph
phases. Each payload phase contains generated XREQ/target code and synchronization
followers; host-page layout and HSP phase counts remain whole-device metadata in
`.ipuexe`. There is no all-tile command broadcast or device dispatch loop.

The linker resolves calls from generated tile programs to kernel ELF symbols.
Kernel artifacts stay reusable and independent of the final tile and device
images. Content-addressed package blobs deduplicate identical linked sections
and generated instruction streams.

## Tile-memory placement

The compiler distinguishes placement lifetime from instruction-specific memory
requirements. `MemoryPolicy` contains ordered resident arenas for model
parameters and ordered transient arenas for activations and ordinary scratch.
An allocation may spill to a later arena but never spans an arena boundary.
Resident placement is checked against every allocation in the complete schedule
because parameters are loaded before execution and remain live throughout it;
transient placement only considers overlapping phase lifetimes.

FP16 GEMM B blocks use ordinary 64-bit loads and do not require interleaved
SRAM. The AMP accumulator/output block still uses paired PACE accesses and is
fixed in the IPU21 interleaved element. Kernel-fixed allocations are included
in policy placement, so resident data may use otherwise-free interleaved space
without overlapping active AMP scratch.

The SigLIP runner defaults to placing resident data high-to-low and transient
data in low ordinary SRAM, then interleaved SRAM, then high ordinary SRAM.
`IPU_SIGLIP_RESIDENT_ARENAS` and `IPU_SIGLIP_TRANSIENT_ARENAS` accept ordered
comma-separated `base..limit` address ranges for placement experiments. These
controls change allocation preference, not kernel memory semantics.

## Migration boundary

`LoweredTileProgram` and the exchange scheduler implement the compile-time
model. `device/static_runtime.S` supplies startup, worker rendezvous, and
completion. The Rust emitter generates the ordered exchange and compute calls
for each tile. The role-based command table and its dispatcher have been
removed.

## Static-lowering hardware evidence

The static emitter consumes `LoweredTileProgram` directly, emits distinct
straight-line programs, and branches to per-tile executable exchange rows
without a command table. Hardware tests verify point transfer, fanout,
multicast, an all-tile permutation, and two-launch randomized graphs with sparse
compute between exchanges. The randomized suite passed 18 deterministic cases,
including a second-launch receive on physical tile 0.

The phase boundary reproduces the SDK's internal exchange sequence. Every tile
executes `sync INTERNAL`; active tiles then perform the local worker rendezvous
and execute their complete exchange row, while inactive tiles execute
`sans 0; sync ANS`. D2D phase transitions do not use a host packet, GSP, or a
release multicast, and no tile is reserved for synchronization.

An alternating 11-stage reduction followed by a dense all-tile permutation
passes hardware acceptance. The earlier packet loss came from routing
independent single-destination groups through the multicast scheduler. The SDK
uses its point-to-point schedule for those groups; the compiler now does the
same, finalizes the receiver's source selector, and converts its receive address
to the compiler-allocated absolute exchange-window address. Fanout and groups
with a nonzero dependency offset continue to use the multicast scheduler.
