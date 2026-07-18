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

Host operations are not yet accepted by the static packager. Host-page layout
and HSP transitions remain whole-device metadata in `.ipuexe`; adding them must
not reintroduce an all-tile command broadcast or device command dispatch loop.

The linker resolves calls from generated tile programs to kernel ELF symbols.
Kernel artifacts stay reusable and independent of the final tile and device
images. Content-addressed package blobs deduplicate identical linked sections
and generated instruction streams.

## Migration boundary

`LoweredTileProgram` and the exchange scheduler implement the compile-time
model. `device/static_runtime.S` supplies startup, worker rendezvous, and
completion. The Rust emitter generates the ordered exchange and compute calls
for each tile. The role-based command table and its dispatcher have been
removed.

## Static-lowering hardware evidence

The static emitter consumes `LoweredTileProgram` directly, emits distinct
straight-line programs, and branches to per-tile executable exchange rows
without a command table. Hardware tests verify a 64-word point transfer,
fanout, multicast, an all-tile permutation, and an 11-stage dependent reduction
with compute between exchanges. Diagnostic readback verified every expected
output in those graphs.

A sparse compute tail followed by an unrelated dense exchange remains a red
hardware gate because the static programs do not yet emit a reusable all-tile
device barrier at that boundary. Replaying the recovered GSP sequence there
leaves the packet origin running while all follower tiles wait, so that failed
path is not retained.

The next static implementation therefore needs an explicit, hardware-verified
GSP-to-exchange phase transition and a reusable device barrier. It must not hide
the issue by reserving physical tile 0, choosing a completion tile that ignores a
waiting origin, serializing through the host, or restoring runtime roles.
