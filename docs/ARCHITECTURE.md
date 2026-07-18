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

Host operations are also compiled into the tile streams in declared call order.
Only the host endpoint owner and transfer target receive host-exchange actions.
Host-page layout and HSP transitions remain whole-device metadata in `.ipuexe`.
There is no all-tile command broadcast or device command dispatch loop.

The linker resolves calls from generated tile programs to kernel ELF symbols.
Kernel artifacts stay reusable and independent of the final tile and device
images. Content-addressed package blobs deduplicate identical linked sections
and generated instruction streams.

## Migration boundary

`LoweredTileProgram` and the exchange scheduler already implement the intended
compile-time model. `device/graph_runtime.S` and the role-based command table are
legacy bring-up machinery and must not be extended. They remain only until the
static tile emitter covers synchronization, exchange, compute, completion, and
host actions and passes the existing hardware gates.
