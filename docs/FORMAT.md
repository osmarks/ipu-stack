# Application format

The schema is `schemas/application.capnp`. Cap'n Proto supplies framing and
forward-compatible field layout; large blobs may be zstd-compressed.

Each blob is addressed by SHA-256 of its uncompressed bytes. Tile segments name
a blob range, final SRAM address, memory size, and access flags. Identical code
and constants are stored once even when referenced by every tile. The build
digest covers all semantic package fields, including bindings and host calls.

Bindings map a logical tensor byte range to one or more `(tile, address)` SRAM
ranges. Host calls map input and output ranges onto driver-attached pages and
specify the number of HSP phases. This keeps offline application packaging
separate from the Linux transport used to load or invoke it.

Per-tile compiler commands are eight little-endian 32-bit words. They are a
compiler IR, not Colossus instructions:

- `Exchange`: opcode, phase, source tile, destination tile, tensor, byte count.
- `Compute`: opcode, phase, operation, output tensor, input count, input tensors.
- `End`: opcode `0xff`.

Executable lowering resolves tensor IDs through SRAM allocations, emits exchange
plan instructions, chooses specialized kernel sections, and applies ELF
relocations. Keeping this intermediate form explicit avoids coupling graph
scheduling to the device runtime ABI.
