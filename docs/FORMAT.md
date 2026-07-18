# Application format

The schema is `schemas/application.capnp`. Cap'n Proto supplies framing and
forward-compatible field layout; large blobs may be zstd-compressed.

Each blob is addressed by SHA-256 of its uncompressed bytes. Tile segments name
a blob range, final SRAM address, memory size, and access flags. Identical code
and constants are stored once even when referenced by every tile. The build
digest covers all semantic package fields, including bindings, host calls, and
ordered device-configuration writes. Configuration writes let the compiler
carry hardware setup that cannot be expressed as tile images without requiring
a matching SDK schedule capture at load time.

Bindings map a logical tensor byte range to one or more `(tile, address)` SRAM
ranges. Host calls map input and output ranges onto driver-attached pages and
specify the number of HSP phases. This keeps offline application packaging
separate from the Linux transport used to load or invoke it.

The command graph is not serialized into tile SRAM. Executable lowering resolves
tensor IDs through SRAM allocations, emits final exchange rows, chooses
specialized kernel sections, and generates one straight-line machine-code
program per tile. The resulting code and data are ordinary package segments.
