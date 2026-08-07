# Application format

The schema is `schemas/application.capnp`. Cap'n Proto supplies framing and
field layout.

Each tile segment stores its initialized bytes directly alongside its final
SRAM address, memory size, and access flags. A segment's data may be shorter
than its memory size; the remaining bytes are zero-initialized. There is no
blob table, compression codec, content hash, deduplication, or build digest.
Configuration writes carry hardware setup that cannot be expressed as tile
images without requiring a matching SDK schedule capture at load time. The
schema's `compilerVersion` field identifies the package producer.

Bindings map a logical tensor byte range to one or more `(tile, address)` SRAM
ranges. Host calls map input and output ranges onto driver-attached pages and
specify the number of HSP phases. Their optional input and output batch
boundaries group slices transferred concurrently during one device/host
rendezvous; absent boundaries mean one slice per rendezvous. This keeps
offline application packaging separate from the
Linux transport used to load or invoke it.

No graph or allocation policy is serialized. Final code, exchange rows, and
data are ordinary package segments at caller-selected SRAM addresses.
