# Hardware bring-up

The direct path uses the Graphcore kernel driver ABI but does not load Poplar:

1. Attach `/dev/ipu0`, map the configuration BAR, and issue the ICU reset
   mailbox transaction.
2. Replay a captured symbolic register configuration.
3. Install Graphcore's secondary tile bootloader through the hardware
   autoloader.
4. Attach a transport buffer, configure exchange-buffer request IDs, and submit
   final tile images in groups of 64.
5. Send the execute sentinel and use HSP marks for loader and application startup
   synchronization.
6. Attach application host pages and drive named calls through `HostSession`.

`APPLICATION_LOAD_BASE` is `0x4c014`: the secondary loader places its generated
launch instruction at `0x4c010`, then copies payload bytes immediately after it.
One-frame tiny images stall the current loader protocol. The transport therefore
pads payloads to the smallest established working envelope, `0x4134` bytes. The
padding is not recorded as an application segment or treated as an architectural
SRAM requirement; its precise protocol cause remains an investigation item.

The worker fixture uses `runall` once. All six workers execute barrel-threaded,
write through their distinct vertex-base values, synchronize locally, and let
the supervisor aggregate the words. There are no per-worker mailbox loops.
