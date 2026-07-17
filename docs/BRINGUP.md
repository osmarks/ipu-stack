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

`APPLICATION_LOAD_BASE` is `0x4c010`, matching SDK ELFs. The first image word is
an explicit reserved `nop`; application entry is `0x4c014`. Direct SRAM reads
confirm that the secondary loader installs the complete framed payload beginning
at `0x4c010`.
One-frame tiny images stall the current loader protocol. The transport therefore
pads payloads to the smallest established working envelope, `0x4134` bytes. The
padding is not recorded as an application segment or treated as an architectural
SRAM requirement; its precise protocol cause remains an investigation item.

The worker fixture uses `runall` once. All six workers execute barrel-threaded,
write through their distinct vertex-base values, synchronize locally, and let
the supervisor aggregate the words. There are no per-worker mailbox loops.

## Exchange runtime status

The Rust planner's point-to-point and multicast rows match the independent C++
oracle, and the single-packet multicast plan passes on hardware when executed
inside Poplar's runtime. The direct runtime now reaches the same plan with:

- preloaded launch/global-sync credits rather than host delays between device
  phases;
- the SDK supervisor's global master sequence and per-tile worker sync bases;
- `A6=1` on senders and receivers;
- `sans 0; sync 1` on every tile without a plan in an exchange launch;
- plan code in a separately allocated executable SRAM region.

The launch roles must not be conflated. Before the command loop, non-origin
tiles execute `sans 1; sync 1` as part of the device-wide synchronization. One
configurable physical tile emits the global packets and release, while every
tile participates in the barrier. The packet origin is not reserved afterward:
it executes the same sender, receiver, or `sans 0; sync 1` role as any other
tile in every payload epoch. Omitting the inactive payload role lets the packet
origin run ahead and deadlocks active endpoints. The older one-shot diagnostic
runtime still excludes its packet origin; the command-loop runtime does not.

Exchange configuration is executable-specific. A fresh SDK capture for logical
tile `0 -> 274` differs from the checked-in capture in four MMIO records, and
its global-sync descriptor route is `0x21a` rather than `0x211`. The fixture
therefore accepts the complete descriptor words, release word, packet origin,
and SRAM addresses instead of embedding one capture's values. Generating those
four allocation records and the corresponding descriptor is still required for
a topology-independent runtime.

Hardware acceptance with the checked-in configuration and master route
now covers:

- complete source, destination, and guard checks at counts 1, 52, 64, 65,
  512, 1,024, and the maximum 4,148 words;
- routes in both directions across physical rows and columns;
- four disjoint sender/receiver pairs in one exchange launch;
- one 1,024-word source stream received simultaneously by physical tiles 32
  and 53.

The fixture places worker sync storage, receive staging, executable plans, and
outgoing data in separate SRAM regions. Placing plan and source in the same
SRAM element was reproduced as `TEXCPT_CONFLICT` at the sender's `send`
instruction.

The multi-pass command-table runtime also completes an all-device reduction.
All 1,472 tile scalars, including physical tile 0's scalar, exchange and add
through 11 binary-tree rounds. The
compiler emits direction-specific point-to-point rows for one-to-one edges,
single-send multicast for fanout, and a conservative maximum of 16 groups per
epoch. The resulting 97 launches produce `1084128` on physical tile 0, and all
sampled tiles reach the terminal acceptance trap. In the current logical to
physical mapping the reduction root is physical tile 0, which is also the
configured packet origin; this is an exercised combined role, not a reserved
tile.

Native host-output lowering is partially recovered. Rust independently emits
the short and long host packet headers, arbitrary-range chunk plans, the tile-0
`[1, 0]` command-read XREQ, the `[2, 0]` D2H XREQ, and the source command,
payload, and zero-byte-close sequence. Disassembly of the SDK fixture shows
that D2H is a two-tile operation: physical tile 0 executes `sync 15` and sends
the XREQ while the source tile switches its incoming mux, sends the payload,
and waits in `sync 0`. A command-page H2D read precedes that operation.

The current `run-output` prototype reproduces page attachment order and staged
GS2 handoffs, but is not accepted as working: the attached page remains zero,
and some tail schedules leave the source in WAEX. Experiments with short versus
long packets, absolute versus relative source addressing, source addresses
`0x50120` and `0x60000`, exact SDK packet-table locations, and explicit command
ID 1 did not change that result. These variants were removed or kept behind
the structured assembler rather than accumulated as constants.

TDI reports both inactive and WAEX as context state zero. Architectural
exceptions are only classified when exception metadata is nonzero; attempting
retirement break or instruction injection against WAEX is not a reliable way
to recover its program counter.
