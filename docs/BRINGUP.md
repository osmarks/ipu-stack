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
origin run ahead and deadlocks active endpoints.

The Rust compiler now emits global-sync configuration rather than accepting
captured words on the command line. For C600 the repeated command-loop protocol
derives the canonical physical tile chain `[0, 1, 5, 13]` from the GSP hierarchy
extents `[1, 4, 8]` and builds the five-level descriptor route one two-bit step
at a time. The resulting four configuration-register writes travel in the
`.ipuexe` and override the generic
initialization capture before application loading. Packet and release SRAM
addresses are allocated after the fixture's live data, not supplied as target
constants. Hardware acceptance uses `artifacts/c600-init.ipucfg`, not a capture
from the tested exchange schedule.

SDK captures also show translated GSP chains, and translating all four selectors
plus the descriptor route has passed on hardware. Translation of the packet
emitter itself has not been established, so the compiler does not conflate
those two decisions. The command-loop runtime permits canonical physical tile 0
to perform payload work.

Hardware acceptance with the checked-in configuration and master route
now covers:

- complete source, destination, and guard checks at counts 1, 52, 64, 65,
  512, 1,024, and the maximum 4,148 words;
- routes in both directions across physical rows and columns;
- four disjoint sender/receiver pairs in one exchange launch;
- one 1,024-word source stream received simultaneously by physical tiles 32
  and 53;
- one 1,024-word source stream received simultaneously by all other 1,471
  tiles through the direct command-loop runtime, with sampled first and last
  words exact;
- one launch in which physical tile 0 multicasts 64 words to physical tiles 9
  and 32, then physical tile 32 sends the received buffer to physical tile 53.
  Both the independent multicast receiver and final relay destination were
  exact.

The fixture places worker sync storage, receive staging, executable plans, and
outgoing data in separate SRAM regions. Placing plan and source in the same
SRAM element was reproduced as `TEXCPT_CONFLICT` at the sender's `send`
instruction.

Multicast source and destination addresses are embedded in the plan
instructions. The direct runtime therefore clears `A7` for multicast senders
and `A4` for multicast receivers; point-to-point plans continue to use those
registers as relative bases. Treating an absolute receiver like a relative one
completes without an exchange exception but leaves its destination unchanged.

The multi-pass command-table runtime also completes an all-device reduction.
All 1,472 tile scalars, including physical tile 0's scalar, exchange and add
through 11 binary-tree rounds. The
compiler emits absolute single-receiver rows for one-to-one edges and
single-send multicast for fanout. The resulting launches produce `1084128` on physical tile 0, and all
sampled tiles reach the terminal acceptance trap. In the current logical to
physical mapping the reduction root is physical tile 0, which is also the
configured packet origin; this is an exercised combined role, not a reserved
tile.

The diagnostic runtime samples the tile cycle counter through worker 0 because
`$COUNT_L` is not accessible from supervisor mode. The worker samples before
and after each active plan; the reported interval therefore includes a fixed
worker/supervisor handoff around the plan. A direct two-receiver multicast gave
the following repeatable intervals:

| Words | Sender | Receiver physical 32 | Receiver physical 53 |
|------:|-------:|---------------------:|---------------------:|
| 1     | 204    | 552                  | 534                  |
| 52    | 258    | 606                  | 582                  |
| 64    | 270    | 618                  | 594                  |
| 65    | 270    | 618                  | 594                  |
| 1,024 | 1,230  | 1,578                | 1,554                |
| 4,148 | 4,350  | 4,698                | 4,680                |

Both the one-word and maximum-size cases preserved the expected first word on
both receivers. The compiler no longer estimates an epoch as `156 + words`;
it decodes each generated row's delay and send fields and takes the maximum
event horizon. This route-sensitive horizon is the basis for placing multiple
roles on one tile without inserting another BSP synchronization.

The scheduler colors only tile-role conflicts. Colors are emitted as timed
slots in one variable-length plan program, not separate epochs. Each row's
first event is rebased against that tile's preceding event horizon, active rows
are padded to the launch horizon, and only one `sync 3` and one return remain.
Groups whose source is exchange staging are topologically ordered after the
group that fills that staging allocation. The runtime's combined absolute role
clears both `A4` and `A7`; sender and receiver addresses are carried by the
generated instructions. The fixture derives plan stride from the longest tile
program rather than assuming the original nine-word row size.

`run-diagnostic` uses a deliberate coordinator completion trap after every
tile has stored its completion word. This makes TDI result collection
deterministic without a host delay. It is diagnostic termination, not the
production host-completion protocol; that still depends on completing native
host exchange.

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
