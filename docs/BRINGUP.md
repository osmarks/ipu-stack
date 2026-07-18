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

The worker bring-up program uses `runall` once. All six workers execute barrel-threaded,
write through their distinct vertex-base values, synchronize locally, and let
the supervisor aggregate the words. There are no per-worker mailbox loops.

## Exchange runtime status

The Rust planner's point-to-point and multicast rows match the independent C++
oracle. The static runtime executes those plans directly with:

- one straight-line instruction stream generated for each tile;
- per-tile worker sync bases and local supervisor/worker rendezvous;
- `A6=1` on senders and receivers;
- `sans 0; sync ANS` on every tile without a plan in an exchange launch;
- plan code in a separately allocated executable SRAM region.

The loader's startup rendezvous releases all tile supervisors together. Each
later exchange launch uses the SDK sequence recovered from tile ELF
disassembly: every supervisor enters the internal sync zone, active tiles
perform the local worker rendezvous and execute a plan beginning with its own
internal sync, and inactive tiles execute the `sans`/ANS sequence. No host or
GSP transaction occurs between D2D launches. There is no command loop or
reserved synchronization tile; physical tile 0 participates normally.

Hardware acceptance with the checked-in configuration and master route
now covers:

- complete source, destination, and guard checks at counts 1, 52, 64, 65,
  512, 1,024, and the maximum 4,148 words;
- routes in both directions across physical rows and columns;
- four disjoint sender/receiver pairs in one exchange launch;
- one 1,024-word source stream received simultaneously by physical tiles 32
  and 53;
- one 1,024-word source stream received simultaneously by all other 1,471
  tiles, with sampled first and last
  words exact;
- one launch in which physical tile 0 multicasts 64 words to physical tiles 9
  and 32, then physical tile 32 sends the received buffer to physical tile 53.
  Both the independent multicast receiver and final relay destination were
  exact.

The graph packager places worker sync storage, receive staging, executable plans, and
outgoing data in separate SRAM regions. Placing plan and source in the same
SRAM element was reproduced as `TEXCPT_CONFLICT` at the sender's `send`
instruction.

Multicast source and destination addresses are embedded in the plan
instructions. The direct runtime therefore clears `A7` for multicast senders
and `A4` for multicast receivers; point-to-point plans continue to use those
registers as relative bases. Treating an absolute receiver like a relative one
completes without an exchange exception but leaves its destination unchanged.

The generated straight-line programs also complete an all-device reduction.
All 1,472 tile scalars, including physical tile 0's scalar, exchange and add
through 11 binary-tree rounds. The
compiler emits absolute single-receiver rows for one-to-one edges and
single-send multicast for fanout. The resulting launches produce `1084128` on physical tile 0, and all
sampled tiles reach the terminal acceptance trap. In the current logical to
physical mapping the reduction root is physical tile 0, confirming that tile 0
is not reserved by synchronization.

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
endpoint actions on one tile without inserting another BSP synchronization.

The scheduler colors only local endpoint conflicts. Colors are emitted as timed
slots in one variable-length plan program, not separate epochs. Each row's
first event is rebased against that tile's preceding event horizon, active rows
are padded to the launch horizon, and only one `sync 3` and one return remain.
Groups whose source is exchange staging are topologically ordered after the
group that fills that staging allocation. Generated active absolute calls clear
both `A4` and `A7`; sender and receiver addresses are carried by the
generated instructions. The packager derives plan stride from the longest tile
program rather than assuming the original nine-word row size.

`scripts/hardware-e2e.sh` compiles the static startup runtime and a separate
`add_u32` kernel, packages schedules, runs them through the direct loader, and
validates diagnostic bindings. The passing hardware paths cover an all-tile
affine permutation, multicast with a dependent relay in one exchange phase, and
18 seeded two-launch randomized graphs. Exchange rows only move data; following
compute phases call the linked kernel directly. The 11-stage reduction followed
by the unrelated dense permutation also passes with device-internal phase
synchronization.

`run-diagnostic` places its completion trap on the first output tile, falling
back to the first scheduled tile when the graph has no output binding. Every
tile stores its completion word first. This makes TDI result collection
deterministic without a host delay. It is diagnostic termination, not the
production host-completion protocol; that still depends on completing native
host exchange.

## Native host exchange

The prior command-dispatch implementation was removed with the device
interpreter. `package_graph` lowers host bindings to static per-tile host phases
before and after graph work. Inactive payload tiles use `sans 1; sync 1`; the
XREQ owner and target execute independently generated straight-line programs.

SDK archive members are named by physical tile. For a tensor mapped to logical
tile 100, the host operation is inline in `t_260.elf`, not `t_100.elf`. The
Rust target-operation encoders reproduce that member's 64-byte H2D and D2H
instruction and packet words. Physical tile 31 provides a second route oracle:
the XREQ owner is `target & 0x3d`. The two XREQ words are a 46-bit bitmap; the
target selects bit `2 * (target / 64) + ((target >> 1) & 1)`, with bits 0-23
in word 0 and bits 24-45 in word 1. Extracted SDK vectors for physical tiles
31, 81, 260, and 785 verify the row, pair, and word-boundary fields.

Transfers are split at 4-KiB attached-page boundaries. H2D destinations outside
the packet field's 16-KiB tile window use an automatically allocated staging
range and a generated target-tile copy. A 64-KiB round trip at `0x60000` passes
on hardware; D2H reads directly from that high address.

TDI reports both inactive and WAEX as context state zero. Architectural
exceptions are only classified when exception metadata is nonzero; attempting
retirement break or instruction injection against WAEX is not a reliable way
to recover its program counter.
