# IPU21 supervisor exchange instruction reference

This reference records the exchange instruction behaviour used by ipu-stack.
It separates facts verified against the SDK assembler/disassembler and emitted
Poplar programs from timing interpretations inferred from those programs.

The executable decoder and consistency checker are in
`ipu_exchange::diagnostic`. Every row produced by `PhaseProgramBuilder` is
decoded and checked against its scheduled send intervals and incoming-control
events before codegen can use it.

## Architectural model

### Multicast loopback (C600 hardware, 2026-09-06)

An ordinary 32-bit multicast can include its transmitting tile among the
receivers. The source tile executes the combined send and receive controls;
its own data returns through the fabric into the configured destination.
`Topology::multicast` and `PhaseProgramBuilder` support this combination.
Duplicate receivers remain invalid. Source-only multicast has not been
validated and remains unsupported. Paired multicast loopback is supported;
see the paired receiver checks below.

The `exchange-stress --exchange-pattern loopback` diagnostic checks the source
and every remote receiver against independently prepared expected data. Eighteen
random routes, each including the sender and remote receivers, passed at lengths
1, 2, 51, 52, 53, 64, 65, 128 and 512 words, each with standard and interleaved
destinations. Source buffers were standard SRAM at `0x65000`, destination buffers
at `0x60000` or `0x98000`, in separate banks. This demonstrates those bank
arrangements, not arbitrary overlapping or same-bank transfers.

Reproduce with:

```sh
target/release/ipu-trivial-test c600-init.ipucfg \
  --workload exchange-stress --exchange-pattern loopback \
  --tiles 1472 --exchange-cases 18 --exchange-max-words 512 \
  --exchange-max-transfers 1 --exchange-compute-delay 1 \
  --exchange-diagnostics --package /tmp/loopback.ipuexe \
  --device-lock artifacts/layout-sweep/device.lock
```

Results: `artifacts/exchange-loopback/loopback-bank-matrix.log`. An earlier
eight-route test with 128 words and standard destinations also passed.
The copy planner now folds standard-to-interleaved local copies into an existing
multicast when source view and copy order match, the spans are word-aligned, and
the local receiver introduces no additional message boundaries. It requires at
least two existing remote destinations; standalone copies, point-to-point sends,
same-class copies, and staging conversions retain their local kernels. This is
a conservative eligibility heuristic, not an exact comparison of both schedules.
The physical scheduler includes the self-receiver in its receive-bus and memory
hazards. Snapshot validation checks every repeated source address against the
self-receiver's memory elements. The same checks apply to paired loopback.

### Receive controls

There is no receive instruction. A receiver writes incoming words through two
independently timed configuration streams:

- PIC changes the incoming memory pointer.
- XPIC changes the incoming exchange mux/source.

These are logical exchange-event streams, not separate issue lanes. The PIC,
XPIC, delay, and send instruction families all execute in the supervisor/main
pipeline and cannot be dual-issued with one another.

Incoming data selected by XPIC is stored through the pointer selected by PIC,
which auto-increments. The names are not expanded in public Graphcore material;
“PIC” and “XPIC” below are therefore mnemonic names, not claimed acronyms.

The exchange fabric is statically timed. A sent word carries no destination
identifier. Receivers select the appropriate input at the scheduled time, and
other tiles ignore the word. Graphcore's synchronization patent describes the
same operations as `PUTi-MUXptr`, `PUTi-MEMptr`, `SEND`, a compact merged send,
and a two-word merged exchange instruction:

<https://patents.google.com/patent/US10963003B2/en>

Figure 5 also exposes the relevant physical pipeline. Instruction fetch, a
SEND's SRAM read, and an incoming exchange write can contend for the same SRAM
element even though the fabric route itself is stateless. The compiler must
therefore validate placement-dependent memory-element conflicts as well as
endpoint timing.

### Address deltas and possible stride controls

The SDK architecture register inventory names `INCOMING_DELTA` (0xa2) and
`OUTGOING_DELTA` (0xa8), each with an 18-bit address field shifted by two bits.
A C600 diagnostic sets both to 8, snapshots them, transmits 16 consecutive words
from 0x65000 to 0x60000 on two other tiles, and snapshots them again. The source's
outgoing delta becomes 0x65040; each receiver's incoming delta becomes 0x60040.
Unused deltas remain 8. The received words are contiguous. These registers are
mutable address offsets/cursors, not configurable per-word increments in this
mode. Snapshots happen inside the tile program before host exchange alters state.

Run `ipu-trivial-test c600-init.ipucfg --workload exchange-stress
--exchange-pattern delta --device-lock artifacts/layout-sweep/device.lock
--package /tmp/exchange-delta.ipuexe`. The test checks both payload and CSR
snapshots through host readback. Its source is
`crates/ipu-tests/src/exchange_stress/delta.rs`; logs are in
`artifacts/exchange-delta/`.

The complete SDK supervisor register list has no explicitly named stride CSR.
Other exchange registers include `INCOMING_SINIT`, `INCOMING_FORMAT`,
`INCOMING_DCOUNT`, `ANS_DCOUNT`, and `EXCHANGE_ADJ` (four-bit `COFF`, one-bit
`SEWS`). This inventory does not establish all their semantics or rule out an
undocumented mode. The public IPU21 Tile Vertex ISA, section 2.13, omits the
exchange interface details. No usable constant-stride exchange mode has been
established; software currently describes separate contiguous bursts.

## Event time

The fields called `delay` or `count` below contain `events - 1`. Thus a zero
field advances one exchange event. PIC/XPIC effects carried by `delaypic` and
`delayxpic` happen at the terminal event. A control merged into `sendpic` or
`sendpicp` happens at the first send event; the remaining words continue to
leave on successive events.

This exchange-event timeline is distinct from the ordinary supervisor
instruction count. One instruction can represent many serial send events.

## Recovered encodings

The formulas use assembler operand order. An initial `send` carries an
absolute tile-memory word address. `sendoff` continues the current outgoing
stream and uses its smaller address field as a source delta.

```text
delay(events - 1)
  0x40a00000 | ((events - 1) & 0x7ffff)

delaypic(events - 1, selector, value)
  0x60000000
  | (((events - 1) << 19) & 0x03f80000)
  | ((selector << 18) & 0x00040000)
  | (value & 0x3ffff)

delayxpic(events - 1, selector, value)
  0x64000000
  | (((events - 1) << 14) & 0x03ffc000)
  | ((selector << 13) & 0x00002000)
  | (value & 0x1fff)

send(words - 1, source_word_address, sctl)
  0x78000000
  | (((words - 1) << 21) & 0x07e00000)
  | ((source_word_address << 3) & 0x001ffff8)
  | sctl

sendoff(words - 1, source_delta, sctl)
  0x70000000
  | (((words - 1) << 21) & 0x07e00000)
  | ((((words - 1) >> 6) << 14) & 0x000fc000)
  | ((source_delta << 3) & 0x00003ff8)
  | sctl

sendpic(words - 1, control_selector, control_value)
  0x70100000
  | (((words - 1) << 21) & 0x07e00000)
  | ((control_selector << 18) & 0x000c0000)
  | (control_value & 0x3ffff)
```

`sctl` is the three-bit send-control field, not an opaque format number. Two
bits enable the two exchange-fabric directions independently; values 1 and 2
select one direction and 3 broadcasts in both. The remaining bit selects
64-bit rather than 32-bit items. Internal tensor exchanges use values 1, 2,
or 3 for ordinary words, and 5, 6, or 7 for paired transfers. A zero
direction field advances the outgoing event stream without putting a packet
on either route.

There is no separately encoded `delaypicp` mnemonic in the IPU21 encoder or
assembler tables. A directionless `sendpicp` performs the useful equivalent:
it advances an event interval and applies both controls without transmitting a
packet. SDK-generated receiver-only rows use this form.

`sendpic` continues the implicit outgoing stream while applying one incoming
control. Selectors 0 and 1 carry the two XPIC forms; selectors 2 and 3 carry the
two PIC forms. This is why an XPIC immediate is only 13 bits in that encoding,
despite the shared 18-bit value field.

### `sendpicp`

`sendpicp` is not an ordinary dual-issued bundle. PIC, XPIC, and send all use
the supervisor/main pipeline. It is a special aligned two-word supervisor
instruction. The PC skips the inline payload and advances by eight bytes.

The prefix is:

```text
0xf0000000
| ((pic_selector << 27) & 0x08000000)
| (((words - 1) << 21) & 0x07e00000)
| ((source_word_address << 3) & 0x001ffff8)
| sctl
```

The following inline word is not decoded or executed independently:

```text
(xpic_configuration_14_bits << 18) | pic_configuration_18_bits
```

`pic_selector` is the high selector bit for the PIC value in the inline word.
The XPIC value uses all fourteen high bits of that word. The instruction must
start at an eight-byte boundary.

Unlike `sendpic`, a transmitting `sendpicp` restarts the outgoing stream from
the absolute `source_word_address` in its prefix. The address must therefore
name the first word sent by this instruction, not the beginning of the larger
message. A directionless receiver-only form uses `sctl = 0`; its source address
is immaterial and ipu-stack encodes zero.

For example, this SDK-generated full-duplex instruction sends 43 words from
word address `0x14021` in direction 1 while changing both incoming controls:

```text
f54a0109 19015000        sendpicp 42, 0x14021, 1, 0
```

It follows an initial send from `0x14000` which has already transmitted 33
words. Encoding only the direction at bit 3 instead would select address 1 and
leave `sctl` zero, silently disabling this portion of the outgoing message.

An SDK receiver row for four consecutive 52-word messages demonstrates the
form directly:

```text
64000082                  delayxpic source 130
40a00032                  advance to the first cutover
f6600000 03014048         52 events; XPIC=192, PIC=0x14048
f6600000 03095000         52 events; XPIC=194, PIC=0x15000
f6600000 04016000         52 events; XPIC=256, PIC=0x16000
f7200000 19017000         58 events; XPIC=0x640, PIC=0x17000
```

The outgoing portion can be deliberately unconsumed. The patent explicitly
permits sent packets with no receiver; this is how a receiver-only row can use
the compact merged form without a separate outgoing graph edge.

## Evidence and diagnostic workflow

The encodings above are checked in three independent ways:

1. The SDK's `IPUArchInfo_py3` encoder/disassembler verifies individual prefix
   fields.
2. Small Poplar copy graphs supply complete SDK-generated rows, including the
   inline payload and scheduling choices.
3. `ipu_exchange::diagnostic` decodes ipu-stack rows and checks them against the
   phase builder's declarative send/control schedule.

For an ipu-stack failure, inspect in this order:

1. logical transfer spans and placed byte addresses;
2. scheduled sender/receiver event intervals;
3. decoded row instructions and incoming-control events;
4. row placement and all instruction-fetch/send-read/receive-write SRAM
   elements;
5. hardware PC, exchange error state, and read-back row words.

This ordering localizes a fault to lowering, scheduling, encoding, placement,
or execution instead of inferring all five from a high-level numerical failure.


## Paired sender lane reservation

Paired receivers do **not** need equal SRAM destination addresses. Each tile
programs its own incoming pointer; pairing shares the incoming item stream.
The previous equality condition in width selection and mapping costing was a
compiler restriction, removed on 2026-09-09. Eight-byte address alignment and
complete physical receiver pairs are still required by the supported path.

Hardware replay of the current ViT's QKV, MLP-up and MAP-KV preparation phases
passes with original, unequal receiver addresses and all eligible non-loopback
multicasts forced to paired width (32,768 sampled words per phase). Captures and
logs are in `artifacts/encoder-fusions/paired-phase/`. At unchanged placement,
production width selection reduces their scheduled horizons from 10,772 to
8,501, 15,854 to 13,276, and 13,499 to 9,229 cycles respectively. These are
exchange schedule lengths, excluding arrival skew and preceding computation.

Only 8, 9 and 7 receiver pairs respectively have mismatched addresses, but the
old all-or-nothing multicast check disqualified 579, 727 and 505 transfers.
Some transfers additionally include the sender pair. Splitting exceptional
receivers off without fixing address
eligibility barely improved QKV/MLP horizons: the slow receivers remained on
the critical path. No such splitting or address relocation is needed for the
independent-pointer correction.

The sender-pair exclusion was subsequently removed too. Paired multicast may
include the sender and its partner: borrowing the partner's transmit lane does
not exclude its receive role in that same transfer. The row builder merges
the receive updates and borrowed-transmit reservation, preserving both.
Source/destination memory-element overlap remains invalid and is checked for
every Repeat source address.

Hardware validation covers both sender lanes, standard and interleaved receive
banks, different pointers on the two receiver tiles, and a remote receiver pair.
The full QKV/MLP-up/MAP-KV phases also pass with all transfers paired (32,768
sampled words per phase). Scheduled horizons become 7,092 / 10,063 / 8,549 cycles
at unchanged addresses. Captures are `all-paired.json` and `loopback-banks.json`
in the artifact directory above; `replay-all-paired-*.log` and
`replay-loopback-bank*.log` record the checks.

Hardware replay on 2026-09-05 verifies that a paired-width sender borrows only
its physical partner's transmit lane. The partner can receive an ordinary
transfer during that interval. Reserving its receive lane as well needlessly
serializes full-duplex work. Local sends and other borrowed transmissions must
still be excluded. The source tile's own SRAM hazards remain unchanged; borrowing
the partner's transmit lane does not read the partner's SRAM.

A 64-tile probe sends 4,096 words from logical tile 0 to the receiver pair 2/3
using paired width, while tile 4 sends 2,048 ordinary words to tile 1. Borrowing
occupies 31..2079 and receiving occupies 165..2213; hardware readback passes.
Both pair orientations and the reversed construction order are covered by
`borrowed_transmit_lane_allows_receive_but_excludes_local_send` in codegen tests.
The complete historical and reconstructed MLP exchange phases also pass paired
hardware replay. See [the layout diagnosis](PROFILE_LAYOUT_DIAGNOSIS.md#paired-transfers-were-overconstrained).


## Composing multicast receive boundaries (2026-09-06)

Multicast does **not** require a whole route-latency gap between transfers.
The ordinary receiver's payload arrives 52 + 2 × physical-row events after
its XPIC source selection. Source selection and local pointer ownership must
therefore be scheduled separately. Absolute-address receive rows are also used
for repeated unicasts; this encoding is not a multicast-only timing constraint.

Two boundary errors were hidden by the old outer scheduler's receive-end guard:

* The reverse-engineered 52-word primitive programmed PIC one event early to
  avoid an aligned composite instruction. That works in isolation, but the row
  composer treated PIC setup as first payload arrival. In a four-transfer replay,
  the next destination received the previous transfer's last word and subsequent
  words shifted by one. PIC setup now uses the same payload-arrival offset as
  other lengths. The standalone neutral control is one event later; composition
  puts source teardown at the exact end of the word stream.
* The composer permitted a previous paired-format teardown and the next XPIC
  source selection in one directionless SENDPICP. A two-transfer replay faults
  with this combination. The SDK sequence uses separate controls. Primitive
  paired generation already rejected coincident format/source events; the same
  restriction now applies across primitive boundaries. The scheduler separates
  the events instead of delaying the next transfer until all old payloads arrive.

The second restriction is a hardware-tested encoding restriction of this
implementation, not a claim that every possible combined format control is
architecturally forbidden. Ordinary PIC-pointer plus XPIC combinations remain
supported. Other local constraints remain: outgoing streams cannot overlap,
receiver source/pointer windows cannot overlap, paired transmit borrowing reserves
the partner's outgoing lane, SRAM hazards can delay transfers, and composite
instructions need alignment. Those are distinct from an outer route-latency gap.

A third error affected ordinary-to-paired transitions: checking the new PIC
pointer against the previous payload end is insufficient. Paired format activates
**two events before** that pointer update. Activation must wait for the ordinary
payload to drain, or its last words are interpreted in the wrong format. Hardware
reports `TEXCH_RERR_MODI`. A 972-word unicast followed by a 352-word paired transfer
reproduces the fault on logical receiver 3 or 46; receivers 2 and 47 pass because
their role in the pair imposes a later source constraint. All four pass when format
activation, rather than pointer setup, respects the preceding ordinary payload.

### SDK comparison and repeatable hardware probes

`scripts/sdk-multicast-sequence.cpp` builds, saves and executes four SDK Copies
from logical tiles 0/4/6/8 to tile 2 and a configurable second receiver. Compile
with the SDK enabled:

```sh
g++ -std=c++17 -O2 scripts/sdk-multicast-sequence.cpp -o /tmp/sdk-sequence \
  -I"$POPLAR_SDK_ENABLED/include" -L"$POPLAR_SDK_ENABLED/lib" \
  -Wl,-rpath,"$POPLAR_SDK_ENABLED/lib" -lpoplar
/tmp/sdk-sequence /tmp/sdk-ordinary.poplar_exec 52 5
/tmp/sdk-sequence /tmp/sdk-paired.poplar_exec 352 3
```

Both pass on SDK 3.4.0 and C600 hardware (416 and 2816 words). Extract their tile
ELFs with `../ipu-exchange-re/tools/extract_gc_exe.sh`; disassemble with that
repository's `tools/sdk-exchange-oracle/instructions.cpp` and `libipu_arch_info`.
For logical tile 2 / physical tile 64, the compute-exchange sequence is:

| SDK sequence | XPIC source events | PIC pointer events | Format events |
|---|---|---|---|
| Four ordinary 52-word multicasts | 2, 54, 106, 158 | 56, 108, 160, 212 | none |
| Four paired 352-word multicasts | 1, 177, 353, 529 | 56, 232, 408, 584 | enable 54, disable 758 |

These are decoded instruction timings of SDK programs that passed hardware
checks, not timestamp measurements of the fabric. They prove tightly consecutive
multicast streams are executable. In particular, the SDK retains paired format
across all four messages. The composer now retains that format across contiguous compatible transfers,
tracking the format interval separately from XPIC source ownership. It removes the
old format teardown, redundant activation, and paired neutral/source bounce at
the shared boundary. A format change or a gap retains the explicit transition.
The ordinary-to-paired drain constraint remains in force. The exact source and
pointer events above are covered by a regression test.

`python3 scripts/exchange-boundaries.py --sdk SDK_DIRECTORY --output /tmp/boundaries`
replays ordinary count boundaries, paired sequences, distant and repeated sources,
and mixed ordinary/paired transitions through the production scheduler. Each case writes
its snapshot, package and hardware log. Run it from the repository root, with no
other process using the device. The receive destinations use distinct addresses,
so pointer-boundary errors cannot be hidden by contiguous output allocation.

Tighter paired source windows also exposed a cross-transfer SENDPICP encoding
hazard: a paired XPIC source selection combined with an ordinary PIC pointer write
faulted with an address error in the automatic MLP. The phase composer now only
combines ordinary XPIC controls with pointer writes. Paired source/neutral and
format controls remain separate incoming-control instructions (they can still
share an outgoing send through single-control SENDPIC). This restriction does not
restore the route-latency gap or the redundant paired-format transitions.
