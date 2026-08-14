# IPU21 supervisor exchange instruction reference

This reference records the exchange instruction behaviour used by ipu-stack.
It separates facts verified against the SDK assembler/disassembler and emitted
Poplar programs from timing interpretations inferred from those programs.

The executable decoder and consistency checker are in
`ipu_exchange::diagnostic`. Every row produced by `PhaseProgramBuilder` is
decoded and checked against its scheduled send intervals and incoming-control
events before codegen can use it.

## Architectural model

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

## Event time

The fields called `delay` or `count` below contain `events - 1`. Thus a zero
field advances one exchange event. PIC/XPIC effects carried by `delaypic` and
`delayxpic` happen at the terminal event. A control merged into `sendpic` or
`sendpicp` happens at the first send event; the remaining words continue to
leave on successive events.

This exchange-event timeline is distinct from the ordinary supervisor
instruction count. One instruction can represent many serial send events.

## Recovered encodings

The formulas use assembler operand order. `packed_source` contains a word
address or delta together with the low route bits used by the hardware.

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

send(words - 1, packed_source, format)
  0x78000000
  | (((words - 1) << 21) & 0x07e00000)
  | ((packed_source << 3) & 0x001ffff8)
  | format

sendoff(words - 1, packed_source, format)
  0x70000000
  | (((words - 1) << 21) & 0x07e00000)
  | ((((words - 1) >> 6) << 14) & 0x000fc000)
  | ((packed_source << 3) & 0x00003ff8)
  | format

sendpic(words - 1, control_selector, control_value)
  0x70100000
  | (((words - 1) << 21) & 0x07e00000)
  | ((control_selector << 18) & 0x000c0000)
  | (control_value & 0x3ffff)
```

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
| (((words - 1) << 21) & 0x07e00000)
| (packed_source << 3)
| format/control bits
```

The following inline word is not decoded or executed independently:

```text
(xpic_configuration_14_bits << 18) | pic_configuration_18_bits
```

The instruction must start at an eight-byte boundary. ipu-stack currently
uses the verified zero form of the extra prefix control/format fields and
32-bit words. Those fields remain named as raw fields in diagnostics until a
program differential establishes their individual semantics.

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
