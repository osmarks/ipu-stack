# IPU21 patched-breakpoint reverse-engineering brief

## Objective

Determine the complete debugger setup and resume sequence which makes a
memory-resident `trap 0` or `trap 1` instruction stop a supervisor or worker as
`TEXCPT_PBRK0` or `TEXCPT_PBRK1`. The public Tile Vertex ISA manual describes
the instruction and exception causes but omits the relevant quiescence and
running-program debugger material.

## Established facts

- IPU21 `trap 0` assembles to `0x41801000`. Its immediate low bit selects PBRK0
  or PBRK1 according to the manual.
- `IPUDebugLLD::insertPatchedBreakpoint`, at `0x5800` in the SDK 3.4
  `IPUDebugLLD.cpp.o`, only validates the address, reads and records the
  displaced word, obtains the architecture trap encoding, and writes it to
  tile memory. It makes no visible CSR or TDI enable write.
- `clearPatchedBreakpoint`, at `0x4ce0`, removes the host bookkeeping entry and
  restores the displaced word.
- Statically linked `trap 0` instructions were observed to retire rather than
  leave either supervisors or workers in a visible excepted state in the
  independent runtime.
- Setting TDI register 5 bit 0 did not help. That register is the TDI
  `DBG_ECSR` view and is not a patched-breakpoint enable.
- Directly querying `libipu_arch_info.so` gives the IPU21 supervisor CSR block:
  `DBG_CTL=0x73`, `DBG_ECSR=0x74`, and `DBG_ECLR=0x75`.
  `DBG_CTL.CHAN_EN` has mask `0x3`; `DBG_ECSR.EPCM` also has mask `0x3`.
  Attempts to enable channel 0 before a static trap did not produce a durable
  PBRK stop.
- The TDI register window is
  `0x30000 + physical_tile * 0x40 + register * 4`. Relevant indices are:
  context status 0, run-break 1, injected instruction 3, instruction owner 4,
  debug exception state 5, exception clear 6, debug data 7, TDI status 8, and
  TDI status clear 9.
- TDI instruction injection works for quiescent contexts. The independent
  driver can read PC, status, M registers, and SRAM from stopped contexts.
- `IPUDebugLLD::setPC`, at `0x3d50`, encodes an absolute `bri` from `pc >> 2`
  and injects it. On IPU21 that encoding is `0x40800000 | (pc >> 2)`.
- A pre-handoff TDI test proved that RBRK works with the normal bootloader and
  current `ipucfg`: requesting a supervisor retirement break before the final
  application release stopped tile 0 with `TEXCPT_RBRK`.
- `DBG_RBRK.ATOV` is atomic override. It permits a retirement break while the
  context is in protected atomic/exchange execution; clearing it did not make
  `trap 0` produce a durable PBRK exception.

## SDK artifacts

The SDK used for these observations is selected by this repository's `.env`.
The target-access archive contains `IPUDebug.cpp.o`, `IPUDebugLLD.cpp.o`, and
`RemoteIPUDebug.cpp.o`; an extracted object was inspected at
`/tmp/ipu-debug-archive/IPUDebugLLD.cpp.o`. Re-extract it rather than relying on
that temporary path.

Useful symbols in `IPUDebugLLD.cpp.o` include:

- `insertPatchedBreakpoint` `0x5800`, size `0x193`
- `clearPatchedBreakpoint` `0x4ce0`, size `0x186`
- `setPC` `0x3d50`, size `0x344`
- `tryExecuteInstruction` `0x3840`
- `enablePCMirror` `0x2250`
- `enableRBreak` `0x2320`
- `waitForException` `0x2b40`
- `enableIBreak` `0x6310`

## Questions to answer

1. Which higher-level debugger attach or session-initialization routine runs
   before `insertPatchedBreakpoint`, and what device/TDI/CSR state does it set?
2. Is a memory write through the debugger path materially different from an
   application-loader write, for example because it performs an instruction
   synchronization or invalidation operation?
3. How are PBRK system-call channels routed, acknowledged, and cleared? Trace
   uses of `DBG_CTL.CHAN_EN`, `DBG_ECSR`, and `DBG_ECLR` outside
   `insertPatchedBreakpoint`.
4. Does supervisor PBRK handling differ from worker PBRK handling? In
   particular, determine the TDI context-state transitions and exception-PC
   phase for each.
5. What exact sequence does the public debugger use for continue/single-step
   after a patched breakpoint: restore word, set PC, clear exception, execute
   one instruction, and reinsert the trap?

## Candidate fallback

A debugger stop can be built around the proven TDI retirement-break path: park
each supervisor at an operator boundary while its workers are inactive, request
RBRK, and retain a continuation address for host-injected resume. The general
post-load rendezvous and resume protocol is not yet proven, so this experiment
has not been retained as production code. It should remain opt-in and separate
from release execution if implemented before the proper PBRK sequence is
understood.
