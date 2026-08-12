# IPU21 patched-breakpoint reverse-engineering findings

These findings supplement `PATCHED_BREAKPOINT_RE_BRIEF.md`. They were obtained
from SDK 3.4 target-access objects, the Graphcore LLDB Colossus plugin source,
the public IPU21 ISA manual, and C600 hardware tests.

## Corrections to the brief

### Trap encoding

IPU21 `trap 0` is `0x41801000`, not `0x4180100f`. The low nibble is the
`zimm4` operand, so `0x4180100f` is `trap 15` and selects PBRK1. Evidence:

- the SDK instruction encoder returns `0x41801000` for operand zero;
- `generated_ipu21_objdump_trap.txt` maps bytes `00 10 80 41` to `trap 0`;
- a fresh SDK assembly of `device/worker_support.S` produced those bytes; and
- `libipu21.a`'s `_doSyscall` uses `01 10 80 41`, or `trap 1`.

### Supervisor CSR instruction encoding

The CSR index is placed directly in the low byte of `get`/`put`. For example,
the target-access-derived operation for `put $C_DBG_DATA, $m0`, where
`DBG_DATA=0x70`, is `0x43008070`. Consequently:

```text
put CSR[index], $m0 = 0x43008000 | index
put CSR[index], $m1 = 0x43008100 | index
```

It is not `index << 4`. Any experiment using `0x43008730` to address CSR
`0x73` wrote a different register.

The previous `ipu-exchange-re` note calling CSR `0x73` `DBG_ECSR` is not
conclusive. Its actual diagnostic names the index `debugExceptionControl`,
ORs bit zero into it, and enables a separately configured IBRK. That proves the
IBRK registers at `0x80/0x81`, but does not distinguish `DBG_CTL.CHAN_EN` from
`DBG_ECSR.EPCM` at `0x73`.

## Debugger attach and memory writes

There is no hidden patched-breakpoint enable in the LLDB process attach path.
`ProcessColossus::DoAttachToProcessWithID` establishes process metadata,
refreshes the thread list, and halts the threads. Software breakpoint enabling
then uses LLDB's ordinary software-breakpoint machinery and the process
plugin's `DoWriteMemory` path.

`DoWriteMemory` selects a quiescent context and calls
`IPUDebug::writeTileMemory` once per word. The target-access implementation
uses an injected SRAM store. No instruction-cache invalidation or instruction
synchronization operation occurs. This agrees with the ISA's executable SRAM
model and means loader and debugger writes are not materially different for
instruction visibility.

`IPUDebugLLD::insertPatchedBreakpoint` itself only validates, records the
displaced word, obtains the architecture trap word, and writes it. It does not
configure a CSR or TDI register.

## Exception observation and resume

TDI context status has two distinct excepted values:

```text
2  TCTXT_STATUS_EXCEPTED_DBG
3  TCTXT_STATUS_EXCEPTED_NDBG
```

PBRK0/PBRK1 must therefore be sought in state 2, while the current FP-fault
fallback correctly uses state 3. A generic wait should accept both and then
classify `$SSR.ETYPE` or `$WSR.ETYPE`.

LLDB handles PBRK0 as a software breakpoint and PBRK1 as a system call. The
target-access resume mechanics are:

- `restoreThread` restores saved state, then clears all relevant exceptions in
  one TDI `DBG_ECLR` write and adjusts the run-break mask;
- `runThread` restores thread state, clears cached stopped flags, and updates
  the TDI run-break register;
- `singleStep` adds a single-step stop record between `stopThread` and
  `restoreThread`, then waits for the next exception; and
- `resumeFromSyscall` explicitly advances PC by four, restores the thread, and
  runs it.

The higher-level software-breakpoint step-over policy is LLDB-owned: temporarily
restore the displaced instruction, arrange a single step (the Colossus plugin
can use an internal software breakpoint), resume, and re-enable the original
breakpoint afterward. It is not implemented inside `insertPatchedBreakpoint`.

## Hardware results

An SDK `SupervisorVertex` containing `trap 0` stops in debug-excepted state
with `TEXCPT_PBRK0`, PC at the trap, and `$SSR = 0x20`. At that point `ANS` is
clear. `DBG_CTL`, `DBG_ECSR`, and `TDI_CTL` are zero and `DBG_RBRK` has only
`ATOV` set, confirming that a separate breakpoint-enable flag is not required.

The same instruction emitted inline in an ipu-stack tile program, immediately
after the selected device work and before host output exchange, stops all 1472
supervisors in debug-excepted state. Their PCs identify the trap in the
generated-code debug range, SRAM readback matches the package image, and
`$SSR = 0x20`, exactly matching the SDK case.

A trap reached after the final host-output exchange is not durable: execution
continues into normal supervisor completion. This placement must not be used
for checkpoints. SDK-generated code also places ordinary vertex calls after
an internal-exchange dispatch boundary rather than directly after its external
host exchange. Operator checkpoints naturally occur before host output and do
not require recreating that transition.

An inline checkpoint resumes by applying the ordinary software-breakpoint
step-over rule. PBRK leaves the saved PC at the trap: clearing `DBG_ECLR`
without replacing the instruction immediately traps again. Injecting
`put $PC, $m0` changes neither the saved exception PC nor its readback. The
working sequence is to replace the dedicated trap word with IPU21 `nop`
(`0x19e00000`) through an injected SRAM store, then clear the exception. A
single-invocation diagnostic package reaches each checkpoint once, so it does
not need to restore that trap before continuing. Alternating PBRK0 and PBRK1
provides an additional unambiguous checkpoint-generation marker to the host.

The complete 0x80000-byte device-configuration BAR was also compared after SDK
and ipu-stack loads and was identical. The behavior is therefore determined by
the instruction's runtime placement, not by `ipucfg`, bootloader selection, or
a hidden device configuration write.

The following did not make PBRK durable:

- TDI `DBG_ECSR` bit zero;
- `TDI_CTL.SEPEX` set before the final application-release mark;
- supervisor writes of `3` to correctly encoded CSR candidates `0x72` through
  `0x7f`; or
- extra instruction spacing between the CSR write and `trap 0`.

These negative tests were performed with temporary probes, which were removed.
No diagnostic code remains in the runtime.

## Pre-handoff RBRK result

The decisive pre-handoff test succeeded. Immediately before the secondary
loader's final application-release mark, `DBG_RBRK` read as `0x40000000`.
Setting supervisor-context bit zero and releasing the application stopped
physical tile 0 in debug-excepted state (`TCTXT_STATUS = 0x10`) at PC
`0x4c0f8`, classified as `TEXCPT_RBRK`.

`DBG_RBRK.ATOV` is the atomic override bit. SDK strings describe enabling a
retirement break with an “override atomic” option and setting ATOV while an
exchange is active. It allows an RBRK to interrupt execution that would
otherwise be protected as atomic; it is not a global exception-enable bit.
Clearing ATOV while requesting RBRK made no difference to the unresolved
post-load stop behavior, nor did preserving it.

This establishes that the normal bootloader and current `ipucfg` permit durable
TDI retirement-break and patched-breakpoint exceptions. Static checkpoints
should use an inline `trap 0` before host output exchange, wait for supervisor
state 2, verify `$SSR.ETYPE == TEXCPT_PBRK0`, and read tile SRAM through an
inactive worker context. The temporary SDK oracle and runtime probes used for
these experiments have been removed.
