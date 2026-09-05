# Hardware bring-up

The ordinary workspace test suite is offline. `ipu-tests` builds a trivial
package through `ipu-codegen`, round-trips it through `ipu-package`, loads it,
and checks supervisor completion and inactive worker contexts.

Run it with:

```sh
IPU_CONFIG=config.bin \
POPLAR_SDK_ENABLED=/path/to/poplar \
scripts/hardware-e2e.sh
```

Optional variables:

- `IPU_DEVICE` selects the device node and defaults to `/dev/ipu0`.
- `IPU_TEST_PACKAGE` selects the generated package path and defaults to
  `/tmp/ipu-trivial.ipuexe`.

Numerical GEMM smoke (64 active tiles):

```sh
cargo run --release -p ipu-tests --bin ipu-trivial-test -- \
  "$IPU_CONFIG" --sdk "$POPLAR_SDK_ENABLED" --workload gemm-smoke --tiles 64
```

Use `batched-gemm-smoke` for batched activations, `mlp-smoke` for the small
GEMM–GeLU chain, or omit `--tiles` for full-device GEMM. Full-device GEMM
currently spends several minutes scheduling exchanges during package building.
These smoke tests pack and check logical elements using compiler storage maps;
they require a fresh compilation and reject `--reuse-package`, since the binary
package does not serialize those maps.

## Configuration replay sequencing

The runner propagates startup and execution failures directly. It does not invoke
`gc-reset`, retry initialization, or replay workloads automatically. The temporary
reset workaround was removed after the configuration replay fix below survived
repeated startup and numerical checks.

The startup/run failures were reproduced with write-only configuration replay.
The captured `IPUCFG1` image records the SDK's MMIO writes, but omits its reads
and waits. `Device::replay_configuration` now reads CCSR after each write to drain
posted PCIe writes before the next configuration transition. Reading the written
register itself is avoided because some configuration registers have read side
effects. This adds no fixed sleep and applies to both CLI and library users.

The controlled comparison on the C600 was:

- Readback enabled: 100 attention-smoke loads and numerical runs, no retries.
- Readback removed again: failures returned immediately; the first run needed
  three resets, and another needed retries as well.
- An SDK-style firmware board-attach request alone did not prevent failures and
  was not retained as a fix.
- After an explicit SDK parity reset, readback enabled: 300 consecutive loads
  and attention-smoke numerical runs, no retries or failures.
- Twenty full-device projected-attention loads/runs passed without retries,
  each checking 839,808 outputs (maximum absolute error 0.000930).
- Canonical batch-one MLP passed without retries (maximum error 0.011719).
  GEMM and batched GEMM passed 262,144 and 786,432 exact checks without retries.
- All 133 release workspace tests pass. Clippy passes with the existing
  `too_many_arguments` and `type_complexity` allowances.

The supplied Linux 7.x module patch was inspected but not changed. Its exposed
PCIe error and parity counters were clear. These experiments isolate a reliable
configuration-replay condition; they do not prove every internal hardware timing
requirement or rule out unrelated kernel defects.

After removing automatic recovery, fresh GEMM, batched GEMM, canonical batch-one
MLP, projected attention, and attention-smoke builds all passed numerically.
Twenty further attention-smoke loads/runs passed consecutively using the same
binary with no reset path. All 133 release workspace tests and the Clippy check
above also passed after the compiler modularization.

## Completion-state checking

A September 5 MLP validation run completed host exchange but failed the old
supervisor-halt check. All 1,472 supervisors reported state 3 at
`ipu_stack_static_complete+0x10`; sampled completion words were 1 and SSR was
`0xa0` (`InvalidProgramCounter`). The existing terminal `br $m0`, with `$m0 = 0`,
can therefore remain visible as a fault instead of the all-zero status observed
in other runs. This agrees with the invalid-PC exception and context-status
encodings in the [IPU21 ISA](https://docs.graphcore.ai/projects/isa/en/latest/_static/TileVertexISA-IPU21-1.3.1.pdf).

The instruction bytes are unchanged. A named `ipu_stack_static_completed` label
now identifies that terminal branch. The checker accepts state 3 only when the
exception is `InvalidProgramCounter`, the saved PC matches that exact label,
and the tile's completion word is 1. Worker-state checks and numerical output
checks remain. Faults elsewhere are not accepted, and workloads are not retried.

An experimental completion breakpoint after final host output was discarded:
post-exchange debug stops were again not durable/observable. This matches the
earlier [breakpoint investigation](PATCHED_BREAKPOINT_RE_FINDINGS.md#hardware-results).
The checker change does not claim to resolve that underlying debug-interface
behavior; host-transferred output remains the basis of numerical verification.
