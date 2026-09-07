# Materialized attention softmax improvements

## Hardware results (2026-09-07)

Batch-one SigLIP attention: 16 heads, 729 tokens, head dimension 72, FP16 products
and FP32 accumulation. Whole-model timings use the profile renderer's entry crop.

| Implementation | Softmax kernel cycles | Whole-model cycles |
| --- | ---: | ---: |
| Original | 40,326 | 234,378 |
| Four-wide maximum and MIX pipeline | 27,498 | 221,550 |
| MIX plus selected split-row execution | 21,780 | 215,832 |

This reduces softmax time by 46.0% and whole-model time by 7.9%. Each distinct
program was timed once. The arithmetic-only comparison used the old planner
binary with new assembly to isolate the kernel change. The final result uses
updated costing and automatic materialized product selection. All 839,808
constant-input model output checks passed (maximum error 0.000930); Gaussian
operator diagnostics also passed, with final sampled error 0.000040. Streaming
attention, which shares this kernel and merges state across key blocks, passed
a separate Gaussian diagnostic with final sampled error 0.000028.

Profiles and logs are under `artifacts/softmax-upgrade/`:

- `arithmetic-attention/`: arithmetic-only package, profile, and HTML.
- `final-attention/`: combined package, profile, HTML, and query summary.
- `guarded-attention/`: final source rebuild after the padding-safety guard.
  Its package hash matches `final-attention/`, so hardware was not timed again:
  `1dad1834aaa1397f1c81e0f8f3cff29609bb152d13be71eeb252ddc77ce186bd`.
- `guarded-diagnostic/`: final Gaussian checkpoint validation.
- `flash-diagnostic/`: Gaussian streaming-attention regression check.

## Arithmetic

The kernel still scans for a row maximum, then writes unnormalized FP16
exponentials and FP32 maxima/denominators. Normalization remains folded into the
attention merge.

`attention_softmax_panels.inc` supplies common arithmetic to both row schedules.
The maximum pass uses interleaved 64-bit loads and `f16v4max`. `f16v4mix` computes
four `scale * (score - maximum)` values with FP32 intermediates and simultaneously
returns the preceding four results as halves. Stores overlap conversion of the
exponentials to FP32; denominator sums remain FP32. The full-panel body drops
from 71 to 47 issue groups. The narrow masked tail retains its previous sequence.

MIX uses a half-precision scale (relative error about 0.0000658 at head dimension
72). It avoids an intermediate half-precision subtraction, which could overflow
before scaling. Device tests cover random scores, equal scores, and opposing
maximum finite FP16 values with overflow trapping enabled. Random long-row
probability error against the host reference was below 0.000004. These are ML
numerical checks, not bitwise-equivalence requirements.

`f16v8acc` remains an alternative summation design: it shares accumulator lanes
with MIX, so it is not an additional free optimization of this pipeline.

## Worker scheduling

Whole-row execution assigns rows round-robin to six workers. Seven or eight rows
need two rounds, leaving several worker slots unused. Split-row execution assigns
three contiguous, panel-aligned segments per row to workers and runs three local
stages: partial maxima, exponentials/partial sums, then final sums. There are two
additional local barriers, no new global phase or exchange, and no new allocation.

The existing extra 16 halves per row hold exactly the required state: two FP32
row values, three FP32 partial maxima, and three FP32 partial sums. Maximum and
sum scratch occupy separate ranges so workers cannot overwrite a maximum while
another worker still reads it. The worker stack pointer is preserved.

Splitting is not always worthwhile. On uniform 128-key inputs, an eight-row
kernel would grow from roughly 5,184 to 7,116 cycles. One shared analytical
selector in `kernel/cost.rs` prices both schedules, including launches, worker
rounding, and partial reductions. The selected mode becomes a scalar kernel-call
argument; mid remains a whole-device softmax operation. At 729 keys the measured
seven/eight-row split durations are 21,762/21,780 cycles; the model uses 21,780.
Costing is approximate, particularly for other input distributions and tails.
Profile useful-work metadata describes the new arithmetic and row reductions.

## Safety and validation

The padding-reuse pass now recognizes attention's embedded FP32 state. An arena
containing that state is not an all-FP16 arena: finite FP32 words can encode FP16
NaNs when reused as activation padding. The guard preserves required clears; it
did not change the measured attention package.

The kernel compilation cache now hashes recursive local quoted includes. Editing
the common assembly or split-worker include therefore invalidates the cached
object. A regression test covers include changes and cycles.

`softmax_check` verifies probabilities, FP32 row state, zero padding, and output
guards while timing a supplied reference kernel and the current kernel. It
reserves whole SRAM elements for output buffers so host-readback code cannot
share an element with SEND data. Example after sourcing the SDK:

```bash
git show d4f7629^:device/attention_softmax_f16.S > /tmp/reference-softmax.S
cargo run --release -p ipu-tests --bin softmax_check -- \
  --sdk "$POPLAR_SDK_ENABLED" --reference /tmp/reference-softmax.S \
  --keys 128,129,729,768 --split-rows --pattern extreme
```

Omit `--split-rows` to check whole-row execution; use `--pattern random` or
`constant` for the other input sets. Host regression tests cover schedule
selection and retention of padding initialization with embedded FP32 state.
Final checks: 135 codegen tests, four ELF tests, and the codegen doctest pass.
Clippy passes with the existing `too_many_arguments` and `type_complexity` lints
allowed; unrestricted `-D warnings` still reports those pre-existing warnings.
