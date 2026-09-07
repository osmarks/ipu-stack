# Materialized attention softmax investigation

The batch-one independent-product profile in
`artifacts/attention-product-grids/independent/summary.json` measures 40,326
cycles per softmax invocation on all 1,472 tiles. The phase spans 42,666 cycles
including start skew, out of 234,378 cycles for the cropped workload. This is
not primarily an imbalance between tiles.

`device/attention_softmax_f16.S` scans each row for its maximum, then writes
unnormalized FP16 exponentials and FP32 maxima/denominators. Normalization is
already folded into the final attention merge. There is no separate division
pass to eliminate.

## First experiment: packed affine transformation

For every two scores, the current exponential pass converts to FP32, multiplies
by the scale, adds the negative scaled maximum, converts back to FP16, computes
FP16 exponentials, converts those to FP32, and accumulates their sum. Stores and
subsequent loads overlap some arithmetic. The analytical model counts 71 issue
groups per complete 16-key panel, including the maximum scan.

IPU21 ISA 1.3.1 section 3.7.3.3.24 documents `f16v4mix`: four half inputs from
each source, two half coefficients in TAS, FP32 dot-product intermediates, and
simultaneous readout of the previous accumulator result as four halves. Set
the coefficients to `scale` and `-scale`, and broadcast the row maximum as the
second source. This computes `scale * (score - maximum)` without first rounding
the difference to FP16. A pipeline can potentially replace eight preparation
instructions per four scores with one mix instruction, plus warmup/drain and
any register-management overhead. Eight writable arithmetic registers make the
actual schedule important; instruction savings are not yet measured speedups.

The scale becomes FP16 (relative error about 0.0000658 at head dimension 72).
A host check using 2,048 Gaussian rows of 729 FP16 scores at each standard
deviation 1, 8, 32, and 128 gave maximum absolute normalized-probability changes
of approximately 0.00000114, 0.0000171, 0.000206, and 0.000282 respectively.
This used NumPy exponentiation rounded to FP16, not the IPU exponential or its
rounding controls; it is a plausibility check, not device validation. Preserve
consistent stored maxima for the merge, and test odd keys and overflow modes.

A simpler fallback uses packed multiply/add instructions. Removing just two
issue groups per pair saves 16 per full panel: about 8,640 cycles for two row
rounds and 45 full panels, before changed setup/tail costs. That suggests roughly
31,700 rather than 40,326 cycles, but is an instruction-count estimate only.

## Alternative: direct accumulation of exponentials

Section 3.7.3.4.2 documents `f16v8acc`, which adds eight half values into eight
FP32 accumulators. It could replace per-pair conversion and sum instructions,
with one final accumulator read/reduction. Some current operations overlap
loads/stores, so auxiliary instruction counts overstate cycle savings.
It also overwrites accumulator lanes used by `f16v4mix`; these should initially
be evaluated as alternative designs, not combined savings.

The maximum scan can separately use `ld64` and `f16v4max` on its interleaved
input. This requires suitable alignment and a correct narrow tail, but no new
layout. The current loop processes only two halves at a time.

## Second experiment: split rows among workers

There are 11,664 logical rows (16 heads times 729 queries), approximately eight
per tile. The kernel assigns whole rows round-robin to six workers. Seven and
eight rows both require two worker rounds, matching the flat measured duration.
For eight rows, only two workers have a second row: useful row-slot occupancy
is 8/12. The profiler's near-100% lane-occupancy estimate measures logical
padding, not this worker imbalance. The ISA hardware-context section specifies
that workers cannot claim the execution slots of exited workers.

Splitting each row into three panel-aligned segments produces 24 tasks for an
eight-row tile, four per worker. This needs local partial maxima and sums plus
local synchronization/relaunches; it does not require a global exchange or a
change to the whole-device operation. The ideal work-balancing bound is about
two-thirds of the current row work, before those costs. Seven-row tiles and
tails require explicit accounting. Merely assigning 12 whole rows to fewer
tiles still takes two row rounds and does not improve the critical path.

Prioritize the arithmetic kernel first: it changes neither planning nor
ownership. Then measure whether split-row overhead is justified on the faster
kernel. Update analytical costing and useful-work metadata with whichever
instruction sequence is actually selected. No proposed kernel has yet been
implemented or benchmarked on hardware.

## Arithmetic implementation and hardware check (2026-09-07)

Implemented four-wide maximum loads and pipelined `f16v4mix` affine preparation,
with FP32 denominator accumulation and the existing masked pair tail. The inner
full-panel cost falls from 71 to 47 issue groups. Cost and useful-work metadata
now describe this sequence.

The standalone `softmax_check` binary checks normalized probabilities, FP32
maxima/sums, zero padding, and output guards against a host reference while timing
both a supplied old source and the new one. The first sweep covered 80 shapes
and passed; maximum probability error in the long-row random cases was below
0.000004. Its buffers reserve full SRAM elements to prevent host-readback code
from sharing an element with SEND data.

With unchanged attention planning/costing for an isolated kernel comparison,
materialized attention fell from 234,378 to 221,550 cropped cycles. Softmax fell
from 40,326 to 27,498 cycles; all 839,808 model output checks passed. Artifacts:
`artifacts/softmax-upgrade/arithmetic-attention/`. This comparison intentionally
uses the original planner binary with the updated assembly, so its profile
useful-work metadata still describes the old kernel.
