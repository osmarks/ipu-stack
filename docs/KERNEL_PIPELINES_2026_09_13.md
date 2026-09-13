# Add, layernorm and softmax instruction pipelines

The shared elementwise helpers now overlap loads with arithmetic without
changing their arithmetic order, worker ownership or tensor formats:

- Four-half add uses three bundles per quad instead of four, with explicit
  priming/draining. The final iteration does not load past either input.
- The aligned FP32 layernorm sum pass uses two bundles per eight halves
  instead of three. Centered variance uses five per four halves instead of
  six. The same helpers serve ordinary, residual, distributed and FP8-output
  layernorm. Centering and accumulation remain FP32.
- Residual addition with statistics issues its second vector add alongside
  the first output store: six bundles per eight halves instead of seven.
- Whole-row and segmented softmax clear accumulators once before the panel
  loop, rather than once per panel. MIX's discarded first read remains safe;
  subsequent panels reuse accumulators already read successfully by GACC.

A two-quad add variant was evaluated and discarded: its extra dispatch and
compiler register allocation penalized short broadcast rows. The retained
pipeline needs no additional layout or width selection.

## Direct hardware measurements

All entries below use one row, normal alignment and the same deterministic
inputs before/after. Times include the kernel call and worker setup.

| Kernel | Width / keys | Before cycles | After cycles |
|---|---:|---:|---:|
| Add | 1152 | 1644 | 1368 |
| Add | 2152 | 2652 | 2124 |
| Layernorm | 1152 | 7140 | 6732 |
| Add + layernorm | 1152 | 8784 | 8376 |
| Statistics | 1152 | 3384 | 2976 |
| Whole-row softmax | 768 | 12822 | 12258 |
| Segmented softmax | 729 | 5412 | 5232 |

Short single-wave adds can cost six extra cycles; three-row broadcasts are
unchanged or faster. The narrow/unaligned layernorm path is unchanged.
Instruction estimates were updated alongside the kernels. Some short-row
residual/statistics fusions now lose against standalone add + layernorm;
retaining the cheaper path is intentional. Fusion ownership/lifetime tests
use wider rows where the fusion still wins.

## Validation

`elementwise_check --fp8 --residual`: 874 cases pass, including widths 48,
96, 104 and 120 around pipeline boundaries, one/three rows, offset views,
in-place outputs, FP8 outputs and residual/statistics outputs. Existing cases
include constant and large-mean inputs; the variance calculation is unchanged.

Softmax: 108 paired old/new checks span random, constant and extreme finite
inputs, whole/segmented rows, 16/17, 64/65 and 729/768 keys, and 1/6/7 rows.
Another 16 cases cover 729/768 keys through 17 rows. Reference probabilities,
masked output and canaries pass. The fixture's default all-width package exceeds
its fixed code/data reservation; these runs split widths into pairs.

The codegen suite passed 282 tests, with two fusion-selection assertions needing
wider fixtures after the cost change (five ignored). All three residual-fusion
tests and all eight kernel-cost tests pass after that adjustment.

Artifacts: `artifacts/kernel-loops-20260913/`; `reference/` is the original
source, `elementwise-before.log` the initial baseline, `edge-check/` and
`edge-check.log` the final elementwise checks. `softmax-{whole,split}-*` holds
the paired softmax checks. Prototype add measurements are also retained but
are not the implementation's final timings.

## Full model

The same saved compatible-ownership BS1 recipe as the bias-GeLU experiment,
with 27 layers, fused QKV, FP8 weights at scale -4 and B1024 exchanges:

| Renderer-cropped metric | Before | After |
|---|---:|---:|
| Cycles | 11,070,624 | 11,001,942 |
| Time | 7.380416 ms | 7.334628 ms |
| FP32-reference cosine | 0.994221269 | 0.994221269 |

The combined improvement is 0.62%. This is a kernel-only comparison of the
loaded recipe, not a new layout search. Both resident inferences pass. The
first repeated layer is instrumented; the remaining repeats are aggregated.

Profile: `artifacts/kernel-loops-20260913/full/model.html`, raw data alongside
it, and exact placement under `full/memory/`. The previous profile is
`artifacts/bias-gelu-20260913/final/model.html`. `full.sh` records the build and
reference command; `full.log` records cosine and hardware results.

Other inspected hot loops (FP8 casts and reduction addition) already overlap
most of their memory/arithmetic work. Their remaining cost is largely staging,
loads/stores and layout choices; no additional instruction substitution was
retained for them in this pass.
