# Encoder padding, output fusion, and softmax

Workload: one So400m/14 encoder layer with embedding and MAP, 378×378 input,
729 tokens, width 1152, MLP width 4304, fused QKV, FP8 GEMMs at scale −4.
Times use the profile renderer's startup cutoff, at 1.5 GHz. The encoder span
is the interval from operation 3's first sample to operation 23's last sample;
overlapping operation/kernel totals must not be added to estimate this interval.

## Initialization belongs in the writer

The original down-projection preparation cleared up to 38 KiB on 54 tiles at
batch one (5,082 cycles), and 69 KiB on 64 tiles at batch two (9,048 cycles).
These were FP16 staging allocations consumed by the FP8 pack/cast. Coalescing
padding holes also cleared intervening logical data before copies overwrote it.

The cast now distinguishes physical row stride from readable columns, and
physical rows from readable rows within **each** batch matrix. It writes zero
FP8 padding without loading FP16 padding. The existing scalar ABI carries the
column bound and packs the two matrix-row bounds into the unused FP16
source-scale word. Matrix rows above 65,535 retain the unmasked path and its
required input initialization; this is an encoding limit, not a layout cutoff.

A small supervisor loop invokes the same worker for each batch matrix. The
worker retains the flattened output-panel stride but reads and clears only
this matrix's rows. Small matrices distribute whole panels among workers.
There is still one mid cast operation and one descriptor ABI.

Two alternatives were discarded after hardware profiling. A matrix loop
inside every worker increased register spills and added 31,854 cycles at batch
one. A separate padded worker restored that fast path, but its extra code
moved the batch-two linked support end address from 409,456 (`0x63f70`)
to 410,944 (`0x64540`), crossing a 16 KiB bank boundary and moving generated
code up one bank. These are absolute addresses, not code sizes; SRAM begins
at `0x4c000`, and the address span includes reserved regions and holes. The resulting
package ran at 981,288 cycles, with unchanged GEMM kernels also running slower.
Reusing the worker removes about 460 lines of generated assembly. It adds
per-matrix launch overhead but avoids carrying a duplicate packing loop.
Unpadded matrices use zero row-bound metadata and retain the single flattened
launch. Only matrices with actual row padding need separate launches.
Executable memory elements are reserved whole to avoid instruction-fetch/write
conflicts. The compact linked image ends at address `0x63ff8`, eight bytes
below the next element, so subsequent code changes can still affect placement.
For scale, tile zero contains 48,108 bytes of linked runtime/kernel executable
sections below this address, excluding separately allocated exchange/program
code; the end address itself is not a footprint measurement.

Low removes a padding-only clear only when every kernel reader ignores those
bytes, with no externally visible output, outgoing copy/exchange, or unsafe
repeat binding giving the allocation another role. Fused FP8 GeLU also skips
column padding, with separate input stride and physical output width. This
proof is local and works in mixed-precision arenas; it does not assume that
arbitrary FP32/FP8 bits are finite FP16 numbers. Other clears remain necessary.

The cast hardware checker covers 1,404 cases / 7,517,232 bytes, including NaNs
in unread columns and rows, two, three and eight batch matrices, masked/empty tails, bank
placements and output guards. All comparisons pass bitwise. The elementwise
checker passes 690 cases, including padded GeLU inputs and outputs.

## Final attention output

Intermediate attention accumulators remain FP32. The last merge reads the
previous accumulator explicitly and stores a separate FP16 result, including
output padding. Single-block attention uses the same entry point without
reading previous state. The redundant FP32 result and following C++ cast are
removed. Kernel specialization and instruction costing distinguish FP16 and
FP32 output.

A small full ViT passes reference validation. A two-head, 729-token streamed
attention test exercises eleven FP32 merge blocks and a final FP16 block, with
maximum absolute error 0.000031. Artifact: `artifacts/encoder-next/flash/`.

## Fusions across ownership changes

GeLU can commute through coordinate-preserving FP16 copies and run as a fused
FP8 producer on the cast's owners. Layernorm can do this only for complete
rows; its affine parameters use the existing pointwise broadcast tiling and
ordinary mid copies. Copy costs and resulting memory lifetimes are included.
The original producer input must have compatible storage, bypassed results
must have no other readers, and intervening writes to protected storage block
the rewrite. Independent arithmetic can intervene; repeat boundaries cannot.

Residual/statistics fusion can keep the residual on its existing owners while
copying the small FP32 statistics to different layernorm owners. The existing
residual redistribution remains an ordinary mid operation. Complete feature
rows are required; changing feature partitions needs a different statistics
reduction and is not silently treated as a local norm.

These are costed rewrites of existing copies and explicit values, not an
additional beam search or a new temporary-buffer representation. Availability
does not imply that a particular full-model plan selects every fusion.

## Softmax

Each 16-key panel now sums eight nonnegative FP16 exponentials per lane before
converting the partial sum to FP32. The denominator across panels remains FP32.
Stored exponentials are unchanged. This removes four issue groups per full
panel (47 → 43 including the maximum pass and loop overhead). The useful-work
annotation also follows the revised arithmetic rather than the old loop.

| Local shape | Previous cycles | New cycles |
|---|---:|---:|
| 6 rows × 729 keys | 16,884 | 15,804 |
| 17 rows × 729 keys | 50,244 | 47,004 |
| 17 rows × 768 keys | 49,302 | 45,846 |
| 7 rows × 729 keys, segmented | 23,490 | 21,954 |

The partial-sum rounding is intentional. Hardware tests check probabilities
against an FP64 reference, denominator error, masked zeros and write guards.
The 17×729 random case has maximum probability error 3.4e−6. Both whole-row
and segmented paths pass, including constant inputs, ±65504 inputs and odd
tails. There are 91 distinct tested shape/pattern/path cases across the
`artifacts/encoder-next/softmax*` directories with successful `run.log` files.

## Full-model measurements

| Version | Batch | Cropped cycles | Encoder span |
|---|---:|---:|---:|
| Exchange-audit baseline | 1 | 652,740 | 463,908 |
| Exchange-audit baseline | 2 | 959,136 | 715,188 |
| Final | 1 | 650,028 | 459,030 |
| Final | 2 | 965,118 | 713,736 |

The final batch-one package is in `artifacts/encoder-next/unmasked-b1/`;
it passes with maximum absolute error 0.065186. Its long down-projection
padding clears are absent. Remaining fills peak at 600 cycles in operation 12,
1,500 in operation 34, 228 in operation 38, and 240 in operation 31.

The final batch-two package is in `artifacts/encoder-next/unmasked-b2/`;
it passes with maximum absolute error 0.101074. Remaining fills peak at 966
cycles in operation 12 and 1,500 in operation 34. Both directories include
rendered `profile.html`, the source `profile.ipuprof`, and operation/kernel
query results.

Batch one improves 0.4% overall and 1.1% over the encoder interval. Batch two
is 0.6% slower overall but 0.2% faster over the encoder interval. These are
small end-to-end changes, despite the local softmax and padding improvements.
Packing between padded matrices still adds launch overhead, and code placement
remains sensitive to occupied memory elements. The batch-two profile must not
be described as an overall speedup.

The final measured batch-one plan uses direct FP16 attention output but does
not select FP8 GeLU/layernorm or residual/statistics fusion. The ownership
rewrites are available and costed; they do not force a fusion when its target
layout is unattractive. Batch two selects `add_layer_norm_moments` followed by
`layer_norm_apply`, but does not select FP8 GeLU/layernorm.

Earlier `padding-b2` and `combined-b2` experiments are not isolated per-change
timing comparisons: kernel sources evolved while those packages were being
compiled. `complete-b1` and `complete-b2` contain the superseded slow cast loop
and should not be used as final performance results. `verified-b*` contains
the separate padded worker; `compact-b*` still splits unpadded batch matrices.
Use `unmasked-b*` for the final implementation.

## Validation

The complete codegen library suite passed: 208 tests, five ignored. After the
cast worker-entry change, the 20 kernel tests passed again. Clippy passed for
codegen and test binaries with the repository's argument/type-complexity
allowances, and formatting checks passed. Hardware checks above cover casts,
elementwise kernels, softmax, full models, and multi-block attention.
