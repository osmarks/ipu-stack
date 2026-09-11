# Consolidated package support storage

Branch: `baseline-local-planner`.

**Residency correction:** measurements below made before the “Persistent inference
and host-aperture reuse” section validated only one inference. The allocator then
allowed parameter storage to be reused after its last operator use. Those runs
were not evidence of a valid resident model. The corrected implementation now
passes three successive 27-layer inferences after one parameter upload; see the
final section for current profiles and measurements.

Package construction now reserves the linked section payloads, places host and
per-tile generated code into their remaining holes, and only then closes the
occupied executable SRAM elements to writable allocations. Exchange tables are
allocated after that closure. This permits unrelated executable objects to
share elements without letting exchange patching or tensors write into them.
Explicit tile-program packages still protect linked elements before admitting
caller-specified data addresses.

The runtime's 1760-byte state allocation remains at its existing address. Worker
and supervisor stacks grow downwards within that allocation. Its formerly
reserved tail up to the next 16 KiB boundary is now available for data; code
placement still begins at the next boundary. Host descriptors use part of this
tail in the 27-layer package.

## Measured 27-layer reservations

All runs use batch one, FP8 scale -4, fused QKV, and baseline-only planning.

| Version | Available tensor bytes per tile |
|---|---:|
| Before consolidation | 417376 |
| Consolidated executable storage | 433760 |
| Also expose runtime-state tail | 448384 |

The recovery is **31008 bytes (30.28125 KiB)**. It is capacity released from
support reservations, not a reduction in the selected tensor working set.
The final ranges are `368488..376832`, `507904..524288`, and
`525512..949168`. Profiling remains enabled.

GEMM inventory also now records the initialize/accumulate modes actually called.
Previously every row specialization retained both modes. Worker bodies in the
F16/FP8 assembler shared the last accumulate wrapper's section, preventing the
linker from discarding that wrapper even when it was unused. Independent worker
sections and exact mode retention reduce the linked image's end from 409864 to
403576, a **6288-byte span reduction**. The combined code still occupies the
same number of elements, so this gives code headroom but no further tensor
capacity in this particular build. With the current packing, another 9792 bytes
of code-span reduction would cross the next 16 KiB boundary.

## Validation and remaining failure

- 219 codegen tests and one doctest pass; four manual tests ignored. Clippy and
  workspace all-target checks pass for the final changes.
- Small three-layer FP8 ViT passes after each change, including exact GEMM mode
  retention (maximum absolute error 0.189453).
- Full-size one-layer ViT passes after support consolidation, before exact GEMM
  mode retention: maximum absolute error 0.077148; cropped runtime 970434 cycles
  versus 988890 before these changes. This reflects changed physical placement,
  not faster arithmetic. Profile: `artifacts/baseline-local-planner/support-packed-full1/model.html`.
- Full-size one-layer ViT also passes with final GEMM mode retention: maximum
  absolute error 0.077148; cropped runtime **970428 cycles (0.646952 ms)**.
  Final profile: `artifacts/baseline-local-planner/retained-gemm-full1/model.html`.
- The final 27-layer baseline still fails tensor placement after about 60 seconds.
  It clears exchange-table and executable-storage admission. A chronological
  attempt on tile 184 fails on a 1024-byte interleaved attention buffer with
  32768-byte alignment; size-ordered placement then fails on the 82944-byte
  QKV weight sequence. On another tile, a size-ordered attempt leaves an
  81328-byte contiguous hole for an 82944-byte weight sequence. These are
  different tile constraints, not one universal remaining deficit.

Logs are under `artifacts/baseline-local-planner/{code-packed27,support-packed27,retained-gemm27}/`.
This does not establish that shrinking code alone will make the current baseline
fit, nor that a different tensor layout cannot fit.

## Compact exchange scheduling

`--exchange-stream-words 256` selects address-ordered stream waves for package
construction (`PipelineConfig.exchange_stream_words`). The latency-oriented
scheduler remains the default. Cache recipes carry the selected mode through
phase splitting, relocation, local optimization and final placement. Ordinary
and paired candidates are ranked by maximum encoded row size, then total row
size and latency. Compact mode does not subsequently apply latency-only ordering
repairs that could undo its storage reduction. Hardware timing, dependency,
Repeat-source and instruction-alignment checks are unchanged. The stream-order
materializer is shared with offline replay.

| 27-layer baseline mode | Maximum exchange-table bytes | Planning until rejection |
|---|---:|---:|
| Latency-oriented | 63784 | 59.90 s |
| 256-word waves | 40488 | 32.77 s |
| 1024-word waves | 39912 | 33.21 s |

Both compact choices reserve three rather than four 16 KiB elements for the
exchange table. Available tensor storage rises another 16384 bytes, to 464768
bytes (453.875 KiB) per tile. Both still fail tensor placement; 1024-word waves
save no additional element, so 256 is the preferred tested compact setting.
These build times are individual samples, not isolated scheduling benchmarks.

The full-size one-layer 256-word run passes hardware/reference validation
(maximum absolute error 0.077148) at **1011180 cropped cycles / 0.674120 ms**,
versus 970428 cycles for latency-oriented scheduling: **4.20% slower**. Its
exchange tables are 40280 bytes, versus 62632 previously, and package planning
takes 60.51 seconds. Profile:
`artifacts/baseline-local-planner/streams256-full1/model.html`.
The small three-layer compact run also passes (maximum absolute error 0.189453).
219 tests, the doctest, Clippy and workspace all-target checks pass.

### Further machine-code reduction candidates

Inspection of the one-layer profile's called GEMM entry points and their current
cached ELF sections finds 23 called entries across 15 objects, containing 6372
bytes of supervisor wrappers. Used small/large row pairs contain 3112 bytes of
second worker bodies. These are not entirely removable bytes: shared dispatch
and row-parameter handling would need replacement instructions. A common
supervisor and shared worker body with a small row-specific setup are concrete
candidates; the inner AMP loop need not become an interpreter.

The compact 27-layer program still reserves about 21.6 KiB for generated tile
code and 9.1 KiB for host code. Per-word Repeat patch call sites can instead use
bulk descriptors plus a shared patch loop, trading some data bytes for fewer
instructions. That is separate from the already implemented bulk cross-phase
row patching. Per-tile kernel linking/reservations offer another opportunity but
require more changes to package address planning than sharing GEMM bodies.

## Shared GEMM dispatch

F16/F8 entry points now populate a small frame and call one supervisor per
precision/coefficient-load mode. Row, column and inner extents no longer duplicate
the coefficient feed and dispatch loop. Worker inner loops are unchanged.
The full-size one-layer compact run passes with the same 0.077148 maximum error.
Linked end falls 403576 → 399080 (4496 bytes); runtime rises 1011180 → 1012860
cycles, **0.166%**. This does not cross another executable-element boundary.
Profile: `artifacts/baseline-local-planner/shared-supervisor-full1/model.ipuprof`.

## Shared GEMM workers

Workers are now shared across rows, inner/column extents and coefficient-load
modes. Only precision and the output store permutation select a worker body.
The entry frame supplies quotient/remainder for six-worker row partitioning and
the panel byte stride. Initialize/accumulate and retained-panel entry points
still use the existing AMP inner loops.

The one-layer run passes (maximum error 0.077148), at **1026456 cycles**: +1.342%
versus shared supervisors, +1.511% versus the original compact baseline.
Linked end falls another 7512 bytes, to 391568 (12008 bytes total). Combined
executable storage releases one 16 KiB element: the upper standard tensor range
now starts at 458752 rather than 475136. The memory-address change also affects
exchange placement; the whole-model delta is not a pure kernel microbenchmark.
Profile: `artifacts/baseline-local-planner/shared-worker-full1/model.ipuprof`.

## Bulk Repeat patching

Each exchange now calls at most two descriptor loops: table-backed replacements
and exact arithmetic progressions. Per-word call sites and their old helpers
are removed. Immutable destination/value/step descriptors are appended after the
generated program's completion jump; their addresses are fixed up after code
emission. They do not consume writable exchange-table space. Duplicate patch
locations are rejected before grouping, and mixed descriptor relocation is
covered by the randomized codegen test.

For 27 layers, generated program storage (including the descriptors) falls
**21620 → 18884 bytes**, saving 2736 bytes. Together with shared GEMMs, the linked
span and generated program shrink by 14744 bytes. Available tensor storage rises
from 464768 to **481152 bytes per tile**, after element rounding. The build still
fails tensor placement on tile 184 for the 82944-byte QKV weight sequence; there
is no measured full-27 runtime. Admission reaches this failure in 37.05 seconds.
The exchange table remains 40488 bytes. Log:
`artifacts/baseline-local-planner/bulk-after-full27/run.log`.

Renderer-cropped hardware cycles, baseline-only planning and 256-word streams:

| Workload | Original compact | Shared supervisors | Also shared workers | Also bulk patches |
|---|---:|---:|---:|---:|
| Full-size, one layer | 1011180 | 1012860 | 1026456 | same path; no Repeat |
| Full-size, two layers | 1777680 | — | 1797174 | 1794522 |
| Small, three layers, 64 active tiles | 388776 | — | 390888 | 382656 |

Bulk patching alone saves 2652 cycles (0.148%) on the two-layer table-backed case,
and 8232 cycles (2.106%) on the small three-layer arithmetic case. Combined
changes are +0.947% and -1.574% respectively versus the original compact images.
All numerical checks pass with unchanged maximum errors: 0.086670 for full two
layers and 0.189453 for small three layers. The one-layer maximum remains
0.077148. Each binary was measured once; the small runs are not extrapolations
of full-model performance.

A separate packed-output F16 GEMM checks row-panel boundaries at local
R128/K64/C64. It passes with maximum error 0.000977. The kernel itself changes
**19944 → 20088 cycles (+0.722%)**, while other kernels in that benchmark retain
their cycle counts. This isolates dispatch/row-parameter overhead from the
physical-placement changes in the ViT comparisons. Logs/profiles:
`{before,shared}-packed-gemm/` under the same artifact directory.

Rendered profiles:
- `artifacts/baseline-local-planner/shared-worker-full1/model.html`
- `artifacts/baseline-local-planner/bulk-after-full2/model.html`
- `artifacts/baseline-local-planner/bulk-after-small-r3/model.html`

219 codegen tests and the doctest pass (four manual tests ignored), as do Clippy
and workspace all-target checks. Full-size two-layer profiles separately capture
the pre-sharing, shared-worker and bulk-patch binaries. Small three-layer runs
exercise arithmetic patching; the two-layer runs exercise value tables.

## Remaining 27-layer capacity (placement-only probe)

A temporary diagnostic replayed the same final low program through the normal
allocator, extending the standard range `475136..524288` downwards in 4 KiB
increments. All other ranges, layouts, tensor lifetimes, alignment/separation
constraints, and allocator strategies were unchanged. Hypothetical placements
were discarded: the diagnostic returned the original failure and never loaded
an image with tensors overlapping real support storage. Production sources and
the CLI binary were restored afterwards.

- Extra 0–12 KiB: still fails on an 82944-byte weight sequence.
- Extra 16–48 KiB: gets further but fails on a 27648-byte sequence.
- Extra **52 KiB (53248 bytes): every tile places successfully**.

This brackets the first success on the tested grid between 48 and 52 KiB, not
an intrinsic information-theoretic memory deficit. It suggests a practical
budget of roughly **64 KiB / four 16 KiB elements** for unchanged layouts and
this allocator. At +52 KiB, tensor capacity is 534400 bytes per tile. Exchange
rescheduling at the hypothetical addresses and a complete package build were
not tested, so this is an estimate of the remaining placement requirement.

The current linked span plus host/generated program storage is about 42 KiB,
occupying three standard elements. Small further code reductions alone cannot
bridge the measured gap. Better placement, less persistent/scratch storage, or
combined code/exchange storage reductions are needed.

The encoder baseline already uses FlashAttention with 64-key blocks (eleven
full blocks plus a 25-key tail), with online merges and FP32 intermediate state.
MAP uses a single 768-wide padded key block. This predates the code-sharing
changes. The old optimized September 9 ViT used full materialized attention;
the current baseline ranks candidates by local peak memory, then exchange row
storage, then cycles. The comparison runs disabled local optimization with
`--optimization-steps 0`.

Artifacts: `artifacts/baseline-local-planner/capacity-probe-full27/run.log`,
`probe.patch`, and `memory/`. The patch is diagnostic-only and is not applied.

## Reuse constrained-buffer element tails

The extra-capacity estimate above described the previous allocator's rounding,
not additional live tensor payload. `distinct_element` marked both operands of
a GEMM/loopback separation constraint, aligned their starts, and reserved each
buffer through the end of its last element. That unnecessarily excluded ordinary
buffers (including unrelated weights) from the tails.

Placement now retains element-aligned starts for constrained allocations but
reserves only payload plus the declared kernel access tail. Two live constrained
allocations still cannot share an element: the later one's aligned start would
intersect the earlier one's occupied bytes. Ordinary allocations can use the
remaining bytes because any operand requiring separation is itself marked and
aligned. Repeat groups retain their existing contiguous spans and member strides.
The special-case truncation at the final partial SRAM element is consequently
unnecessary and removed.

A diagnostic probe first confirmed all-tile placement at the actual ranges,
without scheduling or executing the hypothetical image. The production change
then builds the full 27-layer package, including relocated exchange schedules,
in 65.38 seconds (one sample while a one-layer build also ran). The package was
loaded for numerical validation. All 219 codegen tests, the doctest, Clippy and
workspace checks pass. Randomized allocation tests now explicitly verify element
separation between constrained live allocations alongside byte non-overlap.

The one-layer hardware/reference run passes with unchanged maximum error
0.077148 and **1026420 cycles**, versus 1026456 before tail reuse. No kernel
instructions or plan selections changed. Artifacts:
`artifacts/baseline-local-planner/tail-reuse-full1/` and `tail-reuse-full27/`.

## Full-model validation and host reference

The 27-layer, batch-one package now executes successfully. With the original
unquantized randomized inputs and weights and FP32 reference arithmetic, the
minimum output-embedding cosine similarity is **0.994212810**, exceeding the
required strict **0.99** threshold. Maximum absolute error is 0.410389.
This is one deterministic randomized model, not an accuracy result for trained
SigLIP weights. The comparison includes quantization and device arithmetic
approximations together.

Use `--reference-run --reference-fp32` for this reference. Without
`--reference-fp32`, the reference retains input, GEMM-operand and intermediate
rounding to the selected device precisions. Device inputs are encoded in their
actual storage formats in both modes. Complete ViT runs check cosine separately
for every embedding and reject nonfinite or zero-norm results; operator tests
retain their elementwise tolerances. Profiles are saved before numerical
validation so that failed comparisons still leave an inspectable trace.

The earlier 27-layer execution failed the old elementwise tolerance on one of
1152 outputs (0.459961 reference versus 0.707031 device). It did not fail to
execute. The later FP32 comparison above applies the requested whole-model
criterion instead.

The host-reference profile found excessive OpenMP/BLAS synchronization, scalar
attention products, FP8 conversion, and needless cloning of every layer's
weights into every Repeat iteration. Input generation and binding packing now
run in parallel; Repeat binds only explicit region arguments; attention uses
SGEMM; BLAS defaults to at most four threads while respecting explicit
OPENBLAS_NUM_THREADS/OMP_NUM_THREADS settings.

| Reference workload | Input preparation | Evaluation | Total |
| --- | ---: | ---: | ---: |
| Three layers, old device-precision reference | 10.828 s | 14.547 s | 25.375 s |
| Three layers, accelerated device-precision reference | 1.878 s | 7.675 s | 9.553 s |
| 27 layers, accelerated FP32 reference | 5.923 s | 20.960 s | 26.883 s |

The earlier 27-layer device-precision reference took 167.53 s combined.
That comparison changes reference precision as well as implementation;
the three-layer comparison is the like-for-like 2.66x speedup.
The accelerated three-layer reference passes at cosine 0.999580294.

The full 27-layer profile spans **20,030,544 cycles / 13.353696 ms**, cropped
at the renderer's normal start. Only the first Repeat iteration has detailed
kernel samples; subsequent iterations are included in the repeat-remainder
timing. Artifacts are in
`artifacts/baseline-local-planner/reference-fp32-full27/` (`model.html`,
`model.ipuprof`, `query.txt`, `run.log`).
Before/after host measurements and the perf capture are in
`reference-before-full3/` and `reference-after-full3/`.

Tests cover independent scalar attention agreement, FP32 versus quantized GEMM
reference behavior, Repeat argument binding, and per-embedding cosine rejection.
All ten test-binary unit tests, workspace all-target checks and Clippy pass.

## Current Repeat base-relocation audit

The 27-layer package has ten repeated phases with changing source addresses.
Six are compatible with one OUTGOING_BASE displacement per sender for the
whole phase: phases 7, 10, 31, 37, 40 and 47 (normalization preparation, QKV,
attention output projection, MLP normalization preparation, up and down
projections). They account for **317350 of 317433 source-word patches** per
Repeat transition across tiles. Downprojection alone accounts for 303323,
with a maximum of **225 words on one tile**.

The other four phases mix moving bias parameters and stationary sources:
79 sender rows in total, each with two displacement runs. They cannot simply
set one constant base for the entire phase. Cross-phase row-sharing patches
are a separate category; these counts do not claim to eliminate them.

These are structural eligibility counts, not a hardware validation of nonzero
BASE with every paired/multicast encoding. The audit is
`artifacts/repeat-address-audit/phases-bulk.rs`, with output in
`tail-reuse-full27/base-audit.txt`.

The element-tail allocator change preserves Repeat strides and these
opportunities. A future noncontiguous Repeat representation would need to
preserve common displacement within each sender/phase's relocation group.
Displacements may differ between iterations without breaking BASE eligibility,
but independently placed chunks in the same row can break it. Such placement
can also lose arithmetic patch compression and require address tables; it is
not needed to fit the present 27-layer package.

## Implemented OUTGOING_BASE relocation

Compatible Repeat phases now encode source offsets relative to an existing
iterated-parameter pointer and load that pointer into OUTGOING_BASE before the
barrier. They allocate no new pointer table or runtime helper. Every sender's
transfers, including stationary transfers, must agree on their displacement
through every iteration; otherwise the phase retains the previous word patches.
Paired offsets retain their eight-byte alignment. Ordinary subsequent rows
explicitly restore OUTGOING_BASE to zero. Cross-phase receive/source setup
patches remain independent and are preserved.

The full 27-layer FP32-reference run passes with unchanged cosine 0.994212810
and maximum error 0.410389. Cropped runtime is **19792236 cycles / 13.194824 ms**,
down from 20030544 / 13.353696 ms: **238308 cycles, 1.19%**. This includes any
schedule changes induced by the new row representation and placement, rather
than isolating instruction-level patch time.

The maximum generated-program reservation falls from 18884 to **18000 bytes**.
The uncompressed per-tile Repeat table bound falls from 25380 to **324 bytes**;
those tables were already represented arithmetically, so this is not a 25 KiB
SRAM saving. The exchange-table reservation remains 40560 bytes. The remaining
per-iteration instruction patches use the existing bulk path.

The durable hardware test is `--workload exchange-stress --exchange-pattern base`.
It checks distinct exact u32 payloads on all three iterations of eight cases:
ordinary unicast, multicast, paired multicast and multicast loopback, each with
standard and interleaved sources. Each case then executes an absolute-address
row, checking base reset. All cases pass, including final readback of 1792 words.
Artifacts: `outgoing-base-stress/` and `outgoing-base-full27/` under
`artifacts/baseline-local-planner/`; the latter includes the rendered profile.

## Exact per-tile placement viewer

`--memory-profile-directory DIR` now also writes
`placement-PID.json` and a standalone `placement-PID.html` for the final selected
package. The existing baseline/proposal estimate reports remain separate.

The address map uses the allocator's actual requests and final addresses.
Requests are constructed by the same function for allocation and diagnostics.
Each tile shows its first occupants in the main row and subsequent allocations
at overlapping addresses in rows below it. Alias groups remain one allocation;
they are listed on hover. Repeat sequence members are individually visible,
with stride padding and kernel access tails included. Lifetimes use the
allocator's inclusive event indices. Logical and physical tile IDs are shown.

Package support is explicitly marked as reserved capacity. Actual per-tile image
segments outside global support reservations (notably host descriptors placed in
otherwise unused tensor space) are included. White means no allocation in that row; an address is unused only if it is
empty in every row. Blank space in an additional reuse row is not additional
physical SRAM.

The canvas renders only the viewport. Controls provide byte-address zoom,
tile/address navigation, 12/16/24-pixel row heights (16 by default), optional reuse rows, allocation
highlighting, hover details and pinned selection with clear/reset controls.
Allocations use the cycle profiler’s green/red/gold palette plus purple for
support. Panel borders are retained; allocation bars have no outlines. Dragging across the map
or address ruler zooms to that range; right-click undoes a zoom, Shift-wheel
zooms around the pointer, and Escape cancels a drag.

The 27-layer example contains 633985 allocations/reservations, including
230450 additional reuse-row entries:
`artifacts/baseline-local-planner/placement-full27/memory/placement-48914.html`.
Its JSON is retained alongside it. This was a compile-only capture of the same
27-layer configuration used for the OUTGOING_BASE hardware measurement.

Validation: all 223 active codegen tests and ten reference tests pass, together
with the doctest and Clippy. The placement-report test checks every placed shard
and alias against its actual address and storage size, and rejects overlaps
between live records. Chromium checks cover hover, pin/clear, reuse toggling,
tile/address navigation, zoom, highlighting and reset on the full-model report.


## Persistent inference and host-aperture reuse

Parameter allocations now span the entire inference lifetime, including aliases
and all members of Repeat sequences. The mid memory estimator and its diagnostic
timeline retain parameters through the end too. Repeat-body estimates retain
invariant parameters even when their multiplicity is one.

In-place pointwise candidates cannot overwrite parameters or inputs with later
uses. Logical views preserve their source's read-only parameter provenance. A
parameter used as initial carried state gets an explicit fresh mid Copy before
Repeat, since carried storage is writable. Materialized temporary copies remain
reusable.

The host exchange aperture at **0x50000–0x58000** is now available to transient
standard-addressed tensors between host phases. Inputs, parameters and outputs
cannot borrow it, and unused aperture bytes are never offered to persistent
host descriptors or exchange rows. Packet headers are copied from permanent
descriptors before each active host exchange; incoming staging bytes are
replaced by the host transfer. Both online and offline placement enforce the
same restriction.

Programs that elide padding clears using the arena-wide finite-F16 invariant
continue to reserve the aperture: host protocol words do not preserve that
invariant. This is tracked explicitly when clears are removed. The mixed-FP8
ViT retains its required padding initialization and can use the aperture.
The coarse planner budget remains conservative; it is not increased by 32 KiB
in a way that would incorrectly admit additional resident parameter storage.

The exact placement viewer shows the reusable host reservation at the input
and output boundaries. Parameters show “whole program”; outputs remain live
through host readback. Panel borders, allocation colors, stacked reuse rows and
drag-to-zoom behavior are retained.

The corrected 27-layer package already fits without borrowing the aperture
(`artifacts/baseline-local-planner/resident-full27/`, compile-only). With aperture
reuse, the final full-model run passes **three successive host calls** after one
`initialize`, without SRAM reset or parameter re-upload. All three logical outputs
are bit-identical; each has FP32-reference cosine **0.994212810** and maximum
absolute error **0.410389**.

The final map has 161,375 resident parameter allocations, none overlapping any
other tensor allocation. All 1,472 tiles borrow the aperture: the occupied union
ranges from 5,632 to 32,768 bytes per tile, totaling **46,681,608 bytes** across
the device (about 30.97 KiB per tile on average).

The first invocation's cropped profile spans **19,340,886 cycles / 12.893924 ms**,
versus 19,792,236 cycles / 13.194824 ms in the previous one-shot BASE profile.
This is a comparison of complete placements, not an isolated aperture benchmark.
The reference test saves detailed profiling for the first invocation only.

Use `--reference-run --reference-fp32 --reference-inferences 3` to repeat this
validation. `PackageConfig::invocations` exposes the emitter's existing host-call
loop, and the package's `run` metadata records the count. Intermediate calls finish
at their host-exchange boundary; only the final call waits for terminal device
completion. Without this package option, the original one-inference program
correctly stops after its first call and cannot service a second request.

Current artifacts:

- `artifacts/baseline-local-planner/resident-final-full27/model.html`: execution profile.
- `artifacts/baseline-local-planner/resident-final-full27/memory/placement-58176.html`:
  exact placement, with its adjacent JSON.
- `artifacts/baseline-local-planner/resident-final-full27/run.log`: all three numerical checks.

Regression coverage includes persistent parameter lifetimes, direct/view/Repeat
parameter write protection, memory-estimate retention after last use, aperture
eligibility and reuse under both allocator orders, exclusion from auxiliary
storage, and exact placement-report overlap checks. The rendered full-model
report was also loaded and exercised in headless Chromium.

Validation: **226 codegen tests passed, 4 ignored; all 10 benchmark/reference
tests and the doctest passed**. Workspace/all-target checks pass. Clippy completes
with the repository's existing argument-count/type-complexity warnings. Browser
checks include forward/reverse map and ruler drags, cancellation, right-click
zoom undo, and preservation of panel borders.
