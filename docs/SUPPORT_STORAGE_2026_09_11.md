# Consolidated package support storage

Branch: `baseline-local-planner`.

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
