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
