# ViT broadcast, normalization, fusion, and packing

Changes on main: `aa48f30`, `e145d21`, and `c12fd2e`.

## Representation and implementation

- AddF16 recognizes contiguous row broadcasts and uses direct half2 loads in
  the inner loop. General broadcasting retains its existing fallback.
- Row-major pointwise ownership partitions leading dimensions, including batch,
  before columns. Broadcast operands project out their singleton dimensions.
  Layernorm retains the original candidate and adds whole-row ownership.
- Short layernorms can split features across 2, 4, 8, 16, or 32 owners per row.
  This is an explicit mid algorithm: compute local FP32 mean and centered M2,
  exchange those statistics, then normalize and apply the affine parameters on
  each feature owner. Features remain F16. Centered statistics avoid subtracting
  two large, nearly equal second moments. Ordinary row-local LN remains costed.
- A mid rewrite fuses bias-add/GELU and residual-add/LN when layouts match and
  the sum has no independent consumer. Each replacement must improve the shared
  primitive cost model. Parameter copies may intervene only when they cannot
  overwrite the sources. Repeat yields and externally visible sums are preserved.
  The fused GELU reuses the existing assembly arithmetic; both LN forms share
  the same C++/supervisor source. No general expression compiler was introduced.
- Distributed packing is a separate candidate: gather smaller row-major panels
  onto more owners, pack locally, then transfer packed panels to the consumer's
  original layout. All three steps are ordinary mid primitives. Packed transfers
  use the existing micro-panel mapping, without an unpacked intermediate.
  The original plan remains available for final physical/memory selection.

Fusion and packing run after initial algorithm resolution, before physical
finalist selection. Thus their costs are compared with the shared mid costing,
not yet used to steer the original high-operation beam. This is a limitation,
particularly for memory-constrained searches, rather than a new planning layer.

## Full-size batch 2 hardware result

One-layer SigLIP So400m/14, 378x378 input, batch 2, FP8 scale -4, 1,472 tiles.
The reference check passed (maximum absolute error 0.092285, versus 0.092773
for the previous package). One deterministic hardware run per package.

| Measurement | Previous | New |
| --- | ---: | ---: |
| Cropped device runtime | 1,582,032 cycles | 1,296,198 cycles |
| Time per batch at 1.5 GHz | 1.054688 ms | 0.864132 ms |
| Time per image | 0.527344 ms | 0.432066 ms |
| Ordinary encoder LN owners | 729 | 1,458 |
| Ordinary LN maximum kernel duration | 30,990 cycles | 15,540 cycles |
| MAP LN compute owners | 1 | 64 |
| MAP LN attributed interval, including exchange | 34,950 cycles | 5,208 cycles |
| Long coefficient-packing owners | 160 | 960 |
| Maximum long coefficient-packing kernel | 23,190 cycles | 4,260 cycles |
| Remaining add kernel timeline union | 214,332 cycles | 72,096 cycles |

Overall runtime decreases 18.1% (throughput increases 22.1%). This measures the
combined change and resulting placement; it is not an isolated ablation of each
optimization. Kernel timeline unions overlap and must not be added to infer
wall-clock savings. The fused kernels are present in the profile, including
1,458 calls each to bias_gelu_f16 and add_layer_norm_f16. MAP LN has 64 calls to
each statistics/apply kernel. The selected long packing panels have 128 rows,
with five full panels and one 89-row tail per original owner.

The 41,052-cycle AMP-left tail (92 rows, 80 useful columns padded to 192) remains.
This implementation distributes long BlockMajor coefficient packing, not every
AMP layout conversion. Softmax and GEMM instruction bodies are unchanged.

Final package selection took 217.5 seconds. The selected ordinary placement was
finalist 0; the optional remapping challenger was not selected. Its final coarse
estimate was 1,252,961 cycles versus 1,296,198 measured. Compiler time is separate
from device runtime.

Artifacts:

- New: `artifacts/vit/fused-distributed-full-b2-fp8/profile.html`,
  `profile.ipuprof`, `profile-summary.json`, `operation-summary.json`,
  `kernel-summary.txt`, `model.ipuexe`, and `run.log`.
- Previous: `artifacts/vit/upper-region-full-b2-fp8/profile.html`.
- Smaller hardware checks: `artifacts/vit/row-broadcast-small-b2/`,
  `artifacts/vit/distributed-ln-small-b2/`, and
  `artifacts/vit/fusion-row-test/`. The last uses 64 tokens and explicitly
  exercises both fused kernels and distributed LN; maximum error 0.130371.

`--vit-image-size` now allows diagnostic image sizes without editing benchmark
source; it must be a positive multiple of the patch size. Model widths retain
normal or `--vit-small` defaults. Hardware tile counts must be multiples of 64.

Validation: 169 codegen tests passed, four ignored; six benchmark tests and the
codegen doctest passed. Clippy passed for both crates/all targets with the
repository's existing too-many-arguments and type-complexity allowances. Tests
cover ownership, distributed statistics/ABI, live-sum fusion rejection, and
packed-panel retile without unpacking, including a view mapping.

## Batch 4 feasibility follow-up

Build log: `artifacts/vit/fused-distributed-full-b4-fp8/run.log`.
The new execution paths do not by themselves solve the retained candidates'
placement constraints. Failures include 104,960-byte standard conversion buffers
and executable-region exhaustion during host support assembly. The latter's
`host programs (1 bytes)` diagnostic is a failed free-address probe, not a
measurement of the actual host program size.

A concrete fragmentation example is tile 92 in finalist 4's final placement:
free ranges total 259,688 bytes, but the largest is 92,672 bytes. Its requested
104,960-byte contiguous F16 conversion buffer cannot fit; another 131,072-byte
allocation is live at that event. Other tiles fail 81,920-byte attention AMP-left
allocations with 16-KiB alignment, or a 125,952-byte replicated GEMM input. These
are constraints of this allocation/layout selection, not an aggregate device
SRAM impossibility result.

The run completed unsuccessfully after all 16 variants: eight failed the initial
physical placement screen; of the eight fully scheduled variants, two failed
104,960-byte standard allocations, two failed 65,536-byte interleaved allocations,
and four exhausted executable-region space in host support assembly. Final
selection took 1,110,091 ms (18.5 minutes). There is no new batch-4 hardware
profile. No placement overrides or memory-budget relaxations were applied.
