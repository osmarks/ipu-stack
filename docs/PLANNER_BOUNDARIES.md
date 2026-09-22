# Boundary layouts and private GEMM grids

The planner currently constructs GEMM, Add and GeLU implementations. Their
boundary layout vocabulary is established before implementation construction:

- Activations offer unreplicated row-major storage with whole-row ownership,
  and dense contiguous intervals aligned to eight bytes. Whole rows suit
  row-wise consumers; contiguous intervals balance pointwise work and memory.
- Both use as many nonempty owners as the tensor and configured tile count
  permit. Whole-row ownership spans batch dimensions as well as matrix rows.
- Explicit layout constraints remain constraints. Imported formats remain
  available. Unconfigured GEMM parameters retain the existing compact packed
  format; compute-grid replicas are preparation storage, not resident defaults.
- Compatible Add/GeLU chains propagate their layouts in both directions before
  the vocabulary is frozen. They cannot import arbitrary internal GEMM grids.

There is an existing copy-path limitation for packed matrices with sub-word row
tails. Their default is currently whole-matrix row-major storage, rather than a
split which lowering cannot realize. This is conservative and can prevent large
irregular matrices from fitting; improving that copy path is separate work.

For each input/output boundary combination, GEMM enumerates its internal grids,
tile orders and reduction ownership. Preparation, GEMM, reduction and the final
boundary conversion are one costed fragment. The common search frontier prunes
these fragments while they are generated, so the compiler need not retain every
MidGraph. Boundary combinations and chunks of internal grids are evaluated in
parallel; the same dominance comparison merges the chunk frontiers. A grid
with even one operand/result exceeding total tile capacity is rejected before
construction; no cycle-based grid shortlist is applied.

The approximation is the boundary vocabulary. A cheaper direct connection using
some other native layout is not represented unless that layout is offered. Costs
include the actual neutral-format conversions, not an assumed future fusion.
Exchange costs are still estimates before physical placement and scheduling.

On 2026-09-22, the 64-tile FP8 MLP smoke workload's mid construction fell from
35.6 seconds to 2.15 seconds. Hardware validation retained the same 0.002694
maximum absolute error. These are compiler timings, not device cycle counts.
The full-size SigLIP candidate enumeration still spends substantial time costing
copy/exchange geometry; the small-workload speedup does not establish full-size
planning scalability or device performance.
