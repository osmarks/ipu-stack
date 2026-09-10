# Persistent parameter homes and Repeat materialization

The previous memory reports described ownership before a late
`assign_parameter_tiles` pass. That pass already spread parameter homes, but
only after finalist selection: candidates rejected by memory screening never
reached it. It also balanced parameter-derived values as though all were
additional persistent allocations, and expanded linear layouts into row
fragments to count their storage.

Commit `6a0b954` moves home selection into candidate screening. It groups
persistent inputs by storage group, accounts for sequence multiplicity, and
uses resolved per-owner sizes. Large groups are assigned first using a bounded
rotation search. Unclaimed automatic parameters keep their provisional
distribution. All members of a Repeat sequence share one rotation, which is
propagated through region attachment. The rotated candidate is costed with its
real exchanges and allocations; a rotation that makes a fitting plan infeasible
is not accepted. The old post-selection mutation is removed.

Repeated parameters now have a compact-home alternative when the required
compute layout has larger shards than the provisional storage layout. The
original compute-ready storage remains a candidate. The alternative freezes
the compact input layout and uses existing mid conversions to materialize the
current iteration's operands. Conversion buffers have their own storage groups;
moving persistent homes does not also move those buffers away from compute.
This applies to replicated normalization vectors and concentrated GEMM weights,
without operator-name special cases or new kernels.

Only resident region arguments inherit the sequence's allocation count.
Previously `beam_allocation_copies` also applied it to derived values sharing
the semantic origin, multiplying body-local conversions by the layer count.
The correction counts a per-iteration broadcast/packing buffer once. Outer
Repeat composition still checks the complete set of resident sequence members.

## Validation

Regression tests cover separate homes for weight groups, sequence members with
one consistent rotation, temporary derivatives excluded from the home-load
heuristic, compact repeated layernorm parameters, and concentrated GEMM weights
materialized within the body under a memory limit. Existing randomized
parameter-owner lowering tests remain enabled.

The first full-size trial used home rotation on failed memory screens and
compact broadcast parameters, before generalizing the alternative to weights.
Its artifacts are in `artifacts/vit-parameter-homes-20260910/`:

* Four-layer batch-1 FP8 ViT passed hardware/reference validation, maximum
  absolute error **0.100586**. Cropped runtime: **1903758 cycles / 1.269172 ms**.
  The other three Repeat iterations span 1280640 cycles, **426880 cycles per
  encoder layer**. Profile: `layers-4/profile.html`. Build/run: 588 seconds.
* 27 layers passed QKV planning but failed at the MLP up-projection. The selected
  concentrated weight layout needed **497664 bytes per owner** for 27 layers,
  exceeding the maximum contiguous standard allocation. The smallest reported
  effective peak was 772152 bytes. This motivated the same compact-home
  alternative for weights rather than another GEMM implementation.

Trials of the generalized implementation are in
`artifacts/vit-compact-parameters-20260910/`. The interrupted preliminary runs
under `unclaimed-home-work/` attempted to rebalance unselected provisional
parameters; that unnecessary work was removed before restarting the trials.
