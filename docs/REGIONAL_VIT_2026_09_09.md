# Regional full-size ViT trial and MLP regression check

The workload is the existing complete single-layer SigLIP So400m/14 benchmark:
378×378 image, 729 tokens, width 1152, MLP width 4304, 16 heads, FP8 GEMMs
(scale -4), and the MAP head. This is not a 27-layer build.

## MLP regression check

All numbers below are hardware `profile-query` cropped `profileSpanCycles`, not
scheduled estimates. The workload is batch 2, three Repeat iterations with distinct
weights. The recent profiles instrument the first iteration and retain a timing
sample for the remaining two; the span still includes all three iterations.

| Build | Measured cycles |
|---|---:|
| Older strided-copy optimization | 856,230 |
| Recent pre-regional compact-traversal build | 846,120 |
| Current default planner, before FP8 tile-family fix | 846,120 |
| Current default planner, after FP8 tile-family fix | 846,120 |
| Experimental regional seed with fixed weights | 1,084,056 |
| Regional replacement with private weights and ownership fix | 805,944 |

The default planner has not regressed in this comparison. Both current default
builds passed hardware and numerical checks (maximum absolute error 0.015625).
The regional seed is 28.1% slower than the recent default. Its constrained grids
and canonical intermediate layouts give up performance; the earlier
1,154,313-cycle number was an estimate, not this measured duration.

The detailed first-iteration samples identify extra preparation work in the
regional seed. It adds `rearrange_row_major_to_amp_f16` calls taking up to
41,058 cycles; the default profile has none. Its longest strided-copy call is
26,418 versus 16,590 cycles, and its longest FP16-to-FP8 cast is 12,636 versus
3,642 cycles. GEMM calls remain around 48–49k cycles. These maxima are not
additive phase timings, but they show why canonical boundaries are an expensive
starting policy here.

Profiles/logs: `artifacts/regional-vit-20260909/mlp-default/`,
`mlp-fp8-subsets/`, and the earlier
`artifacts/regional-planning-20260909/mlp-b2-final/`. The recent historical profile
is `artifacts/compact-traversal-20260909/mlp-b2-hardware-fixed/profile.ipuprof`.

## ViT regions

| High-operation range (end exclusive) | Contents |
|---|---|
| 0:3 | Patch projection, bias, positional embedding |
| 3:18 | Attention normalization, Q/K/V projections and views, attention, output projection, residual |
| 18:25 | MLP normalization, up projection, GELU, down projection, residual |
| 25:39 | Final encoder normalization and MAP attention |
| 39:46 | MAP normalization, MLP, residual |

These are explicit input-graph annotations through the test CLI. Boundary formats
and ownership stay fixed during each local search; placement remains global.
The bounded trial keeps four local proposals for ranking, but globally validates
at most one per region and five overall. This avoids spending the entire budget
on the embedding before reaching the attention and MLP regions.

## Initial failures and fix

The initial full-size FP8 trial failed all seed attempts in 6.26 seconds, before
low expansion: memory rejection first, then no legal MAP output projection.
A full FP16 diagnostic also failed seed memory screening, at the encoder MLP.

The FP8 CLI used to discard all F16 GEMM families, including their smaller active
tile counts, and insert only one full-device FP8 family. For the seed's divisor
grids, a one-row 1152×1152 MAP projection has no legal 1472-tile candidate; the
1024-tile family works. FP8 selection now preserves the configured parallel-GEMM
tile-count alternatives. A focused regression test expands this MAP projection.
The MLP comparison above checks the effect of this fix separately.

With the fix, the second seed passed full executable package validation at
1,035,818 estimated cycles. Constructing and validating it took about 163 seconds.
The original trial allowed up to four global validations per region. It was
stopped after the seed and during the first replacement, then restarted with the
one-per-region budget because each validation still takes minutes. This was a
compiler restart, not a repeated hardware timing measurement.

A further restriction matters when interpreting the regional MLP result: the
current live-input contract includes parameter layouts. A region can stage those
weights into a different layout, but cannot change their original host-loaded
representation, even when the weights are used only inside that region. Thus an
unfortunate seed weight layout can make an otherwise better GEMM grid expensive
to reach. The user stopped this fixed-weight trial during MAP-attention validation to
implement region-private parameter retargeting. No final ViT hardware run was
performed with the fixed-weight trial.

## Private-weight follow-up

Private automatically laid-out parameters now participate in region search:
their host-loaded format can change, while activations, shared parameters, and
explicitly pinned parameter formats remain fixed. A regression test starts from
an inconvenient row-major weight, selects a different host format, and verifies
that no device conversion of that weight is introduced. It also checks shared
and explicitly pinned weights.

The trial also exposed a cost handoff bug. Finalization computes a new cost after
support placement and address optimization, but selection was still reading the
provisional schedule. The finalized cost is now stored on the selected plan and
used by ordinary and regional selection. A regression test rejects a provisionally
faster alternative whose finalizer reports a worse package cost.

For the stopped trial, finalized estimates were 991,915 cycles for the seed,
1,008,050 for embedding, 1,043,632 for encoder attention, and 1,097,723 for encoder
MLP. Thus the scoring fix does not change those three rejection decisions.
The previously logged 1,035,818 seed score was provisional.

New logs are `private-weights/run.log` and `mlp-private-weights/run.log` under
`artifacts/regional-vit-20260909/`. The new ViT run retains the same five regions,
one global proposal per region, and the usual FP8 reference tolerances
(atol 0.2, rtol 0.05).

The first private-weight Repeat proposal exposed an ownership bug and was rejected
before scheduling: a new body argument had offset zero while its parameter
storage group was rotated by 184 tiles. `LoweringState::value_in_storage_group`
now inherits the group's offset. A small rotated-parameter Repeat regression
checks both seed and replacement placement. The failed proposal did not replace
the feasible incumbent.

## Hardware results with private weights

The repeated MLP accepted one replacement at 893,778 finalized estimated cycles,
versus its seed's 1,157,979. Hardware and reference validation passed (maximum
absolute error 0.015625). The cropped measured span is **805,944 cycles**, 4.75%
faster than the current default's 846,120, and 25.65% faster than the earlier
fixed-weight regional result. Selection plus package validation took 156.3 s.
Profile: `mlp-private-weights-fixed/profile.html`.

The full-size ViT accepted the MAP-MLP replacement and passed hardware/reference
validation (maximum absolute error 0.085938). Its cropped measured span is
**993,132 cycles, 0.662088 ms**. The finalized estimate is 983,426 cycles.
Profile: `private-weights/profile.html`.

| Region | Finalized candidate estimate | Decision |
|---|---:|---|
| Seed | 991,915 | Feasible incumbent |
| Embedding/position | 1,003,022 | Retain seed |
| Encoder attention | 1,031,126 | Retain seed |
| Encoder MLP | 1,080,383 | Retain seed |
| MAP attention | — | Expansion rejected a rotated probe binding |
| MAP MLP | 983,426 | Accept |

The ViT process was already running when the ownership fix was made. The same
storage-group offset inheritance also applies to derived cast values; a separate
rotated FP16-probe/FP8-projection test now checks expansion and placement. A
compiler-only follow-up for region 25:39 is recorded in `map-ownership-fixed.*`.
It does not repeat device timing and does not validate executable support storage.

The complete ViT trial took **833.2 seconds** for selection and package validation.
This remains too slow for frequent iteration. The result demonstrates working
incumbent preservation and parameter retargeting, not inexpensive global
validation. Neither the ViT seed nor each rejected proposal was separately timed
on hardware, so the modest estimated MAP-MLP improvement is not claimed as a
measured speedup over that seed.

The ownership-fixed MAP-attention compiler-only follow-up completed successfully
in 130.2 s. Its proposal passed expansion, placement and scheduling and reduced
the provisional estimate from 1,035,818 to 1,017,524 cycles. This variant has not
undergone executable-support validation or hardware execution; it is separate
from the measured 993,132-cycle ViT package above.

Final validation: 209 codegen tests passed (five ignored), the documentation test
passed, and release Clippy passed for codegen and ipu-tests with the repository's
existing argument-count/type-complexity allowances. Commits include the FP8
family fix, finalized-cost selection, private parameter retargeting, and inherited
storage-group ownership. Existing user changes to TODO and untracked artifacts
were retained.
