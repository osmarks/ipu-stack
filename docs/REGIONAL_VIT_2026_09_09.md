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
| Experimental regional seed | 1,084,056 |

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
