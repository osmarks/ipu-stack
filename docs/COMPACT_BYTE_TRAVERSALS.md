# Compact byte traversals (2026-09-09)

Storage views now retain affine runs, repeats and short sequences rather than
expanding every element into byte spans and sorting them. Layout factorization
is shared by semantic-order traversal and physical-address traversal. Whole
physical allocations are represented as a single run, including zero-sized
allocations whose pointer is still needed by kernel argument materialization.

Alignment, payload size, contiguity and common copy-fragment counts operate on
compact summaries. Repeated summaries are composed logarithmically. Physical
coverage combines ordered iterators; copy emission and physical exchange
preparation zip lazy span streams. Concrete transfers still have to be emitted
for scheduling. Fragment counting for two fragmented endpoints and signatures
that hash individual endpoints still walk their span streams. This change does
not introduce tile-level details into mid planning or alter layout search.

The former element-coordinate implementation remains only as a test oracle.
Random partial views across all supported element orders and F16/F32/FP8 verify
semantic ordering, physical address coverage, alignment and fragment summaries.
The public host-facing span collector remains an explicit compatibility boundary.

## Measurements

Single finalist expansion, release/native build, same retained candidate before
and after. Independent benchmark workloads used disjoint groups of eight CPU
cores; finalist expansion itself was serial. These are individual measurements,
not medians, and concurrent host work introduces noise.

| First finalist | Before | After |
|---|---:|---:|
| MLP batch 1 | 4.111 s | 0.344 s |
| MLP batch 2, Repeat 3 | 7.005 s | 1.509 s |
| FlashAttention batch 1 | 6.638 s | 1.401 s |
| Materialized attention batch 1 | 1.549 s | 0.959 s |

Times include expansion, simplification and analytical cycle costing, excluding
mid search, footprint screening, placement, exchange scheduling and linking.
Shard, kernel, local-copy, phase, logical-transfer and recipient counts match
for every corresponding measured finalist. No completed old ViT expansion
benchmark exists, so these numbers should not be extrapolated to the entire
ViT compiler run.

The diagnostic `--benchmark-expansion FILE` now defaults to four finalists;
`--benchmark-expansion-limit 0` requests the exhaustive retained set. Artifacts
and commands are under `artifacts/compact-traversal-20260909/`, with the baseline
under `artifacts/mid-expansion-20260909/`.

Hardware/reference checks passed for the small FP16 ViT (maximum absolute error
0.002930) and FP8 batch-2 MLP with three repeats (0.015625). The MLP profile is
`artifacts/compact-traversal-20260909/mlp-b2-hardware-fixed/profile.html`.

The full FP8 batch-2 ViT also builds and passes hardware/reference validation
(maximum absolute error 0.083496, diagnostic tolerances 0.2 absolute / 0.05
relative). Its rendered profile is
`artifacts/compact-traversal-20260909/vit-b2/profile.html`.
Final validation: 197 codegen tests passed, five ignored, the doctest passed,
and Clippy passed for codegen and test binaries with the repository's existing
argument-count/type-complexity allowances.

## Pre-placement follow-up

Commit 036a082 removes whole-device shard scans from primitive operand lookup:
input shards are indexed once by tile, preserving their original within-tile
order for broadcast and writable-alias matching. Physical traversal slicing now
writes into a scratch bounds array rather than recursively constructing nested
vectors of digit selections.

The first retained FlashAttention plan now expands in 0.933 s (previously
1.401 s). Materialized attention is 0.741 s (0.959 s), MLP B1 is 0.289 s
(0.344 s), and MLP B2 Repeat3 is 1.408 s (1.509 s). These remain individual CPU
measurements; timings vary. Component counts match all measured counterparts.
The 197 codegen tests, doctest and Clippy pass; no new device execution was
needed for the lookup/scratch-allocation changes.

Debug logging in `ipu_codegen::low::expand` now separates graph construction,
simplification and pre-placement analytical costing. First-plan measurements:

| Workload | Construct graph | Simplify | Cost graph |
|---|---:|---:|---:|
| FlashAttention B1 | 741 ms | 7 ms | 185 ms |
| Materialized attention B1 | 566 ms | 4 ms | 171 ms |
| MLP B1 | 197 ms | 2 ms | 90 ms |
| MLP B2 Repeat3 | 1150 ms | 4 ms | 253 ms |

The small difference from total expansion is logging and final bookkeeping.
Commands, perf samples, stage logs and result JSON are in
`artifacts/low-expansion-followup-20260909/`. The perf call stacks have incomplete
unwinding, so they identify hotspots but do not support precise inclusive
attribution of all expansion time. Broader use of low-graph costing would
benefit from reusing unchanged expansion work between candidates; these changes
do not implement that cache or change the planning search.
