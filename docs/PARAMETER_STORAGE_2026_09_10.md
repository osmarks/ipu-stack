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


The generalized four-layer trial passed hardware/reference validation with
maximum absolute error **0.107422**, at **1907838 cycles / 1.271892 ms**
(cropped runtime; 827 seconds to build and run). Its rendered profile is
`artifacts/vit-compact-parameters-20260910/layers-4/profile.html`.
The 27-layer trial reached outer Repeat composition but was rejected at
610184 bytes, 23224 bytes above the configured budget.

## Resident sequence lifetimes and compact region fallback

A resident region argument must remain allocated throughout the body: later
iterations still need the remaining sequence members after this iteration's
last local use. The shared memory analyzer now preserves that lifetime.
Body-local conversion buffers retain ordinary lifetimes and are counted once.
This removes a mismatch between body search and outer Repeat composition.

Small compact inputs use approximately 256-byte owner shards instead of
spreading a vector into tiny fragments over hundreds of tiles. The coarser
layout is used only when the complete sequence still fits the per-allocation
budget; large weights retain their broad distribution.

If the normal region search fails for memory or lack of candidates, it retries
with compact repeated parameter homes fixed at the boundary. Compute layouts
remain searchable. This reuses the same search and cache; disabling automatic
homes makes the fallback bounded to one retry. It preserves a coherent compact
storage choice which the mixed beam can otherwise lose one operator at a time.

With resident lifetimes corrected, the 27-layer search reached placement,
but every admitted finalist failed on a 442368-byte standard allocation.
The 16-layer trial also reached placement and failed. Those artifacts are in
`artifacts/vit-resident-sequences-20260910/`.

## Sequence bank separation

The allocator previously padded each sequence member to a complete SRAM
element when the current argument acquired a bank-separation constraint.
For 27 members that can turn small parameter shards into 442368 bytes.
The revised allocator reserves whole elements around the sequence as a unit,
while keeping members at their required kernel alignment and stride. Other
live allocations cannot share those elements. When multiple sequence members
have independent bank constraints, individual element spacing is retained.
This also avoids widening compact sequence strides merely because allocation
moves into the interleaved-address region.

Tests cover compact boundaries and cached failures, resident sequence lifetime,
dense sequence strides with bank separation in both address regions, and the
conservative path for independently constrained members. Hardware trials of
this allocator are recorded separately below when complete.

## Interrupted host build and bounded expansion

The session was reported OOM-killed while three large builds were running.
The interrupted logs under `vit-compact-region-20260910` and
`vit-compact-sequence-20260910` stop around 14:54 UTC. They do not establish a
hardware result for 16 or 27 layers. Host kernel OOM logs were unavailable in
the replacement container, so the exact victim/limit could not be verified.

Inspection found that finalist screening retained every expanded tile graph
until all candidates finished, before admitting only a few to placement.
Commit `ea892e0` screens in thread-pool-sized batches and retains only the same
placement shortlist after each batch. The ranking and compact alternative are
unchanged; tests verify identical results across batch sizes and budgets.
The temporary expansion cache is released before placement as well.

The restart in `artifacts/vit-bounded-screen-20260910` runs only 27 layers,
records periodic process RSS in `layers-27/host-memory.jsonl`, and saves elapsed
time and peak child RSS in `layers-27/host-time.json` on completion. This is
host compiler memory, distinct from the tile SRAM reports.

The bounded-screening run completed without host OOM: **25331244 KiB peak
RSS (24.2 GiB)**, **1277.8 seconds**. All five admitted 27-layer candidates
passed provisional placement. Scheduling rejected them with the old generic
Repeat row-compatibility error; no device execution took place.

Repeat relocation contained two avoidable dependencies. Its SENDPICP helper
reconstructed source offsets from timed instruction counts even though the
restart instruction encodes its address; `4835de8` reads that field directly
and makes compatibility errors identify their failing check. More importantly,
the scheduler's separate source list followed selection order, while the row
builder can insert a send into an earlier free interval. Repeat then paired
that unsorted list with instructions in execution order. The separate list is
now removed: timed activities are ordered chronologically and also supply
relocation sources. A regression inserts a late send first and an early send
second, then verifies exact source/instruction correspondence. A separate
27-iteration compact-broadcast test exercises dense sequence placement and
relocation together.

The decoder-only full retry under `vit-repeat-relocation-20260910` was stopped
intentionally after 150.6 seconds when the ordering bug was identified; it is
not a completed model test.

The send-order run (`vit-repeat-source-order-20260910`) passed Repeat relocation
and provisional placement, but final tile lowering rejected one-byte local
copies. It completed in 1401 seconds at 25620780 KiB peak RSS. Its hundreds of
millions of estimated low-level cycles described byte-wise FP8 permutations,
not useful GEMM execution; there was no hardware timing result.

The compact shards already had eight-byte boundaries. Their **row-major order**
was the problem: transposed FP8 panel population fell back to byte-sized copies.
Compact per-operator storage alternatives now preserve the selected consumer's
element order. An unreplicated matrix grid divides whole physical panels across
the available tiles, minimizing the maximum shard size and then total padding.
The host can populate those native parameter panels directly, and body-local
materialization redistributes them without a byte permutation. Row-major
compact storage remains usable where its conversions are supported, including
F16 under tight budgets. Normalization vectors retain the small linear-home
policy. This is a storage choice, independent of the compute grid.

A targeted FP8 test shows that raw row-major homes produce unencodable local
copies while native packed homes use encodable copies. Finalist expansion also
checks the existing local-copy kernel selector before admitting a plan to
placement, rather than discovering unsupported byte copies after scheduling.
Validation: 224 codegen tests passed, five ignored; Clippy passed with the
existing argument-count/type-complexity allowances.


## Compact micro-panel storage

The exact-order trial (`vit-native-parameter-homes-20260910`) was stopped
intentionally after rejected prefixes showed that large BlockMajor panels
still inflated persistent shards. Compact homes now use the compatible AMP
micro-panel order, allowing the existing panel exchange to regroup them into
the selected GEMM macro panels. A regression covers FP8 AMP and BlockMajor
consumers; another checks that the four ViT weight shapes stay within 10% of
the ideal unreplicated per-tile storage.

The bounded region fallback now constrains storage choices in the existing
operator search instead of freezing provisional row-major formats at the
boundary. This preserves consumer-compatible FP8 packing while forcing the
coherent compact alternative. Tiny linear homes also account for resident
multiplicity when choosing their chunk count, rather than relying on the old
boundary fallback to retain finer shards.

Validation: 224 codegen tests passed, five ignored; Clippy passed with the
existing argument-count/type-complexity allowances. The measured full-model
retry is under `artifacts/vit-panel-parameter-homes-20260910/`.
