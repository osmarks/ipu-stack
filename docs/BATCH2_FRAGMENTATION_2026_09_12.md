# Batch-two allocation failures — 12 September 2026

Two captured failures have different causes. One admits a better placement under
unchanged constraints. The other exceeds the available storage at its live peak;
the final failure on a contiguous weight sequence obscured that capacity deficit.

## Exact-request replay

Temporary allocator instrumentation captured the complete requests for failing
tile 0, including lifetimes, alignments, classes, Repeat assignments, bank-conflict
edges, available address ranges and interleaved offset. It is removed from the
source tree. `artifacts/fragmentation-20260912/replay.py` replays those requests
without compiling the model or scheduling exchanges. It reproduces both original
fallback failures, including the failed request and remaining holes. Successful
placements are independently checked for lifetime overlap and memory-element
separation.

| Capture | Final failed request | Live storage lower bound | Available storage | Result |
|---|---:|---:|---:|---|
| `0-0.json`: proposed tile mapping | 27,648 B output-weight sequence | 458,368 B | 482,496 B | Better ordering fits this tile |
| `1-0.json`: later MLP-down cast | 82,944 B QKV-weight sequence | 515,200 B | 498,872 B | At least 16,328 B over capacity |

Available storage includes the host aperture, optimistically available to scratch.
The live-storage lower bound excludes alignment gaps and bank-separation overhead.
Thus the second deficit cannot be solved by rearranging addresses or scattering
weights under the current kernel/lifetime requirements.

The second recipe differs from the capacity baseline by removing operation 18
from `early_casts`. At the peak, the redistributed FP16 operand (125,952 bytes)
coexists with its FP8 result (62,984 bytes including access tail) and residual
activations. The initial lifetime-ordered allocator fails on scratch. Its fallback
places scratch earlier and later fails on resident QKV weights. The name in the
final error is not the source of the excess memory demand.

## Ordering experiments

The current fallback sorts primarily by descending alignment, then size. This
puts many short-lived, 32-byte-aligned buffers before the much larger, 8-byte-
aligned resident sequences.

The replay compares alignment-first, lifetime-first, size-first, three
size/alignment weightings, and two long-lived-first rules, each with first-fit
and best-fit holes. Only one tested combination fits the mapping capture:
allocations whose lifetime ends at model completion first, then descending size,
with first-fit addresses. This includes final outputs as well as resident
parameters; prioritizing only the resident parameters does not suffice.

This is evidence for a cheap additional local allocator trial, not proof that
the entire mapped model fits. A production change should retain existing
successful placements and try the alternative only on failing tiles, then
validate the complete package and its schedules. It adds no kernel constraint,
scratch arena, or whole-model planning retry. No production allocator ordering
has been changed in this investigation.

## What to prioritize

1. Try the long-lived-first placement order on failed tiles, retaining the
   existing successful allocation paths. The replay establishes an actual
   recoverable case under the same constraints.
2. For the late-cast candidate, keep the existing early-cast plan or reduce its
   concurrent staging footprint by at least 16 KiB on the overloaded tile,
   plus any bank/alignment margin. More precise support-aware peak accounting
   would distinguish this capacity failure from fragmentation earlier.
3. Do not start with noncontiguous Repeat weights. An additional replay split
   sequences into independently placeable members while retaining lifetimes and
   propagating bank-conflict edges to every member. None of the tested orders
   then fit either capture. That is not an impossibility proof for the mapping
   case, but the simpler ordering already succeeds there. The late-cast case
   has a storage lower bound ruling out a placement-only solution.

Raw requests, replay results, peak calculations and the diagnostic build log are
under `artifacts/fragmentation-20260912/`. The diagnostic build retains the working
baseline and passes its two resident FP32-reference checks. The saved search
state includes the rejected late-cast recipe in `visited`, allowing it to be
identified without inferring the recipe from the final allocation error.
