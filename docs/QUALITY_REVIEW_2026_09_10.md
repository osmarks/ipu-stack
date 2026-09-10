# Review after the September 8 cleanup

Baseline: `42ad0eb`, the last production-code cleanup, through `9deab80`.
The review follows the intervening changes by subsystem: graph search and
Repeat attachment, memory estimates and parameter ownership, copy traversal
and caching, exchange scheduling/relocation, package allocation, FP8 and
normalization kernels, and their benchmark/profile interfaces. The changes
include substantial new functionality; deleting it wholesale would undo
requested features and measured improvements.

## Corrections and simplifications

* **Tensor memory and estimated exchange storage are separate.** Previously,
  `MemoryPeaks.standard` and `.total` included estimated rows. Capacity checks
  and profile rendering subtracted them again, and refreshing row estimates
  subtracted the old estimate from an already saturated total. Tensor liveness
  now owns tensor peaks; row refresh changes only the row estimate. Ranking
  explicitly combines storage where it previously used a combined objective.
  Existing capacity decisions and objectives are covered by the memory tests.
  Profile format version 3 documents the changed peak-field meaning.
* **Package selection uses the final placed cost.** Final placement already
  recomputed the schedule cost, but only logged it. Selection compared costs
  from provisional placement. The finalizer now returns its computed cost
  with the artifact; acceptance and ranking consume that result together. A
  regression supplies equal provisional plans with different finalized costs
  and verifies that the better finalized package wins. No extra scheduling or
  costing pass was added.
* **Occupied-range merging has one implementation.** Diagnostic package
  assembly duplicated it for tile-local storage and host descriptors, and
  the tensor allocator had a third version for lifetime-reused ranges. Those
  callers now share the union operation. Diagnostic host descriptor allocation
  also stops at the loader's application limit, like normal package assembly,
  instead of allowing the unavailable tail of physical SRAM.
* **Runtime copy retention follows the selected calls.** A local-copy boolean
  used to retain all four word/strided-copy helpers, with a separate halfword
  flag. Retention now records the actual selected symbols, including inside
  Repeat. Linker dependency handling still retains their referenced support.
  The duplicated classification and over-retention are removed.
* Named the expansion cache bucket/counter fields instead of indexing a four-field
  mutex tuple. Lookup, collision checking, locking and entry limits are unchanged.
* Removed the redundant contiguous-capacity wrapper while retaining its public
  reservation-aware interface.

## Mechanisms retained deliberately

* Compact affine traversal, complete FP8 panels, early multicast ownership,
  shared contracts and the bounded copy-preparation caches address measured
  expansion costs. Coordinate-oracle and cached/uncached comparisons remain.
  The earlier rejected whole-compute caching experiment is already absent.
* Incremental exchange encoding, shared speculative histories, matching and
  schedule replay address the captured large ViT stages. The paired-transfer
  changes are backed by SDK/hardware investigations, not arbitrary restrictions
  to restore. The compact stream-order alternative remains an explicit
  experiment, not an implicit production fallback.
* The row-footprint model remains a ranking estimate, separate from the actual
  encoded table cap. Fragment and shortlist limits still bound compiler work.
  The 80-KiB table cap remains useful: full-depth complete-panel tables need
  approximately 70 KiB. Returning to 64 KiB would reject those before placement.
* Repeat's body cache caches an unchanged body under explicit boundary and
  allocation constraints. It is distinct from the rolled-back regional
  optimizer. Resident-sequence lifetimes, dense sequence allocation and
  compact consumer-compatible parameter homes fix real storage/relocation
  errors. Their failure to make one full-depth plan fit does not make those
  fixes redundant. The ineffective extra allocation ordering was already
  removed in `9deab80`.
* FP8 cast orders, distributed packing and producer fusions represent different
  data-movement choices, not interchangeable wrappers. Their code paths and
  numerical kernels have specific shape/precision contracts and measurements.
  The standalone optimistic-layout search remains diagnostic; it does not
  silently promise nonexistent kernels to production planning.
* Padding-removal restrictions around mixed precisions, attention scratch and
  Repeat aliases are conservative correctness rules. Replacing those with
  precise range/version analysis would be new compiler functionality, not a
  safe deletion during this cleanup.

## Remaining limits

The full-depth failure is still a real live-memory problem for the selected
ownership: weights plus cast input/output exceed the available tensor space on
one tile before other allocations. This review does not claim to fix it or to
provide a proof that all layouts are infeasible. Better activation/parameter
ownership and earlier accounting of final support reservations remain separate
planning work. The expensive search retries and imperfect exchange estimates
also remain; they should be changed through measured planning improvements,
not by adding another unvalidated fallback.

## Validation and size

* 228 codegen library tests passed; five ignored. This includes the added
  finalized-cost selection regression and the updated tensor-memory assertions.
* The subsequent cache field rename passed all 32 low-expansion tests (one
  ignored). Clippy passed after both runs with the existing
  `too_many_arguments` / `type_complexity` allowances.
* A small batch-one, two-layer FP8 ViT with fused QKV and 64 compute tiles
  passed hardware execution and final-output reference comparison (maximum
  absolute error 0.135742, diagnostic tolerances 0.2 absolute / 0.05 relative).
  Detailed profiling readback also passed. Artifacts, log and rendered profile
  are in `artifacts/cleanup-20260910/`. The package includes the functional
  changes through `0b8fa3a`; the final cache field rename was checked separately.
  This is correctness validation, not a full-size performance comparison.
* Relative to `9deab80`, the production portions of the Rust changes remove
  34 lines net, counting comments/blank lines and excluding test changes. The
  reduction is modest; the principal gains are removing mixed memory accounting,
  incorrect provisional-cost ranking and duplicated support classification.
* User changes in `TODO` and existing untracked artifacts were left intact.
