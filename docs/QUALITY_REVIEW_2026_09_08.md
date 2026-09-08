# Review after the main cleanup

Baseline: `de9fb77` (the previous cleanup report), reviewed through `869689b`.
The diff spans 124 files, with about 15,300 insertions and 3,700 deletions,
including tests, documentation and new FP8/model functionality. This is not all
accidental growth. The review concentrated on exchange state/encoding, package
acceptance, candidate generation, memory estimates and copy/ownership lowering.
It is a targeted review, not a claim that every intervening change was verified.

## Corrections made

* Ordinary scheduling used `?` before trying paired scheduling. An ordinary
  encoding or effort failure therefore ruled out a potentially valid paired
  alternative. Both alternatives are now evaluated; either successful result
  can survive. Successful comparisons still prefer strictly shorter schedules.
* Sender insertion scanned every previous send for overlap. Histories now stay
  ordered, making the two neighboring intervals sufficient for this check.
  Out-of-order insertion and failed insertions have a regression test. This
  does not remove the encoder's remaining full-history costs.
* Receive insertion invalidated its cache and removed its stream state before
  checking stream overlap. Rejection now precedes mutation. The public transfer
  builder was already transactional through endpoint clones; this repairs the
  underlying state method as well.
* Memory rejection diagnostics subtracted estimated rows from contiguous
  capacity even though the hard acceptance screen no longer does. Reporting
  now uses the same reservation convention as the check.
* Factor-copy unpacking marked its work as `OperatorKernel`. It now records
  `LayoutRearrangement`, without changing the generated computation.

An experiment restricting receive teardown removal to two indexed timestamps
was discarded. The existing receive-validation benchmark at 4,096 transfers
took 364 ms before and 360 ms after, with identical output checksums; the
1,024-transfer case regressed from 20 to 30 ms. These individual CPU timings
do not establish a speedup and did not justify extra update machinery.

## Remaining assumptions and design issues

**Memory estimates have mixed meanings.** `MemoryPeaks` incorporates an
exchange-row estimate into standard and total peaks, then consumers subtract
it again when they need tensor capacity. Tensor estimates also sum individual
maximum shard sizes, whose maxima need not belong to the same tile. These are
useful conservative estimates but not proofs of infeasibility. Separate tensor
lifetimes, support/executable storage, and estimated rows in the representation;
make ranking and feasibility consume the appropriate quantities explicitly.
Do not simply remove capacity screening: exact placement still matters.

**The expensive acceptance boundary comes too late.** Package assembly learns
exact row size only after scheduling, then rejects code/table or tensor storage.
The batch-four sweep spent 26.8 minutes exhausting eight finalists. Uniform
support reservations and code/table allocation order are additional conservative
choices worth measuring before declaring a layout impossible. Current failures
do not establish that batch four or eight cannot fit on hardware.

**Candidate definitions hide shape-dependent generation.** A
`ConcreteOperatorCandidate` can carry `RowMajorGrid` or preserved-input policies
which rewrite formats in candidate generation. These are templates, not fully
concrete plans. Explicit template generation returning complete plans would
clarify the boundary. Row policies currently partition tokens rather than the
combined batch/token rows; this is a real restriction, not an ISA requirement.
The general GEMM path also explicitly rejects nontrivial right-hand batches;
attention has its own product paths. That restriction is tested and deliberate,
but should not be mistaken for general batched-GEMM support.

**Ownership transformations repeat dependency reasoning.** Copy grouping and
reduction movement both inspect reads/writes through storage groups. A shared
storage-access summary could remove repeated logic and make alias hazards easier
to audit. Preserve distinctions between grouping independent operations and
moving operations across intervening writes; a common predicate alone is not
enough. These passes are conservative and specialized to reduction/copy shapes.

**Schedule quality is essentially horizon-only.** `schedule_score` ignores row
storage, and ordinary/paired selection uses only horizon. A shorter schedule
that adds controls or destroys table sharing can prevent the entire package
from fitting. This is more actionable than a raw transfer-count cutoff.

**Caches are scoped too narrowly, but widening them is not automatically cheap.**
Each finalist starts with a fresh relocation cache. Replay itself may clone and
encode long histories before discovering a mismatch. Reuse across finalists
needs a cheap structural problem identity first, followed by the existing
address/hazard validation. A phase ordinal alone is not such an identity.

## Larger exchange changes worth implementing

1. **Transactional event timelines and incremental validation.** Keep a single
   ordered per-tile send/control timeline. Stage only the modified events and
   stream metadata; encode from the preceding valid checkpoint and commit those
   edits on success. The checkpoint must include lookahead dependence: a newly
   inserted sender or removal of a receive teardown can invalidate earlier
   SENDPICP decisions. Keep emitted words/checkpoints in shared chunks or an
   append/rollback arena, rather than copying the whole prefix. `Arc<Vec<_>>`
   alone would still copy the vector on every speculative mutation. This is the
   highest-priority compiler improvement: current encoded prefixes duplicate
   sender/event histories, compare them from the beginning, and copy retained
   words even when almost all instructions are reusable.

2. **An immutable exchange problem plus mutable schedule state.** Build transfer
   alternatives, dependencies and resource incidence once per placement. Feed
   that same problem to greedy ordering, replay and neighborhood repair. Keep
   the dependency/timing rules and row encoder shared. This separates invariant
   work from trial choices without introducing another tensor planning layer.
   Then cache successful recipes against this problem's structure, not just a
   phase number. Avoid blindly caching expensive failures whose cause may change
   with addresses or the effort budget.

3. **Bounded fast/compact schedule alternatives.** Retain a small number of
   choices trading cycles for encoded row storage, then evaluate actual package
   table sharing. Start with two whole-program policies rather than a Cartesian
   product of per-phase alternatives. Paired mode and receive-stream continuity
   belong in this comparison. This targets both runtime and batch feasibility;
   a heap replacement does not address either objective mismatch.

4. **Local encoding repair instead of whole-phase strict fallback.** The current
   deferred attempt can discover a SENDPICP alignment failure only at encoding,
   then rebuild the phase with incremental validation. An event timeline can
   locate the conflicting controls and reschedule their dependent suffix. This
   is a second step after transactional validation, not a reason to weaken
   checks or assume every parity conflict can be fixed by a local one-cycle gap.

The first change can preserve existing output and timing exactly. Differential
tests should compare full encoding with transactional encoding after arbitrary
insertions, teardown replacement, rejected trials, paired transitions and
loopback transfers. Measure both long endpoint histories and complete ViT
compilation. Device timing changes from compact alternatives or local repair
need hardware validation; retain the existing scheduler as a quality baseline.
None of these proposals relies on relaxing hardware timing constraints or on
the existence of an undiscovered strided exchange mode.

Validation: 164 codegen tests and its doctest passed; the final exchange suite
passed 48 tests, including full-versus-incremental encoding comparisons and the
new rejected-receive/out-of-order-send regressions. Clippy passed for both crates
and all targets. No new hardware timing or whole-ViT compiler speedup is claimed
for these review corrections. The larger timeline/cache/frontier changes above
are proposals, not implemented changes.

## Follow-up production-code cleanup

The next pass removes 90 non-test production lines relative to `b649899`
(150 added, 240 removed, excluding `kernel/tests.rs`). C++ vertex compilation
and supervisor wrappers now have one assembly path; unpack recipes share shape
flags and symbol registration. Fixed-kernel inventory records the ABI-selected
symbols rather than maintaining a second classification through five booleans.
Ownership passes share alias-group rotation, and copy/use analysis shares
`MidOperation::read_values`, including Repeat parameter sequences.

Validation: 164 codegen tests, its doctest and Clippy pass. A small batch-two FP8
ViT with 64 active compute tiles was compiled before and after. All 1,472 loaded
tile images, input/weight/output bindings and profile plans are identical.
Artifacts and the comparison are in `artifacts/cleanup-20260908/`. Intermediate
codelet object names are now consistently derived from their public call symbol;
the linked kernel ABI and program bytes did not change in this comparison.
Exchange scheduling was not changed in this cleanup pass.

## Scheduler follow-up

The shared timeline, incremental encoding, shared problem facts and recipe-cache
work is now implemented and benchmarked on captured ViT phases up to 1,172,736
transfers. Matching readiness and bounded repair were also changed. See
[the capture and experiment report](EXCHANGE_SCHEDULER_BENCHMARKS_2026_09_08.md)
for measurements, the recovered MLP search regression and hardware validation.
Fast/compact package alternatives and local repair of deferred encoding failures
remain proposals.
