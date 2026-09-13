# One schedule with timed outgoing-base changes

The scheduler now groups moving Repeat sources and fixed sources in one schedule.
It no longer builds combined, fixed, and moving alternatives to decide whether to
split a phase. Each phase has one cached recipe; the existing ordinary/paired
selection and validation after placement remain.

Moving sources are preferred among ready work, using the Repeat pointer already
loaded into m6. Memory dependencies can require additional group changes. At a
change, the sender writes OUTGOING_BASE from m6 or the zero register m15. The
receive configuration remains valid throughout the phase. There is no additional
synchronization, worker launch, or common section boundary.

## Switch overhead

The old join cost 46 scheduled cycles, including its prefix, alignment allowance,
and changes to both incoming and outgoing configuration. The new operation is a
single 13-cycle OUTGOING_BASE write. Each tile finds an available instruction
interval after its last send; receive payload can continue during the write.
Independent tiles need not switch at the same time. Alignment and other control
instructions can delay the chosen interval, so 13 cycles is the instruction cost,
not a promise about the change in the phase's critical path.

The row builder reserves that control interval against subsequent receive setup.
Later send insertion cannot cross the committed base transition. Insertion also
checks two-word SENDPICP alignment immediately: retrying a later transfer cannot
repair an already committed, misaligned base change. The decoder verifies base
writes and their timings alongside ordinary receive controls and sends.

A tile's moving sources share a base when all relative addresses remain
representable. Incompatible moving sequences retain ordinary word patches.
Fixed-source sends use zero base and never acquire inverse relocation patches.
If no moving base is representable, row writes use m15 instead of an uninitialized
m6. Relocation tests cover crossing sequences and all Repeat bindings.

## Matched 27-layer SigLIP results

Same saved BS1/BS2 mid plans, FP8 PV, B1024 exchange streams, and two resident
inference calls per package. Cycles use the runtime viewer's initial-sync crop,
not the entire profiling capture. Planning times are single package builds with
16 Rayon threads and include placement and final validation.

| Measurement | Previous split scheduler | Unified scheduler |
|---|---:|---:|
| BS1 runtime | 10,528,050 cycles / 7.018700 ms | 10,446,450 / 6.964300 ms |
| BS2 runtime | 17,198,316 cycles / 11.465544 ms | 17,176,698 / 11.451132 ms |
| BS1 package planning | 86.5 s | 70.640 s |
| BS2 package planning | 366.8 s | 226.758 s |
| BS1 reserved exchange bytes per tile | 43,732 | 43,408 |
| BS2 reserved exchange bytes per tile | 87,000 | 86,672 |
| BS1 Repeat send patch words per layer, all tiles | 365 | 0 |
| BS2 Repeat send patch words per layer, all tiles | 1,049 | 0 |
| BS1 row-sharing patch words per layer, all tiles | 14 | 0 |
| BS2 row-sharing patch words per layer, all tiles | 2,916 | 2,916 |

Runtime improves 0.78% at BS1 and 0.13% at BS2 relative to the split scheduler.
Package planning takes 18% and 38% less time, respectively. Remaining BS2
row-sharing patches are still at most one word per tile/phase. One-time head
patching is unchanged: 11,467 words at BS1 and 7,761 at BS2, across all tiles.

Both resident calls pass at both batch sizes. Minimum cosine similarity against
the FP32 reference is 0.994352454 for BS1 and 0.994151272 for BS2. Extracted
embeddings are byte-identical to the previous split-scheduler packages (2,304 and
4,608 bytes respectively).

An initial unified implementation waited for the tile's entire receive horizon
before switching; BS1 took 10,614,018 cycles. Moving the write into an available
control interval avoids that regression. The resulting benefit is from preserving
overlap as well as removing unnecessary register writes.

## Artifacts and checks

- `artifacts/exchange-base-aware-20260913/final/`: BS1 package, incremental
  `model.html`, `model.capnp`, exact memory dump, resident tensors, patch audit,
  query JSON, build script, and log.
- `artifacts/exchange-base-aware-20260913/bs2/`: equivalent BS2 artifacts.
- `artifacts/exchange-base-aware-20260913/compare.rs`: compares embedding bytes
  using each package's output bindings; ignores profiling-buffer differences.
- Previous comparison: `docs/EXCHANGE_BASE_SECTIONS_2026_09_13.md`.

Validation: 53 exchange-crate tests and 50 exchange-related codegen tests pass
(four manual benchmarks ignored in total); workspace check passes. Coverage
includes randomized dependency-aware schedules, paired transfers, relocation,
receive-control collisions, and incremental/full row-encoding equivalence.
