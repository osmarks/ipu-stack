# Partial-overlap FP16 to FP8 casts

Implemented using a shifted shared buffer. The earlier same-base experiment,
which protected a prefix in temporary storage, is replaced.

## Address and execution contract

The local IPU21 ISA manual, `TileVertexISA-IPU21-1.3.1.pdf`, sections 2.9.10.2,
2.9.12 and 3.7.5.1.1, requires simultaneous `ldst64pace` accesses to use different
memory elements. Whole tensors can overlap if each invocation reads and writes
separate elements and never overwrites unread input.

The output starts at a 32 KiB-aligned address; the input starts 32 KiB later.
Low expansion divides the cast into complete FP16 panel pairs (AMP-left) or
complete rows/vectors (row-major). Each chunk is the largest prefix whose output
ends before the input's first 32 KiB group. Workers join locally between chunks.
This passes the existing kernel's conservative whole-range bank check, preserves
its simultaneous load/store path, and needs no prefix copy or device-wide barrier.

For 164 rows by 384 columns in AMP-left order, offsets from the allocation base:

| Call | FP16 reads | FP8 writes |
|---|---|---|
| 0 | [32,768, 95,744) | [0, 31,488) |
| 1 | [95,744, 158,720) | [31,488, 62,976) |

The second call overwrites only input consumed by the first. Independent worker
progress within either call cannot violate this property. The 62,976-byte FP8
allocation is replaced by a 32,768-byte prefix: **29.5 KiB saved** before access
tails/alignment. There are two worker launches instead of one.

## Compiler integration

- Mid selects storage donation only for a fresh, last-use FP16 result with the
  same shape, order, ownership and memory class as its FP8 output. Resident
  parameters, region arguments and existing input aliases cannot donate.
- Compatible source copies are forced to materialize so donation cannot
  accidentally overwrite their original source.
- Casts internal to Repeat can donate. Carried outputs and their possible
  zero-copy aliases cannot: Repeat would bind the shifted allocation to the
  previous iteration's input before that input was necessarily consumed.
- Low represents the displacement explicitly with `ShiftedAlias`. Placement
  resolves alias and Repeat address equalities together, rejects inconsistent
  offsets, aligns the backing allocation, and reserves its full union.
- The allocator retains the whole union for the combined lifetime and applies
  bank conflicts to it conservatively. Consumed-tail release and member-specific
  bank conflicts are not implemented. The memory profile displays this actual
  reservation, including the prefix.
- Mid memory accounting includes one prefix per physical shard and retains the
  aliased input's full size/lifetime. Compute estimates sum the chunk calls.
- Capacity baselines enable eligible donation. Local search compares both
  policies for layout proposals, since a slightly slower cast may enable a
  faster complete plan. Saved recipes retain the choice; older states load
  with the previous schema's defaults.

This supports row-major-to-row-major and AMP-left-to-AMP-left conversion.
It does not implement overlapping row-major-to-AMP packing. The previously
rejected late-cast BS2 plan uses that combined conversion and is not fixed by
this implementation. Shapes whose FP8 allocation is at most 32 KiB are excluded
because the prefix would not save space.

## Hardware measurements

Command (with the SDK environment loaded):

```sh
target/release/cast_check --sdk "$POPLAR_SDK_ENABLED" --in-place-only   --output artifacts/shifted-cast-20260912/hardware
```

36 cases cover three matrix shapes, both orders, three aligned base addresses,
and disjoint/shifted execution. **3,797,568 checked bytes passed bitwise**,
including guards and input bytes outside the overwritten output. Arithmetic is
unchanged; no modified device kernel is required.

| Shape | Order | Disjoint cycles | Shifted cycles | Overhead |
|---|---|---:|---:|---:|
| 164 × 384 | Row-major | 16,458 | 17,112 | 4.0% |
| 164 × 384 | AMP-left | 24,174 | 25,464 | 5.3% |
| 97 × 512 | Row-major | 13,134 | 13,788 | 5.0% |
| 97 × 512 | AMP-left | 23,334 | 24,624 | 5.5% |
| 128 × 512 | Row-major | 17,106 | 17,760 | 3.8% |
| 128 × 512 | AMP-left | 27,174 | 28,464 | 4.7% |

All three base addresses gave identical cycles. The replaced same-base
164 × 384 AMP-left version took 34,836 cycles, versus 25,464 now.

Compiler tests check physical offsets, bank separation of every generated call,
complete output coverage, parameter preservation, multiple shards per tile,
legacy conversion selection, Repeat exclusions, and inconsistent alias cycles.
