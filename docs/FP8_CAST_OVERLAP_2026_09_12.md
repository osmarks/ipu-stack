# Partial-overlap FP16 to FP8 casts

Partial overlap is compatible with the ISA's fast cast instructions in principle.
It is not supported by the current cast/placement contract.

## Hardware and current implementation

The local IPU21 ISA manual, `TileVertexISA-IPU21-1.3.1.pdf`, sections 2.9.10.2,
2.9.12 and 3.7.5.1.1, requires the simultaneous accesses of `ldst64pace` to
address different memory elements. It does not require disjoint whole buffers.
The instruction has independent load/store pointers and increments.

`device/cast_f8.cpp` checks whole input/output ranges, conservatively treating
each 32 KiB address group as inseparable. Its pipelined linear and packed-row
loops run only when that check succeeds. Overlapping ranges currently fail it.
`low/call.rs` does not impose a hard distinct-elements requirement on casts:
separate ordinary allocations may share an element and use the slower kernel.
This is different from GEMM's mandatory output/left-input element separation.

Dropping the cast's range check would be incorrect. Besides possible instruction
clashes, six workers process independent row/vector streams. Output writes need
to avoid every worker's unread input, including padding initialization and panel
permutations. No arbitrary cross-worker progress assumption should establish
this safety property.

## A fast path without rewriting the assembly loop

Convert complete chunks in order, joining the workers locally between chunks.
Position the output 32 KiB before a 32 KiB-aligned input. Each chunk reads and
writes separate element groups; later chunks overwrite only previously consumed
input. These are local tile dependencies, not additional device-wide exchanges
or synchronization phases.

A checked address example uses 164 rows and 384 columns in AMP-left order,
matching the captured staging byte count. Each FP8 panel consumes two contiguous
FP16 16-column panels: 10,496 input bytes become 5,248 output bytes. Process three
such pairs per chunk, four chunks in total. This preserves the existing packed
cast's two-stream ordering and its six-worker distribution within each chunk.

With input base `S` and output base `S - 32768`:

| Chunk | Input relative to S | Output relative to S |
|---|---|---|
| 0 | [0, 31488) | [-32768, -17024) |
| 1 | [31488, 62976) | [-17024, -1280) |
| 2 | [62976, 94464) | [-1280, 14464) |
| 3 | [94464, 125952) | [14464, 30208) |

All four pass the current whole-range 32 KiB separation test. Every output
ends before its chunk's input begins. The source allocation plus the preceding
32 KiB occupies 158,720 bytes, versus 188,936 bytes for separate allocations
including the FP8 consumer's eight-byte access tail: **30,216 bytes saved**.
The final output tail lies within already consumed input storage.

This is an address/ordering feasibility example, not a measured hardware run
or proof that a particular full-model placement admits the required span.
Contiguous row-major input followed by AMP packing requires different chunk
geometry; one cannot apply this panel slicing to it without checking its reads.

## Required compiler changes

1. Allow storage donation only at the input's last use, with compatible memory
   addressing classes. Surviving aliases and additional consumers forbid it.
2. Represent the relative source/output offsets and their overlapping backing
   storage explicitly. Current alias groups share a base and reserve their
   maximum extent across the combined lifetime; that is insufficient for the
   shifted view and precise release of consumed input.
3. Lower compatible casts into sequential complete-panel chunks, with descriptors
   adjusted for chunk extents and padding. Keep the existing disjoint variant.
4. Account for launch/join/setup costs and alignment. Requiring a 32 KiB-aligned
   union can itself worsen fragmentation. A more flexible offset/chunk selection
   could relax that requirement, but is not needed to demonstrate feasibility.
5. Preserve downstream GEMM element-separation requirements on the live FP8
   output, rather than unnecessarily excluding every element ever occupied by
   the FP16 input. Retaining the whole union to the output's last use would be
   correct but could lose memory opportunities elsewhere.

The device arithmetic change is small or unnecessary. The main work is a precise
storage-overlap contract and chunk lowering, followed by hardware correctness and
cycle tests. This is broader than relaxing a placement alignment or changing one
cast branch, but does not require an arbitrary in-place permutation engine.
