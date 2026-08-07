# Architecture

The package path has four explicit components:

1. `ipu-exchange` produces exchange rows.
2. `ipu-codegen` lowers a `ComputeGraph`, emits supervisor code, and coordinates
   package construction according to `PackageConfig`.
3. `ipu-elf` compiles and links the static runtime and selected kernels.
4. `ipu-package` stores final tile images and host protocol metadata for
   `ipu-driver` and `ipu-runtime`.

`ComputeGraph` is a shaped, structured SSA graph. Values have globally unique
identities, operations refer to explicit inputs, and `Repeat` contains a shared
region with carried values, invariants, and per-iteration value sequences.
Shapes are semantic and support arbitrary rank; GEMM operates on the final two
axes and broadcasts leading batch axes.

## Mid-level IR

`ipu_codegen::mid` is the layout-aware boundary. Every value has a logical
shape plus a `TensorFormat` containing:

- storage precision (`F8F143` with a tensor-wide power-of-two scale, `F16`, or
  `F32`), with accumulation precision recorded separately on operators;
- element order (row-major or AMP left/right/output order);
- axis tiling, where each axis records its block size, distributed partition
  count, and whether an indivisible extent is rejected or zero-padded;
- replication and logical tile-group size;
- a hardware memory class such as IPU21 standard or interleaved memory.

This is deliberately less specific than placement: it records decisions that
change an operator plan, but not physical tile IDs, SRAM addresses,
lifetimes, or exchange rows. AMP order selects the packing family; axis tiling
contains its block dimensions.

`mid::lower` considers complete operator candidates for each semantic operation.
Candidates record every input and output format, per-operand alignment and
access tails, output aliasing permissions, memory-element relations, and
operation-specific compute precision. They can therefore describe
mixed-precision, alternative-layout, and in-place operator implementations. The initial toy
model compares rough arithmetic throughput with bytes moved. When a chosen
operator format differs from its producer, lowering inserts `CastPrecision` and
`Rearrange` operations explicitly. Repeated regions stay structured; their
iterated value sequences are normalized once outside the body rather than
causing the body to be unrolled.

The toy choices describe the supported generic kernels: FP16 GEMM uses AMP
A16/B16x16/C16 and FP32 uses A8/B8x16/C16. PACE operands require 32-byte
alignment, the left stream includes its pipelined access tail, the output uses
the IPU21 interleaved memory class, and the output and left stream occupy
distinct effective memory elements. This is an inspectable scaffold for a
measured cost model or autotuner, not a claim that those choices are globally
optimal.

`low::lower_to_tiles` turns this into a logical per-tile work list. It assigns
rectangular shards to logical tiles, preserves repeats as reusable tile-local
bodies, inserts synchronized exchange phases, and emits kernel runs whose
operand shards are resident on their execution tile. Its initial policy
conservatively gathers every input shard to every output tile; the explicit
schedule permits a later cost model to replace that policy.

The logical schedule deliberately has no SRAM addresses, encoded exchange
rows, or linked kernel symbols. Package construction remains completion-only
until placement resolves those details into `TileProgram`, the finalized
representation consumed by code generation.

## Finalized tile programs

A tile program is an ordered list of:

- an exchange row and its final address; or
- a kernel symbol, output address, input addresses, and scalar arguments.

The code generator validates only local encoding constraints. It does not check
lifetimes, search memory, merge repeated regions, repack executable objects, or
derive kernel memory requirements.

Optional cycle samples name explicit destination addresses. This is a narrow
mechanism rather than a profiling layout policy.

## Runtime

`device/static_runtime.S` initializes workers and transfers control to emitted
supervisor code. `ipu-runtime` initializes the device, replays configuration,
loads an `Application`, applies package configuration writes, and creates a
driver `HostSession`.

Application construction is intentionally not part of the runtime.
