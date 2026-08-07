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

- storage precision (`F8F143`, `F16`, or `F32`), with accumulation precision
  recorded separately on kernels;
- element order (row-major or parameterized AMP left/right/output order);
- coarse sharding (replicated, rows, columns, or heads) and logical tile-group
  size;
- contiguous or memory-bank-interleaved storage and alignment.

This is deliberately less specific than placement: it records decisions that
change a kernel or exchange plan, but not physical tile IDs, SRAM addresses,
lifetimes, or exchange rows. AMP order parameterizes the useful parts of the
old `A8/A16/A32`, `B8x16/B16x16/B32x16`, and `C16` layout vocabulary.

`mid::lower` considers legal formats for each semantic operation. The initial
toy model compares rough arithmetic throughput with bytes moved. When a chosen
kernel format differs from its producer, lowering inserts `CastPrecision` and
`Rearrange` operations explicitly. Repeated regions stay structured; their
iterated value sequences are normalized once outside the body rather than
causing the body to be unrolled.

The toy choices currently describe the retained generic kernels: FP16 GEMM
uses AMP A16/B16x16/C16, FP32 uses A8/B8x16/C16, and right operands use
interleaved storage. This is an inspectable scaffold for a measured cost model
or autotuner, not a claim that those choices are globally optimal.

The subsequent mid-to-low scheduling and placement pass does not exist yet,
so package construction still emits a completion-only tile program.
`TileProgram` remains the finalized representation below that future pass.

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
