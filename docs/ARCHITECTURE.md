# Architecture

See [Compiler data flow and deletion targets](COMPILER_DATA_FLOW.md) for current
and proposed data-flow diagrams, duplicated cost derivations, and the next
substantial opportunities to reduce implementation size.

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

Operations retain semantics which affect planning: GEMM transpose flags, NumPy
Add broadcasting, and attention causality and scaling. GeLU currently denotes
the exact function. Attention intentionally has no general mask input; the
supported form is either causal or unmasked.

## Compiler construction and executable mid IR

Implementation modules are private; the crate root exposes graph construction,
package building, and the types used by diagnostics.

The planner screens whole-operation choices: formats, GEMM distribution,
attention strategy and reduction staging. Cheap geometry and boundary-storage
heuristics precede detailed evaluation. Detailed branches use executable mid
fragments, shared allocation analysis, and primitive cycle prices. Selected
`ImplementationCandidate` recipes retain their fragments; binding reuses them
when boundary ownership, formats and capacity permit. Deferred movement is
constructed by the same implementation builder in its actual consumer context.
Final construction applies parameter ownership and reprices the executable
`MidProgram` before physical scheduling. Low never expands opaque operators.

`MidProgram` owns ordinary `BlockValue`s for input/output shards, GEMM partials,
packed panels, reduction accumulators, and other intermediate results. Each has
a tensor format, concrete extents, ownership tile, and storage/alias definition.
Logical-value metadata and checkpoint boundaries are retained for diagnostics.
The whole-device `BlockRegion` orders:

- compute blocks with explicit input/output views, selected kernel kinds, and
  bound storage requirements;
- local copies with relative offsets and contiguous/strided patterns;
- exchanges with source/destination views and semantic or physical traversal;
- structured repeats with one shared body and per-tile carried/iterated bindings;
- optional diagnostic checkpoints.

Entries preserve per-tile order; entries on different tiles may overlap until
an exchange synchronization. They have explicit storage aliases and accumulating
writes; the executable region is not a second SSA graph. It contains no SRAM
addresses, linked kernel symbols, or encoded exchange rows.

GEMM builders emit individual compute blocks, input movement, partial values,
and reduction steps. The reusable sum builder accepts groups of independent
block views, deriving complete/streamed receive stages from each group's actual
contributor count. A single contribution becomes a copy. Its current packed
FP16 kernel path requires matching contribution coordinates and storage order;
other layouts require explicit rearrangement before summation. Partial values
are not represented as interchangeable tensor replicas.

Both blocked and materialized attention use the same executable operations.
Query/key/value preparation, QK and probability/value GEMMs, softmax, merge, and
result movement are visible in mid. Their candidate-building helpers are split
into shared task geometry, panel preparation, and the two attention strategies.
No attention implementation remains in low.

`mid/copy` owns relative copy formation and direct-word versus staging policy.
`CopyOrder` specifies coordinate-preserving or allocation-order traversal for
both local and inter-tile movement. `mid/passes` merges adjacent contiguous copies
on each tile, respecting compute, exchange, repeat, and checkpoint boundaries.
It excludes aliasing source/destination allocations and compacts the copy arena.
This is not yet arbitrary composition of chained layout conversions.

## Geometry, costing, and projection

`mid/layout` defines precision, element order, axis tiling, replication, padding,
and memory class. `Layout::resolve` constructs partition bounds shared by
estimation and block construction; logical ranges exclude padding and physical
ranges include padding owned by each shard. Ownership rotations balance parameter
storage before block construction. `storage` computes byte spans from borrowed
format/extents without depending on low. Block adapters add identity checks.

`mid/operator` and `catalogue` define legal choices; `candidates` specializes
and cheaply screens them. `estimate/implementation` builds and caches concrete
fragments. `estimate/program` prices actual kernel calls, copy patterns and
logical exchange spans, composing per-tile timelines across barriers and repeats.
`estimate/memory` uses the same alias, access-tail and lifetime analysis as
placement. `estimate/tensor`, `traffic` and `cycles` retain shared geometry,
coarse conversion screening and target prices. Deferred view/materialization
claims emit explicit consumer-sized buffers and movement; low does not rediscover
them. Shared materialization batches handle destination staging, padding,
local/remote population and final transforms for GEMM, attention and conversions.

`low::lower_to_tiles` only projects the executable mid region into per-tile work
lists. It shares the immutable `MidProgram` through `Arc`, including its block,
copy, exchange, and kernel arenas. It projects repeat bodies and optionally emits
checkpoints. It cannot expand a whole operator or choose a new materialization.
Placement derives lifetimes and SRAM addresses from those explicit operations.

Repeat construction preserves an aliasable carried chain; a fresh yield can
reuse the carried storage after its last read. Iterated input blocks have equal,
aligned per-tile strides including required access tails. Repeated execution
advances base pointers rather than unrolling the body or building pointer tables.

`kernel/abi` defines supported calls and scalar arguments; `specialization`
provides keys shared by object construction and call lookup; `mid/call` supplies
address-independent call shapes and primitive access contracts. GEMM, rearrangement, and attention recipes are separate modules.
`device/worker_call.S` marshals declared registers into C++ vertex fields. Backend
call materialization resolves block views after placement.

`PipelineConfig` supplies candidate construction and packaging with target,
formats, catalogue, and scheduling/profiling policy. `PackageConfig` adds the
build environment. `WorkProvenance` follows graph operations through individual
blocks into placement diagnostics and profiles.

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

### View and kernel specialization contracts

The semantic graph and candidate builders share axis split/merge views.
`ComputeGraph::view` accepts `AxisFactorView`; `split_heads` is only a rank-three
convenience constructor, with no separate graph or mid operator kind. The mapping moves a factor between arbitrary
axes, validates the output shape, and maps rectangular slices back to their
source. Materialized and deferred lowering share this geometry. This is a
split/merge view primitive, not yet a general reshape/permutation composition.
Attention-specific candidate layouts remain, alongside a row-major fallback for
other axis pairs. Their emitted movement is priced directly. The host reference evaluator uses an
independent forward mapping to check the compiler's inverse slice mapping.

Kernel build planning and call emission use the same `KernelSpecialization`
key. ABI scalar arguments are static typed slices rather than strings interpreted at
runtime. The ABI records pointer arity; fixed register constants are shared with
call emission. Mid compute blocks carry TileKernelSpec directly. The build plan retains one specialization-to-symbol map; redundant
GEMM row inventories and unused provisional ABI symbol names are removed.

Packaging still uses provisional scheduling to size exchange tables and generated
code. It may reuse transfer widths and ordering after final placement, but always
rebuilds and validates physical rows. Changed normalized rows trigger optimization
again. Removing this allocation/scheduling cycle is not a current refactoring goal.

Attention-stage assembly workers specialize head/value dimensions and padded
widths; block row counts are ordinary typed ABI arguments. This supports multiple
configurations and more than two block sizes in one package without mapping
intermediate sizes to a "large" specialization. Full-block C++ softmax retains
its query-row specialization; assembly tail softmax shares workers across both
query and logical key sizes.

Candidate recipes use `StorageRequirements` for format and materialization choices.
Mid calls construct `KernelRequirements` from the actual buffers and primitive
kernel kind. These contain only operand formats, alignment/access tails and SRAM
separation constraints; they do not retain candidate aliasing, staging or
materialization policy. GEMM calls, including those inside attention, require
their actual left operand's read tail and output/left SRAM separation. Reductions
and attention stages no longer inherit unrelated enclosing-operator constraints.
