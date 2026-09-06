# Architecture

See [Compiler data flow](COMPILER_DATA_FLOW.md) for the selection, costing and execution boundaries.

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

Operations retain semantics which affect planning: GEMM transpose flags and
attention causality and scaling. Add always uses NumPy broadcasting; GeLU uses
the tanh approximation. Attention intentionally has no general mask input; the
supported form is either causal or unmasked.

## Whole-device mid and tile expansion

The planner chooses formats, distributed GEMM grids, attention strategy and
reduction staging. `MidProgram` contains ordinary distributed tensor values and
whole-device operations. Values specify shape, layout, memory class and ownership
offset. They do not contain a list of tiles, local calls or physical byte spans.

`mid/implementation` decomposes selected algorithms into the same mid IR:

- copies and mapped view windows into ordinary tensor results;
- selected kernel grids with explicit operand windows and allocation reuse;
- sums over an explicit independent-partials dimension;
- structured repeats.

Parallel GEMM uses a leading partials dimension followed by a sum. Output-stationary
GEMM exposes its staged K panels and accumulating output versions. Attention
exposes Q/K/V materialization, products, softmax and merge. Key/value panels
are packed on a small distributed owner grid and then broadcast through ordinary
mid copies. Packing precedes the key-block sequence. Flash places each block's
K/V broadcasts together so generic transfer consolidation can combine them;
materialized attention delays the resident V copy until after softmax so the
large resident K/V matrices have disjoint lifetimes. Key blocks are currently
statically represented as whole-device operations; tile counts do not multiply
this representation. Kernel blocking describes the local calls to enumerate later.

Search recipes retain compact `Arc<MidProgram>` implementations. Final selection
inlines them, resolves claimed deferred views/conversions into consumer-sized
copies, and makes ownership-offset materializations explicit. This resolution
introduces no new search. General copy/view chain composition remains deferred.

`low/expand` realizes selected primitives as a `TileGraph`: local storage values,
relative copies, exchanges, kernel calls and structured repeat bindings. It has
no GEMM or attention strategy builder. `low/call` derives actual operand access
contracts; `low/copy` realizes physical/semantic movement, padding and destination
packing. These physical details do not change the selected distributed algorithm.
`low/passes` merges adjacent contiguous local copies with dependency checks.
`low::lower_to_tiles` projects this graph into per-tile work lists sharing its
arenas through `Arc`. Placement then derives lifetimes and SRAM addresses.

## Geometry and costs

`mid/layout` defines precision, order, partitioning, replication, padding and
memory class. `Layout::resolve` supplies compact partition bounds to both costing
and expansion. `storage` computes physical and semantic byte spans when needed
for actual movement. Parameter ownership rotations happen before tile expansion.

Beam costing in `estimate/mid` traverses compact primitives. It prices maximum
local geometry, approximate endpoint traffic/fragment counts, and coarse live
storage including explicit aliases and temporary requirements. It does not build
a tile graph, enumerate kernel calls, schedule exchange or invoke allocation
analysis. `estimate/primitive` shares kernel prices with the final expanded
timeline evaluator. `estimate/cycles` caches compact operator implementations;
`estimate/mid` composes and prices candidate regions; `estimate/memory` defines
allocation sizes, feasibility and the shared Pareto objectives. Candidate shortlisting
and preliminary beam ranking use these compact execution prices, not boundary
memory as a proxy for cycles. Shortlisting preserves reduction fan-in and result
partition diversity. The implementation cache retains strong references for one
search, so rejected candidates can be reused without rebuilding their regions.

The estimates are deliberately approximate. They sum primitive durations and
conservatively combine local storage maxima, rather than reproduce tile overlap
and physical allocation. Exchange uses a layout-informed coarse fragment-size assumption (smaller for
packed linear redistribution);
contention and actual table sizes are resolved later. Final `estimate/program`
evaluation uses actual tile timelines and can accept measured scheduler phase
prices for finalist reranking. Placement remains the authority on physical fit.

`kernel/abi` defines calls and scalar arguments; `specialization` shares keys
between object construction and call lookup. Backend call materialization resolves
views after placement. `WorkProvenance` retains the source graph operation through
mid primitives and tile expansion for diagnostics and profiling.

Repeat expansion preserves an aliasable carried chain. Iterated inputs have
equal aligned local strides including access tails. Execution advances base
pointers without unrolling the structured body.

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

## Package selection and diagnostics

`package/selection` expands a bounded shortlist in parallel, models tile mappings,
and exactly schedules the best candidates. It retains the projected baseline and
its provisional placement instead of rebuilding them. Infeasible alternatives
are rejected individually; an entirely infeasible shortlist returns its error.
Both finalist ranking and final-package reporting use `scheduled_program_cycles`
to compose actual exchange horizons with the optimized compute/copy timeline.

`package/placement` screens physical tile mappings and SRAM offsets;
`package/profile` owns instrumentation and profile metadata. `package/tile_program`
packages explicit address-resolved programs for hardware diagnostics. The parent
module coordinates linking, memory reservation and final image construction.

`exchange/order` proposes alternative dependency-respecting orders;
`exchange/diagnostic` reports row hazards, endpoint pressure and critical chains.
The physical scheduler and its timing/validation rules remain in `exchange`.
