# Architecture

Start with [Compiler data flow](COMPILER_DATA_FLOW.md). It traces the current
production entry point, the states of each representation, GEMM/reduction,
broadcast Add, mapped copies, physical packaging, and cache lifetimes.

[Compiler structure proposal](COMPILER_STRUCTURE_PROPOSAL.md) records the
2026-09-14 design review and proposed responsibility changes. It is a proposal,
not a description of implemented behavior. In particular, current mid still
mixes unresolved selections with executable operations, and low still has two
movement entry paths.

The proposal now specifies the [compiler driver and its feedback](COMPILER_STRUCTURE_PROPOSAL.md#concrete-compiler-control-flow)
and [concrete source owners](COMPILER_STRUCTURE_PROPOSAL.md#connected-construction-with-concrete-source-owners),
including direct high-to-mid construction and sum as a compute family.

## Workspace

| Component | Current responsibility |
| --- | --- |
| `ipu-codegen` | Semantic graph, distributed layout/implementation selection, mid rewrites, tile expansion, costs, placement, exchange integration, kernel selection, supervisor emission and package construction |
| `ipu-exchange` | Timed exchange programs and encodings, host-exchange packets, topology/multicast construction, plus shared supervisor instruction encoders |
| `ipu-elf` | SDK kernel compilation and artifact caching; ELF inspection, linking and relocations |
| `ipu-package` | Serialized application and profile formats, validation and host bindings; currently also owns several IPU21 memory/loader constants |
| `ipu-driver` | Device access, reset/loading, diagnostic registers and host-exchange sessions |
| `ipu-runtime` | Small convenience API joining initialization, loading and host sessions |
| `ipu-profile` | Cycle-query, phase and useful-work analysis |
| `ipu-cli` | Build/inspection/profile commands and HTML profile generation |
| `ipu-tests` | Workload/reference runners and standalone hardware fixtures |
| `device/` | Runtime, assembly and C++ kernels compiled by the SDK toolchain |

The workspace table describes ownership as it exists. The proposal separates
architectural facts from runtime conventions and compiler policy; it does not
assume that today's crate boundaries are the intended final ones.

## Representation invariants worth preserving

`ComputeGraph` is shaped structured SSA. Add uses NumPy-style broadcasting;
GEMM contracts its final matrix axes and broadcasts leading axes. Implementation
support is narrower than semantic shape support in some cases. Repeat retains a
shared body with carried values, invariants and parameter sequences.

Mid describes whole-device selections with shape, precision, layout and owner
mapping. It does not contain a list of tile calls. Parallel GEMM partials have an
explicit tensor dimension and distributed Sum. Output-stationary GEMM exposes
bounded panels and accumulating versions. Attention decomposes into ordinary mid
movement, products, softmax and merge work.

Low contains concrete shards, relative views, calls, local copies, multicast
recipient groups and structured execution. Placement assigns physical addresses
and must preserve resident parameters across host inference calls. Kernel access
constraints, instruction-fetch restrictions and runtime reservations constrain
that placement. Final package construction can require exchange replay when
addresses change.

The runtime loads the resulting application; it does not select layouts or build
kernels. Profile provenance connects physical work back to semantic operations.

## Related documentation

- [Cycle profiling](PROFILING.md)
- [Memory profiling](MEMORY_PROFILING.md)
- [Historical low-expansion cache experiment](LOW_FRAGMENT_CACHE_2026_09_09.md)
- [Review of cleanup generality](CLEANUP_GENERALITY_REVIEW_2026_09_14.md)

Dated experiment and audit reports are historical evidence. The current flow
map and source take precedence when those reports describe removed planning
paths or old implementation restrictions.
