# Compiler data flow

Updated 2026-09-05 for the whole-device mid boundary. These curated diagrams
supersede the older generated `ipu-stack-package-callgraph*` artifacts.

## Selection and execution

```mermaid
flowchart TD
  G[ComputeGraph: semantic operations and regions] --> C[Catalogue and geometry screening]
  C --> B[mid implementation: distributed tensor primitives]
  B --> CACHE[Compact implementation cache]
  CACHE --> REGION[Compose compact candidate region]
  REGION --> COST[Geometry prices and coarse live storage]
  COST --> BEAM[Beam ranking]
  BEAM --> FINAL[Selected MidProgram: resolve recipes and deferred movement]
  FINAL --> EXPAND[low expand: enumerate tile calls and transfers]
  EXPAND --> TG[TileGraph: storage, views, calls, copies and exchanges]
  TG --> LOW[LowProgram: per-tile work lists]
  LOW --> PLACE[Physical allocation]
  PLACE --> EX[Physical exchange scheduling]
  EX --> REFINE[Optional finalist reranking from actual timelines]
  REFINE --> IMAGE[Linked tile images and package]
```

Mid selection chooses distributed work. Low expansion realizes that work; it does
not rebuild a GEMM or attention algorithm. Cached fragments contain distributed
tensor values, not tile buffers. Final resolution inserts ordinary mid copies
where selected ownership differs, and maps claimed views directly into consumer
windows. Low projection only builds per-tile references to the expanded arenas.

## Shared prices, different precision

```mermaid
flowchart LR
  MID[Mid primitives and tensor layouts] --> GEO[Maximum local geometry]
  GEO --> PRICE[Shared primitive kernel prices]
  GEO --> COARSE[Approximate traffic and storage liveness]
  PRICE --> SCORE[Beam score]
  COARSE --> SCORE
  TILE[Expanded calls and movement] --> PRICE
  TILE --> TIME[Actual per-tile timelines]
  PRICE --> TIME
  TILE --> ALLOC[Access requirements and allocation lifetimes]
  TILE --> SCHED[Exchange scheduler]
  SCHED --> TIME
```

Beam costing never constructs tile graphs or runs physical allocation analysis.
Its exchange approximation assumes a representative fragment size rather than
walking byte spans or predicting the ready queue. Shared kernel prices avoid
maintaining separate GEMM/attention cost algorithms. Actual timelines and
scheduled phase prices remain available after expansion. Coarse memory feasibility
does not guarantee placement, particularly with disjoint ownership groups and
fragmented exchange tables.

Sources: [mid decomposition](../crates/ipu-codegen/src/mid/implementation/mod.rs),
[mid primitives](../crates/ipu-codegen/src/mid/primitive.rs),
[compact costing](../crates/ipu-codegen/src/estimate/mid.rs),
[shared kernel prices](../crates/ipu-codegen/src/estimate/primitive.rs),
[tile expansion](../crates/ipu-codegen/src/low/expand/primitive.rs),
[final timelines](../crates/ipu-codegen/src/estimate/program.rs).

## Explicit decomposition

```mermaid
flowchart LR
  L[Left materialization] --> GEMM[Distributed partial GEMM]
  R[Right materialization] --> GEMM
  GEMM --> P[Tensor with independent-partials axis]
  P --> SUM[Sum: complete or streamed contributors]
  SUM --> O[Output tensor]
```

Output-stationary GEMM instead exposes staged K panels and accumulating output
versions. Attention exposes Q/K/V copies, products, softmax and merge. Key/value packing
on a small owner grid and broadcast to the compute grid are separate mid copies. Selected
view slices are mapped copies whose output is an ordinary mid value. Tile
expansion shares physical copy realization across all these uses, including
padding, direct resident views and destination packing.

**General copy/view chain composition (#4) remains deferred at the user's
request.** Discuss its overlap with planning before implementing it. Resolving
already selected deferred views is part of the current boundary rewrite; it is
not an arbitrary producer/consumer layout optimization pass.
