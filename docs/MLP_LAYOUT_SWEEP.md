# MLP layout sweep and remaining representation limits

Workload: F16 SigLIP MLP, `[1,729,1152] -> [1,729,4304] -> [1,729,1152]`,
without biases. The historical reference uses `4x92x4` then `4x24x15`, both
interleaved parameter storage, row-distributed results and complete reduction
staging. These are `(M partitions, N partitions, K partitions)`.

## Reproduction

```sh
cargo build --release -p ipu-tests -p ipu-cli
python3 scripts/mlp-layout-sweep.py --sdk "$POPLAR_SDK_ENABLED" \
    --output artifacts/layout-sweep --jobs 16
python3 scripts/analyze-mlp-layout-sweep.py artifacts/layout-sweep
```

The driver screens 4,619 geometries with an independent arithmetic/memory
proxy, then chooses representatives across K splitting, row partitions and
kernel output widths. Its 55 initial cases include two controls, independent
changes to each GEMM, swapped orientations, memory alternatives, collapsed
results, streamed reductions, and five jointly changed pairs. Afterward it
crosses the three fastest independent choices on each side. This is a
stratified exploration, not exhaustive optimization of all possible pairs.

Every selected case receives a full compilation and numerical hardware check.
The coordinator resumes completed results, records build failures separately,
and stops submitting new work after a hardware failure. Concurrent builds use
per-artifact compiler cache locks; the test executable's `--device-lock`
serializes hardware ownership after compilation. The lock file must be shared
by all coordinators targeting the same device.

The output directory contains exact command lines, binary identities, logs,
packages, raw profiles, kernel/operation summaries and exchange barrier
measurements. The analyzer emits CSV, Markdown, JSON statistics, SVG/PNG cost
plots and the existing profiler's geometry-indexed kernel measurements.
Runtime is measured from the renderer’s default cut point (the latest tile’s
initial profile entry) to the final profile sample. Full hardware-counter
durations are retained separately as `cycles`; rankings use `renderer_cycles`.
Calibration statistics exclude adaptive cross-products. Predictions and
kernel implementations stay fixed throughout the sweep; fitting and judging
a new model on the same samples would obscure ranking errors.

The third reported prediction substitutes **final placed** exchange schedule
horizons into the expanded estimate. It differs from the package builder's
provisional finalist score, which precedes its final SRAM placement search.
Exchange schedule horizons are compared separately with cycles after the last
barrier arrival; arrival spread is not added to both compute and exchange.

## What the current representation actually permits

There are three separate questions: whether ownership/storage can be
described, whether a selected mid implementation can execute it, and whether
the planner generates it. A missing candidate does not establish an IR limit.

| Choice | Present support | Remaining restriction |
|---|---|---|
| Balanced contiguous shards | `AxisTiling::shard_bounds` balances integral blocks across partitions. | GEMM K uses a whole local K block as its ownership grain, discarding several otherwise useful splits. |
| Different local K/C extents | Product expansion specializes `inner_block` and `output_columns` to each local view. | Candidate generation and operand packing must agree on finer-grained K ownership; this is not a missing machine-code capability. |
| Mixed row/column reduce-scatter | Result layouts and ordinary sum intersections can describe rectangular output grids. | GEMM generation offers `(1,1)` and one all-K row or column split, not general factor pairs. |
| Axis-order changes | Logical grid strides and `RowsFast`/`ColumnsFast` exist. | The compute-grid family and eligible result orders are narrowly generated; arbitrary permutations are not represented. |
| Disjoint or cyclic ownership | A shard has one rectangular extent per axis, or one canonical linear interval. | One tensor shard cannot own a general collection of disjoint panels. Separate values can encode pieces, at the cost of explicit structure. |
| General local strides | Named row-major, block-major and AMP orders have shared physical span machinery. | A tensor does not carry an arbitrary local blocked index map, and kernel ABIs do not accept general output strides. |

Relevant code: `mid/candidates.rs::parallel_reduction_candidates_for_orientation`,
`mid/resolved.rs::AxisTiling::shard_bounds`,
`low/expand/primitive.rs` product expansion,
`mid/implementation/gemm.rs`, and `low/expand/reduce.rs`.

## Alternatives worth evaluating after the sweep

1. **Use more of the existing rectangular layout support.** Try mixed
   reduce-scatter factor pairs, K ownership in 16-element grains, and more
   grid orders before changing the IR. In particular, the historical up GEMM
   scatters four K partials across rows; its result has 16 row groups, while
   the down GEMM gathers into four compute-row groups. An alternative result
   grid could change that gather substantially. The sweep includes matching
   downstream row groups, but does not yet expose arbitrary result factors.

2. **Decouple ownership from fixed grid linearization.** A tile mapping which
   keeps frequent communication within favorable physical groups could help
   without changing numerical storage order. Arbitrary mappings should be
   represented as a placement decision, preserving whole-device mid
   operations. The current scalar axis strides describe only regular grid
   embeddings. Pairing, source buses and SRAM element conflicts make “more
   local” an insufficient objective by itself; score complete traffic.

3. **Describe local storage with a small blocked index map.** Preserve the
   existing 16-element AMP micro-order while allowing outer panel/row strides
   to vary. A copy would materialize the same logical index map that a view
   carries. Kernel selection would advertise which maps it can read/write
   directly. The current GEMM already emits an AMP-left-compatible order and
   uses paced loads/stores, so a new generic view mechanism is not required
   merely to remove an obsolete AmpOutput conversion. Supporting arbitrary
   two-byte permutations at full AMP throughput is a separate question.

4. **Change the reduction organization, not just output placement.** The
   current `Sum` implementation chooses one packed receive buffer or one
   remote partial per global epoch. Bounded groups, a tree, or pipelined panel
   reductions could occupy more tiles with less scratch. Packed reduction
   scratch currently always uses standard SRAM. Interleaved or differently
   blocked scratch may help its kernel, but must be measured together with
   the exchange fragmentation needed to populate it. Trees also alter F16
   rounding order and require numerical validation.

5. **Panel ownership across operations.** Give a tile several separated
   hidden-channel panels, letting the up result and down activation share
   owners without requiring matching contiguous partitions. This requires
   disjoint ownership or a structured sequence of values. It also increases
   weight-feed overhead, local gather work or reduction fan-in in many grids.
   It is not automatically a win just because one exchange disappears.

These are hypotheses. The completed measurements and cost residuals should
determine which merits implementation first.

## ISA review

The IPU21 ISA manual (`~/gc-sdk/TileVertexISA-IPU21-1.3.1.pdf`) suggests
three concrete experiments; these are not changes to the sweep kernels.

* GELU currently computes four elements as two pairs. Using `f16v4mul/add`
  with broadcast constants retains the arithmetic sequence while reducing
  18 auxiliary instructions to 10, including the two existing pairwise tanhs
  (§§3.7.3.3.5, 3.7.3.3.25). Test finite F16 inputs with stochastic rounding
  disabled before timing representative shard lengths.
* Reduction can preload the first remote partial and co-issue each subsequent
  load with addition of the previous value, using a two-bundle `rpt` body.
  This preserves addition order while reducing six issue groups per partial
  to two plus setup/flush. Co-issued sources read the previous register state
  (§2.5.1); repeat bodies must be aligned complete bundles without control
  instructions (§3.7.2.13). Short lengths and single-remote cases need tests.
* GEMM's paced stores can choose independent signed 10-bit atom strides
  (§§3.7.5.1.1–2). Three eight-byte advances followed by `row_stride - 24`
  could place each 16-column group into a larger row pitch directly. This
  requires an explicit output-stride contract; it does not rearrange F16 lanes
  within each 64-bit group. Atom strides span -4096 to 4088 bytes, with
  alignment and distinct-memory-element requirements still applying.

The sigmoid rewrite of GELU is less promising: the manual rates sigmoid at
up to 16.5 ULP error and two cycles, versus accurate one-cycle tanh (Table
4.3). Sparse/broadcast loads and fixed byte shuffles are not general dense
transposes. Vector arithmetic and co-issue are the better first experiments.

## Calibration sample validity

Profile rendering can combine consecutive calls with different dimensions.
Their combined duration is valid for the timeline, but dividing it by the
invocation count does not produce a measurement for the first call's shape.
The exporter now excludes merged samples unless metadata certifies identical
symbols and scalar arguments, and new profiles record those arguments. Older
profiles remain usable for single-call samples. This correction changes
calibration metadata and export, not execution or timeline grouping.

The report also checks instruction-count formulas independently of the frozen
planner estimates. For F16 reduction with `P` partials and `n` physical
output elements, the current worker loop predicts
`282 + 6 * ceil(n/48) * (9 + 6*(P-1))`. For aligned GELU it predicts
`300 + 558 * ceil(n/96)`; old profile metadata gives logical sizes, so padding
can make that diagnostic undercount waves. Interleaved F16 GEMM is close to
`294 + ceil(K/16) * ceil(C/16) * (4*M + 160)`, taking physical `M` from the
selected symbol. This last expression already resembles candidate screening
but differs from primitive costing's larger fixed per-group coefficient.
These formulas are diagnostics, not a model fitted and evaluated on the same
sweep. Tail paths, other precisions and standard weight storage require their
own checks before changing planner prices.
