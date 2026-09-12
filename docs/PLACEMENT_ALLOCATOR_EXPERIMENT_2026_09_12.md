# Placement search experiment, 2026-09-12

The standalone [prototype](../tools/placement_experiment.py) recovers a real
fragmented tile and improves on the current two-order allocator on synthetic
bank-constrained instances. It is not integrated into production. The difficult
near-capacity cases show why adding more allocation orders is insufficient.

## Inputs and reproducibility

The tool accepts the [placement constraint dumps](PLACEMENT_CONSTRAINT_DUMPS.md),
including region boundaries, physical element sizes, reserved ranges, alignment,
Repeat strides, lifetimes, host-aperture restrictions, and distinct-element
edges. It also accepts the earlier two captures in
`artifacts/fragmentation-20260912/captures`.

```sh
python3 tools/placement_experiment.py \
  artifacts/fragmentation-20260912/captures/0-0.json \
  artifacts/fragmentation-20260912/captures/1-0.json \
  --random-cases 100 --seed 20260912 --budget 64 \
  --output artifacts/fragmentation-20260912/experiment.json
python3 tools/test_placement_experiment.py
```

Synthetic instances contain eight physical elements divided into random lanes.
Each lane has sequential random lifetimes and buffers occupying 75–100% of its
width. Distinct-element edges are sampled between simultaneously live buffers
in different elements. The inputs contain mixed standard/interleaved access and
8/32/64-byte alignment, and requests are shuffled. The construction supplies an
independently validated feasible witness that is never given to the allocator.
The main set has a ninth element available: 12.5% capacity headroom relative to
the witness's eight-element arena. The first ten seeds are also tested with
only eight elements. These are deliberately dense adversarial instances, not a
claim about the distribution of real model allocations. The captures, unlike
the synthetic set, exercise persistent weights, reserved holes, and Repeat.

A separate validation routine checks completed placements for exact size,
alignment, containment, region eligibility, lifetime overlap and bank conflicts.
It also validated all **156 successful production dumps** produced by the
compiler's randomized GEMM placement test in `/tmp/ipu-placement-dump-check`.
Four regression tests deliberately introduce invalid overlap, bank placement,
and stride, and distinguish a capacity proof from search exhaustion.

## Algorithm

The prototype reproduces the current lifetime-first and alignment-first orders
using offline lifetime subtraction, then tries allocations ending at model
completion first (descending size within that group). It retains all placement
constraints and uses the same preference for lower addresses and region 0.

On failure, it generates up to four revised orders by promoting the failed
request before actual earlier blockers: earliest, middle, latest, and largest.
A priority queue favors branches that placed more requests. Orders are
deduplicated and the number of complete placement attempts is capped at 64.
A peak-live-byte lower bound rejects provably over-capacity problems first.
An exhausted budget reports **unknown**, never infeasible.

This is a small failure-directed repair experiment, not an implementation of
MiniMalloc or TelaMalloc and not an exact solver. It does not enumerate
alternative addresses for a fixed request order.

## Results

| Set | Current two-order union | With end-first order | Bounded repair |
| --- | ---: | ---: | ---: |
| Two failing real captures | 0/2 | 1/2 | 1/2 |
| 100 feasible synthetic cases, ninth element available | 36/100 | 61/100 | 89/100 |
| 10 feasible synthetic cases, eight elements only | 0/10 | 0/10 | 0/10 |

The two captures are not both fragmentation failures:

- `0-0.json`: 482,496 available bytes, 458,368 peak live bytes. The end-first
  order fits all 232 requests. This only proves feasibility of the captured
  tile, not the entire proposed model placement.
- `1-0.json`: 498,872 available bytes, 515,200 peak live bytes. The tool proves
  a 16,328-byte capacity deficit before searching. Placement cannot fix this
  candidate under its existing lifetime and kernel constraints.

On this host, the 100 main synthetic cases took 3.06 seconds total in Python
for bounded repair: median 11 ms, p95 142 ms, maximum 223 ms. The tight set took
1.02 seconds total, maximum 272 ms. The feasible real capture took about 76 ms
including all three trials and validation; the capacity rejection was immediate
apart from computing the bound. Runtime is bounded by **attempt count**, not a
hard wall-clock deadline; per-attempt cost still grows with request count and
free-range fragmentation. These are experimental Python timings, not projected
Rust speedups.

Increasing the tight-case budget to 512 attempts still returned unknown for
seeds 20260912–20260914 (0.34–0.59 seconds each). More order trials alone are not
an attractive way to approach the memory limit.

## What a stronger allocator should do

The synthetic witness groups compatible buffers into memory elements. Ordinary
first-fit erodes those groupings: a buffer can fit in the earliest byte hole but
span an additional element, removing that element from several bank-conflicting
buffers. Near capacity there may be no spare element to absorb that choice.
Changing order repairs many cases with slack, but this prototype cannot directly
choose a later address or reserve a compatible element grouping.

The next useful prototype is **address branching with constraint propagation**,
in the style of MiniMalloc/TelaMalloc, inside the existing per-tile allocator:

1. Maintain candidate region/element domains for each request, rather than only
   discovering a conflict after committing its predecessors.
2. Branch on a request with few remaining feasible domains. Try aligned hole
   edges and element boundaries, explicitly allowing a later bank even when the
   first hole fits.
3. Propagate each decision to lifetime-overlapping requests and distinct-element
   neighbors. Reject a branch when any required domain empties or a compatible
   set's live-byte bound exceeds its available regions.
4. Backtrack with a node/time budget and retain the existing successful
   placement when optimization runs out of budget.

The ordinary rectangle-packing canonical-placement and dominance rules must be
rechecked against our bank and region constraints; importing them unchanged
could silently prune feasible placements. A small exact CP-SAT/MILP reference
would help evaluate the stronger prototype, but is not a prerequisite for the
cheap end-first order. Neither approach requires changing the model plan or
adding another whole-model compilation retry layer.
