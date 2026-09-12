# SigLIP optimized execution and PV padding — 12 September 2026

The full 27-layer batch-two optimized plan now passes resident hardware validation.
Commit `51a438e` fixes the NaNs reported during the previous local-search check.

## Cause and fix

Softmax produces 729 useful FP16 probabilities, pads that row to 768 columns,
and stores FP32 maximum/denominator state in the following panel. Its allocation
therefore exposes 784 FP16-sized columns, although the final panel is metadata.

The selected distributed PV product uses five K partitions of 160 columns: 800
in total. Its operand copy previously copied through column 784, including the
statistics, and zeroed only the remainder. Finite FP32 bit patterns can encode
FP16 NaNs. The corresponding V coefficients are zero, but zero times NaN is NaN.
The passing materialized control had a grid padded to 768, avoiding the defect.
The preserved Q layout was not the cause.

Attention now presents the probability-only prefix as an explicit mid copy/view
before distributing it. Low expansion aliases the local prefix; there is no new
packing kernel. Padding beyond the prefix is initialized normally. Keeping the
already-zero, aligned probability padding also avoids half-word FP16 exchanges
at the odd logical tail. The FP32 statistics remain available to attention merge.

The regression expands a five-way PV split and checks that no probability
transfer reads columns 768 onward. It fails with the old implementation and
passes with the fix. The complete codegen suite passes: 245 tests, four ignored.

## Full-model results

Randomized weights, fused QKV, FP8 dense operands at scale −4, FP16 attention,
B1024 exchange scheduling, eight local optimization attempts, no cycle profiling.
Each package uploads parameters once and validates two resident inference calls
against FP32. A separate two-call replay checks exact saved hardware outputs and
measures image upload, inference and embedding download, excluding initial weight
upload and device attachment; host polling is 100 µs.

| Full 27-layer SigLIP | Batch 1 | Batch 2 |
|---|---:|---:|
| Baseline policy | Normal | Capacity-first |
| Baseline estimated cycles | 14,500,121 | 25,281,807 |
| Optimized estimated cycles | 12,062,025 | 22,028,118 |
| Estimated reduction | 16.8% | 12.9% |
| Maximum encoded exchange storage | 30,504 B/tile | 68,388 B/tile |
| Minimum FP32-reference cosine | 0.994102280 | 0.993989563 |
| Maximum absolute error | 0.394307 | 0.424015 |
| Measured batch latency, two-call mean | 9.054 ms | 15.866 ms |
| Measured image throughput | 110.4/s | 126.1/s |
| Package planning time | 672.7 s | 467.1 s |

The individual latencies are 9.142/8.965 ms and 15.972/15.760 ms. Batch two's
previous unoptimized resident latency was 18.455 ms, so the optimized measured
latency is 14.0% lower. These are short host-timed samples, not cycle captures.
The two compiler runs overlapped on the host, with 24 Rayon threads each.

The batch-one build started before the padding fix, alongside its diagnosis. Its
PV grid pads to 768 and does not read the statistics; it passed both hardware
calls and replay. The batch-two build uses the fix and reselects the formerly
failing plan. No performance claim here relies on the invalid NaN-producing run.

Artifacts under `artifacts/nan-20260912/{bs1,bs2-fixed}/` contain the packages,
`build-run.log`, `latency.log`, saved resident fixtures and memory profiles.
Exact placement: [batch one](../artifacts/nan-20260912/bs1/memory/placement-467957.html),
[batch two](../artifacts/nan-20260912/bs2-fixed/memory/placement-469868.html).

Reproduction uses the full-model command in
[the relay report](EXCHANGE_GATHER_2026_09_12.md#full-model-revalidation), changing
`--optimization-steps` to 8. Batch one uses `--vit-batch 1` and omits
`--capacity-baseline`; batch two retains it. Set `RAYON_NUM_THREADS=24` and
`RUST_LOG=info,ipu_codegen::place=debug` to capture allocator failures as well.

## Why other candidates fail placement

The terse allocation errors name the request that failed the allocator's final
attempt, not necessarily the allocation responsible for the pressure. In
particular, the 82,944-byte request is **persistent QKV weights**, not scratch:
27 layers × 3,072 bytes per tile. The 27,648-byte request is the attention-output
weight sequence, 27 × 1,024 bytes. There is no additional persistent replication
in these local proposals; their input layouts are fixed to the incumbent's homes.

Examples from the fixed batch-two run, after size-ordered placement also failed:

| Rejected request | Tile | Usable free bytes for persistent data | Largest free span |
|---|---:|---:|---:|
| QKV sequence, 82,944 B | 0 | 83,632 B | 42,408 B |
| Output-projection sequence, 27,648 B | 40 | 51,648 B | 26,800 B |

The QKV free spans are 8,456, 32,768 and 42,408 bytes. Another 3,576 bytes remain
in the host aperture, but resident parameters cannot occupy it. Thus total bytes
alone would just suffice; no individual span holds the sequence. These holes
are after already placed allocations overlapping the sequence's persistent
lifetime, not a global average across tiles.

The lifetime-ordered first attempt instead fails on interleaved MLP scratch;
the size-ordered fallback moves the failure to a persistent sequence. Larger
scratch/support reservations and memory-element constraints change the available
holes. Repeat currently groups each sequence into a contiguous, uniform-stride
allocation. These failures therefore reflect placement/contiguity pressure, not
an exchange-table-budget rejection. No allocator policy or memory limit was
changed for this diagnosis. Extracted examples and the original log lines are
saved in `artifacts/nan-20260912/placement-failures.json`.
