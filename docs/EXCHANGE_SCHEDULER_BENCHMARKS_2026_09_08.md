# Exchange scheduler captures and redesign experiments

Baseline: `82a2f36` (capture support added to `42ad0eb`; scheduling unchanged).
Final implementation: `63482ee`, following `ce3be42`, `95cabd4`, `2937b6d`, `b7c2b9d`, `d6ffbdb`,
and `fdab1be`. Local artifacts live in `artifacts/exchange-redesign-20260908/`.
The large JSON fixtures are intentionally not committed to Git. Their checksums
are recorded in `captures.sha256`; per-phase results are in `results.json`.

## Capture and replay

The new `--capture-exchange-schedule` path expands and places a planner finalist,
then uses the same transfer preparation as package generation. It exits before
scheduling, kernel compilation, linking, or device execution. It therefore
captures candidates that later fail package acceptance. Addresses are from the
ordinary provisional placement, not from the final support allocation or a
physical tile-mapping challenger. Repeat source addresses are retained.

Captured full So400m/14, 378×378, one-layer FP8 ViT candidates:

| Fixture | Finalist | Phases | Largest phase |
|---|---:|---:|---:|
| `vit-b2-f0.json` | 0 | 64 | phase 34: 586,368 transfers |
| `vit-b4-f7.json` | 7 | 59 | phase 28: 782,618 transfers |
| `vit-b4-f2.json` | 2 | 62 | phase 33: 1,172,736 transfers |

Example commands, from the repository root:

```sh
cargo build --release -p ipu-tests --bin ipu-trivial-test --bin ipu-exchange-schedule-bench
RAYON_NUM_THREADS=12 target/release/ipu-trivial-test c600-init.ipucfg \
  --workload siglip-vit-benchmark --vit-batch 4 --fp8-scale=-4 \
  --capture-finalist 2 \
  --capture-exchange-schedule artifacts/exchange-redesign-20260908/vit-b4-f2.json
RAYON_NUM_THREADS=1 target/release/ipu-exchange-schedule-bench \
  artifacts/exchange-redesign-20260908/vit-b4-f2.json --phase 33
```

The default replay fixes the captured transfer widths. `--select-widths` instead
uses production ordinary/paired selection. `--replay-cache --select-widths`
warms the production recipe cache before timing; `--relocate-by 16384` then
moves all source and destination addresses before replay. Warmup and timed
results are both validated. Output now includes whether replay actually reused
the recipe and a row fingerprint. A changed address may legitimately change the
fingerprint even when the instruction structure is identical.

The additional MLP and attention comparisons use
`artifacts/layout-sweep/unified-copy/mlp/exchange.json` and
`artifacts/layout-sweep/unified-copy/attention/exchange.json`. Older snapshots
without explicit transfer widths are not automatically migrated; the historical
Repeat-10 MLP snapshot could not be used directly for this comparison.

## Changes evaluated

* **Explicit timeline edit boundaries:** sender/control mutation records the
  earliest changed input. Encoding no longer stores duplicate input histories,
  sorts them, or compares their common prefixes from the beginning. Checkpoints
  still validate their lookahead dependency, including earlier SENDPICP choices.
* **Shared chunks:** sender events, receive controls, emitted words and encoding
  checkpoints share immutable chunks. A speculative edit copies affected chunks
  rather than a complete endpoint history. Small range extraction remains in
  encoding; the chunk directory itself is still copied when a timeline is cloned.
* **Staged transactions:** strict validation retains the successful endpoint
  updates and encoded rows. Commit uses those exact updates; repeated queries
  reuse a matching staged trial. Failed or mismatched trials cannot commit stale
  state. Sorted sender histories also support binary interval/boundary lookup.
* **Shared scheduling facts:** dependencies, predecessor/dependent incidence and
  initial endpoint pressure are constructed once per address/width problem and
  reused by greedy scheduling, strict fallback, matching and neighborhood trials.
* **Recipe scope:** ordinary and mapped candidates retain independent caches
  across finalist rejection. Immutable recipes are shared, and a structural
  fingerprint rejects incompatible problems before expensive replay. This hash
  is not a validity proof: address hazards, dependencies, encoding and normalized
  row comparison are still checked on replay.
* **Bounded critical repair:** try the existing broader priority with a queue-work
  budget of eight visits per transfer (at least 65,536). If it exhausts that
  budget, fall back to repair within incumbent-order epochs. Previously the
  purported neighborhood pass refreshed candidates throughout the entire phase
  without a work limit. Local-only repair initially regressed MLP phase 1 from
  9,600 to 9,994 cycles; preserving affordable broader searches recovers 9,600.
  Always doing both passes added overhead on smaller attention phases, so the
  final design runs local repair only when the broader pass exhausts its budget.
  Expensive opportunities beyond this budget can still be missed. A candidate
  is accepted only if its fully scheduled horizon improves the incumbent; no
  input phase is rejected merely for exhausting repair work.
* **Incremental matching waves:** dependency releases update readiness directly.
  Parallel edges to a receiver expose only their earliest incumbent edge to the
  cardinality matcher. Randomized tests compare exact orders against the former
  implementation. The production matching path is 14 lines smaller; the retained
  reference implementation is test-only.

No exchange timing constraints, memory hazards or encoding validity checks were
relaxed. These changes target compilation work, not IPU transfer bandwidth.

## Measurements

Runs used release/native-ISA binaries with `RAYON_NUM_THREADS=1`. Independent
replays ran concurrently on separate CPU capacity. Each table entry is one CPU
wall-time measurement, not a statistical median; modest differences should not
be overinterpreted. JSON parsing occurs outside the scheduler timer. Validation
is timed separately. Saved binaries and logs distinguish the baseline,
encoded-chunk, bounded-repair, staged-trial, and full-timeline experiments.

| Capture / phase | Transfers | Baseline seconds | Final seconds | Speedup | Cycles | Max row bytes |
|---|---:|---:|---:|---:|---:|---:|
| b2 / 17 | 209,068 | 99.73 | 51.67 | 1.93× | 12,911 | 12,244 |
| b2 / 31 | 391,310 | 99.09 | 31.13 | 3.18× | 5,641 | 5,788 |
| b2 / 34 | 586,368 | 109.51 | 33.37 | 3.28× | 36,105 | 8,228 |
| b4 / 16 | 325,692 | 211.38 | 108.72 | 1.94× | 21,033 | 12,828 |
| b4 / 28 | 782,618 | 295.81 | 103.22 | 2.87× | 10,040 | 11,644 |
| b4-million / 33 | 1,172,736 | 404.95 | 95.95 | 4.22× | 73,044 | 16,072 |

The complete captured MLP's scheduling time fell from 23.14 s to 15.56 s.
The smaller attention fixture was essentially unchanged (7.93 s to 7.67 s);
those small differences are within the range where CPU timing noise matters.

All six large-phase captures retained their baseline horizons and total/max row
word counts. Final row fingerprints match the staged implementation on these
captures and the attention fixture. MLP phase 1 intentionally changes back to
its baseline horizon and row sizes after recovering the broader-search result;
the other MLP phases match the staged fingerprints. The encoder's randomized
oracle additionally compares incremental output with full encoding after each
mutation, including paired transfers, loopback, rejected updates and earlier
insertions. This does not mean the bounded repair heuristic is identical to the
old global heuristic on every possible graph.

A separate production selection/relocation measurement for batch-two phase 31
(`bench-replay`, before the final timeline change) took 50.47 s for fresh selection
and 19.35 s for replay after moving all addresses by 16 KiB. Replay reported
`reused=true` and retained the 5,641-cycle horizon and 1,671,584 total row words.
The full-timeline implementation (`bench-timeline`, before the final repair
policy adjustment) also successfully replayed relocated phase 34 in 17.22 s,
retaining its 36,105-cycle horizon and row size.

## Validation and limits

164 codegen tests, 50 exchange tests and the codegen doctest pass. Clippy passes
for codegen, exchange and all test-package targets. Tests cover chunk edits and
snapshot isolation, staged commit/rejection, exact matching-order equivalence,
repeat-aware dependencies and full-versus-incremental row encoding.

The final small batch-two FP8 ViT on 64 compute tiles passed on hardware, with
maximum absolute error 0.130188 at atol 0.2 / rtol 0.05. Each changed search
policy was checked before relying on a prior device run: the full-timeline
version was byte-identical to its hardware-tested predecessor, whereas the
later repair-policy versions changed 64 tile images and received their own
hardware checks. See `small-vit-accepted-hardware.log` for the final result and
`small-vit-accepted-comparison.txt` for the reason it needed a fresh run.

This is not a claim that a full batch-four ViT now fits, or that its IPU runtime
improved. Captures deliberately bypass package acceptance, and their table sizes
are unchanged. Fast/compact whole-program selection and local repair of deferred
encoding failures remain unimplemented. Strict fallback still rebuilds a phase;
replay still rebuilds physical rows at changed addresses. Further performance
work should profile these remaining costs using the saved captures, rather than
assuming encoding is always the dominant cost.
