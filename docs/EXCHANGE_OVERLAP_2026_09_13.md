# Separating activation and parameter exchange

The BS1 early-downprojection-cast recipe was rebuilt after profile-buffer
allocation changed. Its resolved exchange snapshot is
`artifacts/exchange-overlap-20260913/early.json`. This is a controlled comparison
at the rebuilt placement, not a claim that its addresses or timings exactly
match the older misranked binary. No search or layout change was made.

Split phases 23 (MLP up) and 27 (MLP down) by whether the source address changes
between Repeat iterations. In these phases those are the resident layer-weight
sources; the stationary sources are activations. Independently schedule each
partition using the production ordinary/paired selection with a B1024 cache.
All three variants retained ordinary transfer width here. Validate each result
with the production exchange validator. Do not change source/destination
addresses, message payloads or destinations.

| Phase | Combined horizon | Weights horizon | Activation horizon | Separate sum | Separate minus combined |
|---|---:|---:|---:|---:|---:|
| MLP up / 23 | 17,676 | 9,286 | 8,270 | 17,556 | −120 |
| MLP down / 27 | 15,957 | 8,353 | 6,025 | 14,378 | −1,579 |

These are scheduled event cycles, excluding phase setup, global barrier,
profiling and address patching. Therefore they do not establish an end-to-end
hardware improvement. Separation adds a boundary but enables one whole-phase
OUTGOING_BASE for weight transfers and removes the long mixed-source Repeat
patch lists described in `EXCHANGE_PATCH_RANKING_2026_09_13.md`.

Maximum per-tile row bytes (for separate phases, sum their bytes on each tile
before taking the maximum):

| Phase | Combined | Separate |
|---|---:|---:|
| MLP up / 23 | 5,280 | 5,256 |
| MLP down / 27 | 6,492 | 6,580 |

These count emitted exchange instructions, not final cross-phase row sharing,
patch tables, host support or added invocation code.

## Why overlap is not buying payload throughput here

| Phase | Maximum weight RX | Maximum activation RX | Maximum combined RX | Tiles hitting both separate maxima |
|---|---:|---:|---:|---:|
| MLP up / 23 | 7,680 | 3,936 | 11,616 | 780 |
| MLP down / 27 | 5,120 | 4,880 | 10,000 | 1,170 |

Counts are ordinary 32-bit words, hence receive-bus cycles. Sender maxima are
much smaller: 1,184 combined words for up and 1,972 for down. Both traffic classes
bottleneck the same receive endpoints. The combined phase cannot hide one
payload behind the other on those receivers. Combining may still reduce barriers
or scheduling slack, but the theoretical endpoint bound equals the sum of the
separate bounds, and this scheduler produces less slack when they are separate.

No conclusion about the SDK's actual parameter scheduling was established by
this experiment.

## Reproduction artifacts

`artifacts/exchange-overlap-20260913/` contains the build script/log, full snapshot,
filtered `combined.json`, `weights.json`, `activations.json`, and `select.rs`.
The latter calls `ExchangeScheduleCache::with_stream_words(1024)` with fresh
production selection per phase and validates the generated schedules. The
`*-selected.json` files contain timings and complete per-tile row-byte arrays.
The ordinary-width replay benchmark logs independently give the same horizons.
