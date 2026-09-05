# Layout and barrier diagnosis, 2026-09-05

This investigation precedes further view/copy architecture changes. Summed
kernel work is not a critical-path cost: sparse local work can hold up an entire
exchange boundary. No compiler choices or kernels were changed in this pass.

## Reproduce the measurements

The new CLI command separates arrival skew from transfer execution:

```sh
ipu-stack profile-barriers profiles/example.ipuprofile
ipu-stack profile-barriers profiles/example.ipuprofile --json
```

It reports shared-clock first/last exchange entry, last exit, last-arriving tile,
arrival spread, duration after the last arrival, and static scheduled event
cycles (unknown when absent from an older profile). Arrival spreads at different
boundaries can overlap and must not be summed. Static event timing is schedule
metadata; entry/exit timing is measured. The difference also includes boundary
execution overhead, not just transfers.

Use `profile-query --at-offset N --samples 20` for the active samples at a
particular point; its default origin is cropped, unlike `profile-barriers`, so
pass `--shared-clock` when correlating offsets. HTML rendering already works in
headless Chromium. The CLI now writes tracing to stderr so JSON stdout is valid.

Local artifacts under `artifacts/profiles/` include rendered `attention.html`,
`mlp.html`, `mlp-historical.html` and corresponding `*-barriers.json` reports.
Input profiles are `/tmp/whole-mid-final-attention.json`,
`/tmp/whole-mid-final-mlp.json` (both binary despite the extension) and
`profiles/siglip-mlp-f16-b1-scheduler-integrated.ipuprofile`.

## Attention

The three projections each reduce nine partials onto 162 tiles. The reduction
samples reach 37,902 cycles. In the interval between the previous exchange's
last exit and the next exchange's last entry, compute occupies only about 10.7%
of device tile-time. Thus the low projection occupancy is primarily the selected
reduction distribution, not evidence of inefficient dense GEMM instructions.
The GEMM intervals themselves have roughly 93.5% measured compute occupancy.
These are time-in-compute measurements, not AMP utilization.

The next three preparation exchanges (physical phases 6, 7, 8) have:

| Phase | Transfers | Scheduled cycles | Measured cycles after last arrival | Endpoint-only lower bound |
|---|---:|---:|---:|---:|
| 6 | 62,342 | 33,542 | 33,780 | 2,624 |
| 7 | 60,816 | 36,987 | 37,224 | 2,624 |
| 8 | 58,508 | 37,377 | 37,614 | 2,624 |

All report maximum identical destination count 1. These are fragmented mapped
Q/K/V distributions, not the subsequent regular block broadcasts. Endpoint-only
bounds omit scheduling constraints; their gaps do not establish achievable
speedups. Sparse strided-copy work extends the critical path between these
exchanges: after phase 7's last exit, the next last arrival is 17,376 cycles
later, with only a few tiles doing local work. Much of other tiles' long
exchange samples is early-arrival waiting.

V packing is followed by a 5,526-cycle tail after phase 8's last exit, before
the first block broadcast's last arrival. Many packing calls run earlier on
tiles that have already finished their exchange; a kernel work total alone
misses this tail. The steady-state attention loop's inter-exchange compute
intervals are roughly 91% occupied; its preparation is a different problem.

## Historical versus current MLP

The historical profile has a 205,392-cycle cropped span. Current measured maximum
tile duration is 327,144 cycles (cropped renderer span 326,928). Historical exact
workload provenance still needs confirmation; names and kernel dimensions are
consistent with the canonical workload but are not a complete manifest.

| Quantity | Historical | Current |
|---|---:|---:|
| First GEMM partial count | 4 | 18 |
| Second GEMM partial count | 15 | 30 |
| First GEMM longest reduction sample | 7,920 | 31,770 |
| Second GEMM longest reduction sample | 8,094 | 18,036 |
| Four scheduled exchange durations | 7,179 / 5,638 / 45,670 / 12,527 | 5,405 / 48,168 / 37,019 / 53,720 |
| Scheduled exchange total | 71,014 | 144,312 |

Some reduction samples merge two invocations: these maxima describe actual
elapsed local work, not per-call latency. Both executions lack standalone
rearrangement kernels. Recovering the earlier distribution is a concrete
experiment independent of new layout-capable kernels.

## What this says about costing

Current MLP's compact beam exchange estimate is 139,960 cycles versus 144,312
scheduled event cycles. Attention's estimate is 381,240 versus 301,869 scheduled.
Therefore a blanket exchange-cost multiplier is not justified. Aggregate
agreement also does not demonstrate correct relative ranking of candidates.

`estimate/mid.rs` assumes 256-byte fragments and estimates sends from destination
bytes and source/destination tile counts. It does not model the coordinate map's
actual fragmentation, source hot spots, or the exact distribution of boundary
arrival times. It separately sums primitive costs, which can overestimate
concurrent work while missing a sparse implementation tail elsewhere.

Next controlled comparisons should retain the historical MLP grids alongside
current candidates, report per-primitive costs, and compare their predicted
ranking with scheduled/device results. For attention, compare projection output
ownership and preparation mappings before changing kernel access capabilities.
Keep coarse costing in mid; improve compact geometry/fragment estimates only
where these comparisons establish a ranking error. Exact scheduler execution
belongs in finalist validation, not beam expansion.

Validation: all nine `ipu-profile` release tests pass, including a new regression
for arrival skew, repeated epochs, cycle-counter wrap and missing schedule
metadata. Strict Clippy passes for `ipu-profile` and `ipu-cli`; the command was
run against all three profiles and its JSON parsed directly.
