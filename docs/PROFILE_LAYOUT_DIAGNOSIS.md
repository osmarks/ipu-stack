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

## Controlled historical-grid experiment

The reconstructed GEMM grids are supported by the current planner. Force them
with these arguments to `ipu-trivial-test --workload siglip-mlp-benchmark
--mlp-batch 1`:

```text
--gemm-plan-constraint 0:4x92x4:4x1:48:interleaved:normal:complete:direct
--gemm-plan-constraint 2:4x24x15:15x1:48:interleaved:normal:complete:direct
```

The first triplet is compute rows/columns/partials. The second pair subdivides
each compute result, so these select 16x92 and 60x24 result distributions.
Current input placement, padding and exchange scheduling still apply; this is
not a byte-identical reproduction of the old compiler/package.

| Canonical MLP measurement | Automatic grids | Reconstructed historical grids |
|---|---:|---:|
| Compact beam cycles | 413,118 | 310,377 |
| Compact beam exchange cycles | 139,960 | 123,960 |
| Expanded analytical cycles | 568,612 | 250,045 |
| Measured maximum tile cycles | 327,144 | 232,722 |
| Maximum numerical error | 0.011719 | 0.011719 |

The historical-grid run passes all hardware numerical checks, uses 1,440 active
compute tiles, and is about 29% faster than the automatic selection. It remains
slower than the saved historical profile. The profiled executable and build log
are `/tmp/historical-layouts-new-model.{ipuprofile,ipuexe,log}`;
the renderer output is `artifacts/profiles/mlp-historical-layouts-new-model.html`.

Temporary instrumentation of candidate generation located the loss before mid
costing: BOTH historical grids survive the initial grid proxy frontier. The
two grids produce fully specified variants, but neither survives
`retain_operator_candidates`. For the default cost model that shortlist ranks
by boundary memory bytes (`operator_cycle_override` is absent), not by compact
mid execution cost. Some later calls enumerate different tile budgets and lack
the first grid, but that is not where its original variants were lost. The
instrumentation was removed after the experiment. This identifies a candidate
screening problem; increasing the final exchange price cannot recover an
already discarded candidate.

## Q/K/V batching experiment

The implementation is saved on branch `experiment/qkv-exchange-batching`.
It extends the existing low exchange consolidation to move independent copy
preparation before a combined exchange and completion work after it. It only
crosses local copies, zero-fill, rearrangement and cast kernels within the same
source operation. Whole-allocation alias/dependence checks reject unsafe moves;
operator compute, checkpoints and repeats remain barriers. Mid layouts and the
selected algorithms are unchanged. The main refactor branch retains its previous
behavior because this experiment substantially increases compilation time.

| Projected attention | Separate preparation | Batched preparation |
|---|---:|---:|
| Exchange phases | 21 | 19 |
| Q/K/V preparation transfers | 181,666 total | 181,666 |
| Scheduled preparation cycles | 107,906 total | 100,381 |
| All scheduled exchange cycles | 301,869 | 294,344 |
| Measured profile span | 782,088 | 775,188 |
| Build and numerical validation | 49.62 s | 317.77 s |
| Maximum error | 0.001230 | 0.001230 |

The batched executable passes all 839,808 numerical checks. GEMM smoke and
attention smoke also pass with the modified lowering. Existing codegen release
unit tests (93), its doctest, the added dependency-ordering regression and strict
Clippy pass. This was a controlled experiment, not a full hardware strategy sweep.

The elapsed improvement is 6,900 cycles (0.88%). The combined phase's last entry
is 284,280 and last exit 384,918 on the shared clock; the next broadcast's last
entry is 392,628. Thus sparse prerequisite work is now before the combined
exchange, with another 7,710-cycle tail afterward. It was not eliminated by
removing the intermediate boundaries. Headless Chromium inspection confirms
one long preparation exchange followed by the same regular attention blocks.

The scheduler takes 147.97 s provisionally and 162.62 s after placement for the
larger phase. It still selects the full-duplex schedule. Reserved exchange-table
space is 23,148 bytes in BOTH baseline and batched builds; batching did not
increase that reservation. The earlier progress statement describing it as
increased was incorrect.

Results: `/tmp/batched-preparation.{ipuprofile,ipuexe,log}` and
`artifacts/profiles/attention-batched-preparation.html`, plus the matching
barrier JSON. Both newly rendered profiles were checked in Chromium.


## Execution-cost shortlisting (2026-09-05)

The operator shortlist now prices compact mid implementations, including staging
and reductions. Preliminary beam ranking uses the same implementation prices.
Only shortlisted regions receive full composed liveness evaluation. Neither step
expands tiles or invokes physical scheduling. Geometry diversity preserves both
reduction fan-in and result subdivision, without spending each diversity slot on
another memory/staging variant of the same geometry. A regression checks that a
historical first-GEMM grid survives despite its larger boundary storage.

The cache now retains compact implementations for the duration of one search;
candidate evaluation uses the existing bounded parallel pool. Weak references
caused repeated reconstruction during screening. An intermediate uncached trial
spent 83 seconds planning; final MLP planning took 11.419 seconds (with concurrent
test compilation). The coarse movement model also uses native packed column
grain for linear redistribution: F16 AmpLeft gets 32-byte representative payloads
instead of 256 bytes. This is a conservative heuristic, not span enumeration.
An intermediate choice otherwise appeared cheap but expanded roughly 195,000
movement fragments.

| Measurement | Previous automatic | New automatic |
|---|---:|---:|
| MLP maximum tile benchmark cycles | 327,144 | 229,314 |
| MLP compact estimated cycles | 413,118 | 307,309 |
| MLP expanded analytical cycles | 568,612 | 282,747 |
| MLP build and numerical validation | 244.67 s | 71.93 s |
| Projected attention cropped profile span | 782,088 | 523,980 |
| Attention build and numerical validation | 49.62 s | 43.05 s |

Device improvements are 29.9% and 33.0%, respectively. These results retain the
main branch's separate Q/K/V preparation, without experimental batching. Both
new profiles were rendered and inspected in Chromium:
`artifacts/profiles/mlp-shortlist.html` and
`artifacts/profiles/attention-shortlist.html`. Attention preparation still has
substantial idle time despite the faster overall selection.

Tradeoffs: MLP peak build RSS is 1,718,228 KiB (1.64 GiB), versus about 1.31 GiB
previously. Its replicated host input payload grows from 45,411,840 to 272,471,040
bytes; weights remain 19,869,696 bytes. Reported device cycles exclude host
initialization, so this is a device-execution improvement, not a measurement of
end-to-end request latency. The new cropped MLP profile span is 228,654 cycles;
it differs from the maximum tile benchmark counter above.

Validation: all 156 workspace release tests and strict workspace Clippy pass.
Hardware passes full MLP (maximum error 0.011719), full projected attention
(839,808 checks, error 0.001230), GEMM, batched GEMM, attention smoke, repeated
MLP, and forced materialized attention (839,808 checks, error 0.000930).
New linear layouts exposed an in-place pointwise bug: expansion selected the
first shard on a tile instead of the shard with matching extents. The fix and
multi-shard alias regression are committed separately as f360c4a.
Build/profile inputs are `/tmp/shortlist-fragments-{mlp,attention}.*`;
additional validation logs are `/tmp/shortlist-validated-*.log`.

## Why reconstructed historical grids remain slower

Compare the same profile window: the saved historical MLP spans 205,392 cycles;
its reconstructed grids span 225,930, a difference of 20,538 cycles (10.0%).
Comparing the old cropped span directly with the new 232,722 maximum-tile
benchmark counter overstates the regression. Historical workload provenance is
incomplete; this is a grid reconstruction, not a byte-identical package replay.

Shared-clock phase analysis shows the initial exchange and first GEMM are
identical. First reduction work is 11,061,624 tile-cycles over 1,472 tiles, with
a maximum sample of 7,920 cycles in both profiles. GeLU work is also identical.
Before the second GEMM, the new lowering clears destination padding by zeroing
whole buffers on 432 tiles. Its largest fill takes 13,482 cycles and clears
50,048 F16 elements. The next exchange's last arrival moves from 93,636 to
107,010, a 13,374-cycle delay.

That exchange's scheduled duration grows from 45,670 to 53,856 cycles (+8,186).
Its measured duration after the last arrival grows by 8,208. The precise source
of this remaining mapping/placement/scheduling difference has not been isolated.
The subsequent GEMM recovers about 1,100 cycles of the accumulated gap. The final
exchange schedule is identical (12,527 cycles), as is final reduction work:
10,195,632 tile-cycles over 1,440 tiles, maximum sample 8,094 cycles. The final
profile endpoints differ by 20,538 cycles.

Thus reduction throughput is not the cause for these fixed grids. Most of the
regression is the new padding-clear tail plus the longer redistribution schedule.
`low/expand/conversion.rs::prepare_mapped_views` currently clears whole copy
buffers when `CopyPlan.clear_padding` is set. Required K padding cannot simply
be left uninitialized. Clearing only uncovered physical ranges is the next
specific optimization; it has not been implemented in this checkpoint.


## Padding, SRAM placement and paired transfers (2026-09-05 follow-up)

### Padding is read, but whole-buffer clearing is excessive

Controlled full-MLP hardware runs distinguish zero padding from irrelevant data:

| Historical grids, current compiler | Numerical result | Maximum tile cycles |
|---|---|---:|
| Normal zero initialization | PASS, error 0.011719 | 232,722 |
| Omit copy destination clears | PASS, error 0.011719 | 219,384 |
| Fill tails with alternating finite 1.0/0.0 | PASS, error 0.011719 | 232,734 |
| Fill tails with alternating half NaN/0.0 | All 839,808 outputs NaN | — |

The automatic grid also passes with clears omitted (221,352 cycles versus
229,314). Thus the omission is not necessarily a numerical failure on a fresh
run. Host packing zeroes weight padding, so finite activation garbage in padded
K is multiplied by zero. NaNs are not neutralized by those weights. A destination
with K range 4032..4304 is physically read through 4320; those last 16 entries
are real kernel reads. The diagnostic modified the fill helper to use immediate
`setzi` values and left all subsequent copies unchanged. No diagnostic runtime
or clear suppression is retained in production.

The historical profile has no equivalent fills, but no corresponding historical
package is available to prove its initialization/reuse contract. The evidence
supports retaining initialization of unwritten K tails, not clearing every byte
of every destination. Row-only padding also triggers whole-buffer clears even
though padded output rows are not observable. A future optimization should clear
only uncovered ranges, or prove an existing producer/initialization already
establishes the required padding. It must not assume uninitialized half values
are finite. Current initialization is deliberately unchanged in this checkpoint.

Inputs: `/tmp/historical-{no-clear,finite-clear,poison-clear}.log`,
`/tmp/automatic-no-clear.log`. The poison comparison rejects NaNs even though its
existing diagnostic prints a misleading maximum absolute error of zero.

### The exchange regression comes from placement-sensitive scheduling

The saved historical snapshot and a fresh reconstruction have exactly the same
multiset of sources, destinations and transfer lengths in every phase. Phase 2
has 5,562 transfers, 118,812 destination endpoints, 4,057,128 source words and
47,503,680 delivered words in both. Replaying the historical snapshot with the
current word scheduler gives its original 45,670-cycle horizon. Reordering the
new snapshot to historical geometry order does not recover it.

| Phase-2 snapshot experiment | Word horizon | Paired horizon after lane fix |
|---|---:|---:|
| Saved historical addresses | 45,670 | 33,081 |
| Reconstructed current addresses | 54,090 | 41,599 |
| Current interleaved addresses shifted together by 4 KiB | 46,056 | 33,081 |

The reconstructed standalone replay differs slightly from the prior packaged
53,856-cycle schedule: package construction may reuse a provisional ordering.
Use the same captured replay input when isolating scheduler changes.

All source and destination addresses in interleaved SRAM were shifted together,
preserving data dependencies and payloads. The word relocation passed hardware
replay (8,192 checked words); the paired relocation passed offline validation.
This is an address-placement experiment, not a proposed universal 4 KiB offset
or a complete relocated MLP package. Other offsets were worse: 8 KiB gave
51,433 word cycles, 16 KiB 57,826, and 32 KiB restored 54,090.

The effective interleaved memory-element size is 32 KiB. Placement changes which
simultaneous sends and receives contend for these elements. The raw number of
potentially conflicting transfer pairs even falls from 9,464 to 8,915; their
position in the critical schedule matters more than their count. For example,
tile 0's activation sends begin at 0x81140 historically and 0x80000 now; its
incoming weights begin at 0x82280 and 0x81140 respectively. The aggregate traffic
and endpoint-only lower bound do not capture this change.

Reproduction inputs: `profiles/siglip-mlp-f16-b1-scheduler-integrated.exchange-schedule.json`,
`/tmp/historical-current-exchange.json`, `/tmp/interleaved-shift-4096.json` and
`/tmp/paired-interleaved-shift.json`. Replay with
`target/release/ipu-exchange-schedule-bench SNAPSHOT --phase 2`.

### Paired transfers were overconstrained

The sender's paired transfer borrows its partner's transmit lane, but the
scheduler reserved BOTH directions on that partner. Hardware verifies that the
partner can receive concurrently. In the first 64-tile probe, the borrowed lane
is occupied during cycles 31..2079 and a normal receive on that tile runs during
165..2213. All 8,192 sampled words pass. Both source-pair orientations and both
scheduling orders were subsequently checked. Unit coverage also checks that a
local send still cannot overlap the borrowed transmit lane.

The fix applies consistently to the ready queue, exact row builder, critical
neighborhood ordering, predecessor bookkeeping, endpoint lower bound and
schedule validator. The row builder retains the borrowed transmit horizon
separately from receive events. This is committed as 0f90bf7.

Before the fix, pairing all 4,028 eligible phase-2 transfers was worse:
57,741 cycles on old addresses and 57,122 on new addresses. Source-cohort trials
found only a tiny historical improvement (45,624) and a modest current one
(51,301). After the fix, pairing all eligible transfers gives the much better
33,081/41,599 horizons above. Both complete phases passed hardware replay with
65,536 checked words each.

The previous per-transfer search required an individual substitution to reduce
the global horizon before exploring combinations. It could miss improvements
across tied paths and rebuilt the entire phase for every trial. The replacement
compares ordinary transfers with all eligible paired transfers, reoptimizing
both complete choices and accepting pairing only for a strictly shorter horizon.
It performs at most two optimization calls and preserves the ordinary fallback.
The existing provisional/final-placement reuse still validates actual rows.

The experiment worktree `../ipu-stack-exchange-scheduler` was also reviewed.
Its August 15 brief explicitly models simultaneous send/receive and SRAM hazards;
these experiments were not all from before full duplex. For example, the exact
checkpoint beam evaluated 32,640 alternatives on the old phase-3 fixture and
still finished at 16,388 cycles. Those marginal results do not test the corrected
paired-lane model. None of that search machinery was imported.

### Whole-program result and remaining imbalance

| Full MLP, with normal zero initialization | Before lane/search fix | After |
|---|---:|---:|
| Automatic maximum tile cycles | 229,314 | 222,408 |
| Historical-grid maximum tile cycles | 232,722 | 221,124 |
| Automatic cropped profile span | 228,654 | 221,742 |
| Historical-grid cropped profile span | 225,930 | 214,332 |

Full MLP numerical checks pass with error 0.011719. Builds plus hardware checks
took 72.32 seconds automatically and 23.56 seconds with forced historical grids.
Profiles: `artifacts/profiles/mlp-paired-fixed.html` and
`artifacts/profiles/mlp-historical-paired-fixed.html`. The automatic profile was
inspected in Chromium. Projected attention remains at 523,980 profile cycles and
passes all 839,808 numerical checks, error 0.001230.

The current MLP's visibly worse balance is real. First GEMM: 495 tiles have 16
output columns while 963 have 32, with kernel durations about 27–28k and 54–56k
cycles. Historically only 112 tiles had 32 columns while 1,360 had 48; their
kernels took about 32k and 48k. Measured compute duty before the first reduction
exchange falls from 96.4% to 78.8%. The final reduction's longest call rises from
8,094 to 14,328 cycles (15 versus 27 partials), although the first reduction gets
cheaper. These are different selected workloads per tile, not slower identical
kernels. Compact costing does price maximum local geometry, but its coarse
exchange model lacks physical bank placement, exact fragmentation and overlap.
The automatic and forced historical compact estimates differ by only about 1%.

The next useful targets are selective padding clears, placement feedback that
accounts for exchange SRAM conflicts, and better tradeoffs between output-column
balance and reduction fan-in. A more complicated global scheduler search is not
required to explain the regressions observed here.

Validation for the completed checkpoint: all 158 workspace release tests and
strict workspace Clippy pass. Hardware checks pass the four paired-lane probes,
both complete paired MLP exchange replays, both full MLP layouts, GEMM and batched
GEMM smoke, attention smoke, repeated MLP, projected attention, and forced
materialized attention (839,808 checks, maximum error 0.000930). Logs are
`/tmp/paired-portfolio-{tests,clippy}.log`, `/tmp/paired-final-validation-results.log`
and `/tmp/paired-validated-*.log`.

## Selective copy initialization (2026-09-06)

Copy expansion now computes the union of destination byte spans and clears its
complement. Summing copied element counts was insufficient: overlapping writes
could conceal an unwritten tail. Semantic coverage excludes physical padding;
physical copies include the bytes actually transferred. This stays in low tile
expansion, outside whole-device mid plans and the planning beam.

The existing eight-byte fill kernel accepts a byte offset and length. Holes are
rounded outward because initialization precedes all copies. Nearby ranges are
merged when clearing their intervening bytes costs less than another launch
(using the existing launch and fill-throughput estimates). This matters on
attention: issuing every hole separately increased fill samples from 1,984 to
35,072 and regressed the cropped profile from 523,980 to 545,340 cycles, despite
unchanged exchange schedules.

Two local overwrite guarantees remove redundant initialization entirely:
fully populated row-major staging needs no clear, and implemented destination
packers write every physical output element, explicitly zeroing padding. Neither
proof depends on the previous occupant of an SRAM allocation. Range bounds and
ABI arguments have regression coverage, alongside randomized coverage checks,
overlapping mappings and padded views into unpadded staging.

For unused lanes still evaluated by GEMM, a non-NaN contract is insufficient:
zero times infinity is NaN too. Reusing such lanes requires finiteness in the
consumer's interpretation, or a proof the lanes cannot affect an observed
result. Finite FP32/FP8 storage need not be finite when reinterpreted as FP16.
This change uses definite writes, not a global assumption about numerical ranges
or an allocation-history analysis. Producers may still overflow or evaluate
undefined arithmetic outside their normal input domain.

Final measurements with range merging:

| Workload | Previous | Selective clears | Counter |
| --- | ---: | ---: | --- |
| Automatic full MLP | 222,408 | 214,890 | maximum tile cycles |
| Historical-grid full MLP | 221,124 | 211,572 | maximum tile cycles |
| Automatic full MLP | 221,742 | 214,224 | cropped profile span |
| Historical-grid full MLP | 214,332 | 204,780 | cropped profile span |
| Projected attention | 523,980 | 523,848 | cropped profile span |

Both MLP results retain maximum error 0.011719. The historical-grid cropped span
is now close to the old saved profile's 205,392 cycles. Attention's final fill
sample count is 1,920, with aggregate fill work falling from 923,232 to 600,000
tile-cycles; its total execution time is essentially unchanged. Projected
attention passes all 839,808 numerical checks at maximum error 0.001230.

Rendered profiles are `artifacts/profiles/mlp-selective-clears.html`,
`artifacts/profiles/mlp-historical-selective-clears.html`, and
`artifacts/profiles/attention-selective-clears.html`. The historical MLP renderer
was checked in headless Chromium. All 161 workspace release tests pass; Clippy
passes with the documented `too_many_arguments` and `type_complexity` allowances.
The unqualified Clippy command still reports those existing lint violations.
GEMM, batched GEMM, attention smoke, and repeated two-block MLP hardware checks
also pass. Logs: `/tmp/selective-final-{tests,clippy}.log`,
`/tmp/{automatic,historical}-selective-merged.log`, and
`/tmp/merged-validated-*.log`.
Forced materialized attention also passes all 839,808 checks, maximum error
0.000930, after the final range-merging change.

## Finite numerical contract and reset investigation (2026-09-06)

Numerical inputs and operation results are assumed finite under the normal input
contract. Low scheduling may therefore omit copy-created K-padding clears when
all tensor storage is F16, every logical destination element is written, and
all consumers use that storage as GEMM activations against zero-padded parameter
coefficients. Parameter provenance follows copies and aliases. Other consumers,
missing logical data, mixed-precision arenas and repeat bindings without an
established parameter proof retain initialization. Discarded row padding remains
zero: arbitrary finite row values could cause overflow despite unobserved outputs.
The runtime enables `FP_ICTL.OFLO`; worker activations inherit it. This is distinct
from `NANOO`, which controls NaN-on-overflow behavior.

The invariant starts at device reset/loading and survives same-precision SRAM
reuse. It does not follow merely from starting a second invocation: a previous
FP32 writer could leave FP16 NaN bit patterns. This first implementation deliberately
uses the stronger all-F16-arena proof instead of allocation-history dataflow across
mixed precisions.

`gc-reset -m` was traced with GDB and compared with the driver. The SDK generates
a short register-initialization program and uses the autoloader to install it
while initializing tile SRAM. The driver's secondary-bootloader installation
already uses the corresponding two-stage SRAM initialization sequence. It is now
factored into one helper, also exposed as `Device::reset_tile_memory`. The API
requires device reset and configuration first; invoking just the autoloader
sequence against debug-halted live state does not establish the same guarantee.

A destructive hardware regression poisons five addresses on tiles 0, 1, 63, 735,
and 1471 with `0x7e007e00`, verifies the poison, resets/reconfigures the device,
clears SRAM, and checks every address again. The addresses cover the bootstrap,
standard SRAM, interleaved SRAM and the final SRAM word. All 25 checks pass; the
autoloader portion takes about 66.9 ms. TDI inspection follows the SDK's all-seven-
contexts halt and ATOV protocol. No TDI instruction encoding or primary/secondary
register-access change was needed. The reproducible check is
`crates/ipu-driver/examples/reset_memory.rs`; its arguments are configuration,
an idle package and the SDK secondary-bootloader ELF.

The final conservative pass removes 72 clear calls from the historical MLP and
36 from the automatic MLP. Maximum tile cycles remain 211572 and 214890,
respectively. Most merged clear ranges cover both K padding and discarded rows,
so the finite-value proof cannot remove the whole call. An earlier prototype
removed all 6528 historical clears and ran in 207786 cycles, but did not preserve
discarded-row zeros; that optimization was narrowed to avoid spurious overflow
faults. Recovering that saving requires separating row initialization from
K-padding initialization rather than dropping their combined clears.

All 163 workspace release tests and Clippy (with the documented allowances) pass.
Hardware validation with overflow faults enabled passes full automatic and
historical MLPs, GEMM, batched GEMM, attention smoke, a repeated two-block MLP,
and both full projected and materialized attention. The latter check all 839808
outputs, with maximum errors 0.001230 and 0.000930, respectively. A deliberate
overflow-fault injection has not been tested. Structured-repeat and mixed-
precision programs conservatively retain their padding clears.

## Finalist expansion versus scheduling

Compact exchange cost takes the larger of maximum outgoing bus bytes and maximum
incoming tile bytes, divides by four bytes per cycle, and adds 600 cycles per
nonempty exchange phase. Geometry-based traffic includes replication, grouping
and padding. Fragmentation also influences staging choices and exchange-table
storage estimates; packed-linear movement uses representative native grains.
This does not model physical SRAM bank conflicts, detailed routes, paired-send
opportunities or the final send/receive schedule.

The manual `profile_mlp_finalist_expansion` test retains eight full-size MLP
finalists without scheduling their exchanges. Compact planning took 12.22 s;
individual expansions took 1.87, 2.54, 2.36, 2.78, 3.70, 2.35, 2.49 and 2.72 s.
Their expanded analytical totals were 282747, 287715, 288507, 281275, 253837,
296064, 288265 and 298810 cycles. Thus another compact finalist can look much
better after expansion, and expansion itself is much cheaper than scheduling.

The existing `--exchange-schedule-finalists 2` does full placement and physical
exchange scheduling for both candidates. That selection stage took 47.60 s,
considerably longer than the individual expansions above. It retained two modern
geometries, not the historical combination; refined estimates were 267829 and
269754 cycles, so finalist zero still won. Hardware passed at 214884 cycles.
A sensible next step is cheap expanded-analytical reranking before scheduling a
smaller subset, rather than physically scheduling every retained candidate.
These measurements do not change the default finalist count.

Logs: `/tmp/reset-{all-writes,mailbox,memory-check}.log`,
`/tmp/finalist-expansion-timing.log`, and `/tmp/mlp-two-finalists.log`.

## Exact historical-plan discard points (2026-09-06)

Instrumenting `retain_operator_candidates` and every beam-pruning boundary with
both complete historical GEMM constraints locates the loss before beam search.
The trace uses the ordinary automatic full-size MLP configuration, including its
shape-aware tile counts and 64-plan shortlist. Ranks below are one-based.

* First GEMM, `4x92x4`, result `4x1`, 48 output columns, interleaved weights:
  it survives its local orientation/tile-count screen, ranked 12th out of 548
  Pareto candidates (1628 variants before Pareto filtering). The pooled screen
  in `lower_operation_candidates` receives 1503 candidates, including this plan.
  It remains Pareto-undominated, but ranks 82nd of 359 and loses the 64-slot
  selection. Its compact implementation costs 100947 cycles.
* The diversity representative taking its slot is `1x269x4`, result `4x1`,
  16 output columns, priced at 92287 cycles. Diversity groups include orientation,
  reduction fan-in, result partitions and output format, but exclude compute
  row/column counts and input memory class. This representative already reports
  a 425984-byte standard allocation and 66640-byte contiguous-memory overflow.
  Region memory validation subsequently rejects it with the same overflow.
  At that point the historical alternative has already disappeared.
* Second GEMM, `4x24x15`, result `15x1`, 48 output columns, interleaved weights:
  its exact variant is discarded in the local candidate-family screen. It ranks
  54th, at 106963 cycles. The immediately preceding candidate has identical
  geometry with standard-SRAM weights and exactly the same estimated cycles,
  total memory, peak interleaved memory and exchange-row bytes. It receives the
  shared diversity slot; all 64 slots fill before the interleaved variant can
  enter through the cost-ranked remainder. This occurs in both generated families
  containing that historical variant.

Thus the historical pair is never presented to complete-program finalist ranking;
raising only `exchange_schedule_finalists` cannot recover it. The existing
`shortlist_prices_execution_instead_of_boundary_storage` regression compares two
forced plans, so it does not cover these pooled/diversity losses.

A temporary feasibility-first sort confirms the first issue but is insufficient
on its own: it preserves a standard-weight version of the first historical grid,
while the interleaved version still loses an equal-cost diversity tie. That
experiment was reverted after measurement. The production ranking is unchanged.
The next costing work should account for known memory infeasibility before
spending diversity slots, and distinguish movement/placement alternatives that
currently receive identical coarse prices. Conflict reduction remains relevant
after selection; exact physical scheduling of more finalists cannot repair an
operator plan discarded at these earlier screens.

Trace logs are `/tmp/historical-search-{final-trace,memory,feasible}.log`.
The manual finalist-expansion benchmark now prints every finalist's GEMM dispatch
and weight memory class, so its timing output also identifies the layouts tested.
The labeled rerun identifies finalist 4 (expanded estimate 253837 cycles) as
`3x162x3` followed by `5x24x12`, both with standard-SRAM weights, rather than the
historical pair. It remains an untested hardware candidate. The run passes and
reproduces all eight earlier expanded estimates; output is in
`/tmp/historical-finalist-geometries.log`.

## Phase-specific gaps and route-pipelining probe (2026-09-06)

The endpoint model should be shared movement analysis over tensor ownership,
logical regions and storage order, not GEMM-specific estimation. Copies/views
supply redistribution; reductions supply contributor-to-result gathers. Their
traffic can enter the same per-phase endpoint and fragment accounting.

The fine-grained exchange intervals in the renderer describe generated schedule
activity, positioned within measured exchange samples. They are not independent
hardware measurements of every individual transfer. Extracting these intervals
from `artifacts/profiles/mlp-selective-clears.html` gives:

| Physical phase | Role | Send intervals | Receive intervals | Median receive payload cycles | Median gap between receives | Scheduled horizon |
|---|---|---:|---:|---:|---:|---:|
| 0 | First GEMM input distribution | 807 | 1614 | 3072 | 93 | 6381 |
| 1 | First GEMM reduction | 6456 | 6456 | 640 | 82 | 3635 |
| 2 | Redistribution before GEMM 2 | 3931 | 55116 | 320 | 86 | 29240 |
| 3 | Final reduction | 155880 | 155880 | 64 | 81 | 25866 |

These are interval counts, including encoded transfer chunks. Phase 2 is heavily
multicast; phase 3 is unicast and much more fragmented. The final reduction has
27 independent partials in the automatic layout, versus 15 historically. Different
partial values cannot be replaced by broadcasting a single value, although their
partitioning and physical packing can change how many fragments are needed.
Low reduction expansion explicitly creates local seed copies into packed reduction
storage and result copies into output views. Those explain some `copy_u64` work
adjacent to exchanges; a local copy is not mandatory exchange teardown. Some
terminal copies currently lack operation provenance in profile metadata.

Barrier arrival spread is separate from exchange scheduling: phase 1 has 32064
cycles of spread after the unbalanced first GEMM, phase 2 has 16428 after reduction,
GeLU and staging, and phase 3 has 12618 after GEMM 2. Moving a transfer inside an
exchange cannot eliminate the preceding compute tail.

A specific outer-scheduler constraint may unnecessarily serialize route latency.
`TransferScheduler::next` returns an endpoint-availability maximum as if it were
only a data-dependency release time. `MaterializedSchedule::append` also starts
from the maximum receiver payload completion. But `PhaseProgramBuilder` already
checks source-selection timing, payload-arrival timing, receive pointers, encoding
and SRAM hazards at their distinct offsets. Starting a new schedule only after
the previous receive completes can leave a route-latency bubble.

A temporary two-line probe passed only actual dependency readiness to the row
builder, leaving its endpoint/encoding checks and memory-hazard checks enabled:

| Saved replay | Baseline initial / optimized horizon | Probe initial / optimized horizon |
|---|---:|---:|
| Historical-grid current-address final reduction, 63720 unicast transfers | 12527 / 12527 | 7991 / 7991 |
| Current-address paired redistribution, 5562 transfers, 118812 destinations | 48852 / 41599 | 42557 / 42557 |

Both probes pass the existing offline schedule validator. The reduction improves
by 36.2%, while the redistribution's optimized result gets 2.3% worse despite a
better initial schedule. Thus removing the conservative release constraint is a
real candidate improvement, but ordering and critical-predecessor accounting must
be adapted too. The probe has not been hardware-tested and was reverted; production
scheduling is unchanged. The source patch and probe executable are retained at
`/tmp/exchange-route-pipeline-probe.patch` and `/tmp/exchange-bench-gap-probe`.
Logs: `/tmp/gap-probe-{baseline,pipelined,reduce-baseline,reduce-pipelined}.log`.

This complements the earlier address-shift experiment: fragmentation, avoidable
route bubbles, placement-dependent SRAM conflicts and pre-exchange compute tails
are distinct mechanisms. A universal per-transfer switchover charge or occupancy
penalty would conflate them. Useful profile annotations would name the actual
blocking endpoint, memory hazard, dependency or encoding restriction behind gaps.
