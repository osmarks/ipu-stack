# Exchange footprint model

Expanded-finalist selection now estimates encoded row storage from endpoint
geometry before placement or scheduling. The old estimate charged a full
nine-word primitive row for every TX/RX fragment. Scheduled rows share controls
and omit much of that primitive setup, so that estimate was a poor byte price.

The new ordinary-transfer estimate charges:

- 8 bytes per send for address setup and sending, plus 4 for a payload over 64 words;
- 8 bytes per receive for source/neutral controls;
- 4 bytes when the receive address differs from the preceding receive's end;
- 8 bytes per tile/phase for entry and return, rounded to eight-byte alignment.

Multicast transmission is counted once at the source. Before placement, receive
addresses are allocation ID plus byte offset, so separate allocations cannot
accidentally appear contiguous. Transfer lengths are split at the ISA limit.
Phase estimates accumulate on each tile before taking the maximum; Repeat bodies
contribute their stored program once. The existing geometry traversal supplies
these features without a second span expansion. The coarse mid beam estimate
has not changed: this refinement applies once concrete span geometry is available.

This is a ranking model, not an upper or lower bound. Capture order can differ
from scheduled order; scheduling changes pointer reuse, delays, bidirectional
instruction combinations and paired modes. Cross-phase normalized-row sharing
is not predicted. The public capture estimator treats transfers as ordinary
Word32 equivalents, matching the pre-width-selection geometry model.

## Validation

Five large ordinary-transfer stages were compared with retained encoded results
in `artifacts/exchange-redesign-20260908/results.json`. These were not rescheduled.
B2 uses `vit-b2-f0.json`; B4 uses `vit-b4-f7.json` (phase IDs are not interchangeable
with those in `vit-b4-f2.json`). No empirical regression coefficients were fitted.

| Batch / phase | Old fragment-slot estimate | New estimate | Encoded bytes | Error | Estimate ms | Recorded scheduling ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 / 17 | 34,312 | 10,560 | 12,244 | -13.8% | 3.621 | 51,673 |
| B2 / 31 | 27,184 | 6,800 | 5,788 | +17.5% | 6.766 | 31,128 |
| B2 / 34 | 43,420 | 9,672 | 8,228 | +17.5% | 9.937 | 33,373 |
| B4 / 16 | 37,768 | 11,704 | 12,828 | -8.8% | 7.439 | 108,724 |
| B4 / 28 | 54,328 | 13,592 | 11,644 | +16.7% | 13.945 | 103,217 |

Old estimates in this table use captured endpoint counts, isolating the pricing
change from earlier logical-span fragmentation/coalescing differences. Times for
the new model exclude snapshot parsing, validation and geometry generation.

Three additional smaller B2 stages were then scheduled and validated with the
current scheduler, without changing the model:

| Phase | Estimated bytes | Encoded bytes | Error |
| --- | ---: | ---: | ---: |
| 0 | 32 | 36 | -11.1% |
| 1 | 136 | 152 | -10.5% |
| 2 | 664 | 648 | +2.5% |

The 64-phase B2 snapshot takes 57.8 ms for all footprint estimates and predicts
75,456 bytes on its busiest tile. The B4 snapshot predicts 96,368 bytes and takes
73.7 ms. These full-table estimates have not been compared with compact packages
for those exact snapshots; phase agreement does not establish whole-table
accuracy or current B2 build feasibility.

Results are under `artifacts/exchange-footprint/`. Reproduce estimates with:

```sh
target/release/ipu-exchange-schedule-bench \
  artifacts/exchange-redesign-20260908/vit-b2-f0.json --footprint-only
```

Omit `--footprint-only` and select `--phase 0 --phase 1 --phase 2` to reproduce the
held-out scheduling checks. The benchmark also reports estimates in ordinary
scheduling runs.

## Admission

Expanded candidates predicted within `exchange_table_budget_bytes` are ranked
before candidates predicted over it. Within budget, execution score (including
the configured storage penalty) orders candidates. If all are over budget, the
smallest predicted excess wins. The bounded admission shortlist retains the
minimum estimated storage alternative rather than the minimum raw fragment
count. An estimate alone never returns an infeasibility error: when all estimates
are too large, the smallest candidate can still be scheduled and checked exactly.
The separate hard fragment limit remains an explicit scheduling-effort safeguard.

Tests cover pointer continuity versus distinct allocations, per-tile phase
accumulation, Repeat storage reuse, priority for predicted-fit candidates, and
retaining an attempt when every estimate exceeds the budget. Full ViT builds
were not repeated for this change.
