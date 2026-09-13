# Patching in the BS1 early-cast misranking

Audited the emitted packages, not just the planner's transfer estimates:

- Before: `artifacts/producer-fp8-20260913/pv-early-full/model.ipuexe`
- Early downprojection cast: `artifacts/fp8-pv-batches-20260913/bs1-step64/model.ipuexe`
- Extraction script, full call/descriptor lists, and summary:
  `artifacts/patch-ranking-20260913/`

Patch calls were associated with exchange phases through their emitted cycle
sample addresses. Arithmetic descriptors identify the actual instruction word
being patched, its initial value, and its step between Repeat iterations.
Source addresses were checked against exact per-tile memory profiles.

## What changes

Counts below are across tiles for one transformer layer. Patch lists are reused
on each of the 27 iterations, including writing the initial values on iteration
zero.

| Metric | Before | Early cast |
|---|---:|---:|
| Repeat address words patched per layer | 6,153 | 11,506 |
| Cross-phase sharing words patched per layer | 14 | 14 |
| Maximum Repeat patch list on one tile/phase | 48 | 87 |
| Maximum cross-phase sharing patch list | 1 | 1 |
| Downprojection phase 27 tiles using OUTGOING_BASE | 1,440 | 711 |
| Downprojection phase 27 Repeat patches | 0 | 5,353 |
| Tiles performing those new patches | 0 | 729 |

The additional 5,353 patches are arithmetic progressions through the layer's
FP8 downprojection weights. That is 144,531 additional word writes over the
27 layers, distributed over the tiles. All 729 affected tiles use zero outgoing
base in this phase; the existing relocation mechanism still operates on the
other 711 tiles. Cross-phase row sharing is not responsible for this regression.

The upprojection phase 23 already has the same kind of problem in both builds:
5,384 patched words across 729 tiles, with a maximum list of 48.

## Why

The early-cast preparation combines fixed-address activation sends with moving
layer-weight sends in the same timed exchange row. A single moving base would
then require inverse relocation of the activation sends. In the affected
phase-27 rows there are 102,093 stationary address-bearing words versus 5,353
moving ones. Every affected tile has more stationary words than moving words;
all also have stationary sources below the initial weight address. Thus the
current whole-row base selection rejects weight relocation both because it
would increase patch work and because its full-source-address base cannot
represent those lower stationary addresses.

The patches are not one per high-level transfer. The 5,353 patched words belong
to 1,568 weight messages: SENDPICP combines two receive-control updates with an outgoing
send and explicitly encodes the source address again. These fields must be
updated too. This is distinct from paired-tile transfer mode. On the worst physical tile, 1180, just two
weight messages account for 87 patched words (64 and 23). Its weight source is
621,848–625,432; the 135 activation messages use much lower scratch addresses
and account for another 179 address-bearing words.

## Time and reduction opportunities

Repeat arithmetic patching uses a serial supervisor loop. The existing bulk
worker path is only in the separate cross-phase row-sharing helper. The measured
phase-27 time beyond its scheduled event horizon rises from 295 to 2,786 cycles.
This 2,491-cycle increase is consistent with the newly long patch work, although
profile entry precedes setup and overlap with other tiles prevents assigning
all of it to patch execution from these timestamps alone.

A smaller implementation change could parallelize long Repeat arithmetic lists,
using the same approach already used for long sharing lists. That speeds patches
but does not reduce their number.

To remove patches, use different outgoing bases for static and moving send
segments within the timed row. Source ordering is encouraging: 467 of the 729
affected rows have exactly one transition between those two source classes;
the remaining rows have two to six transitions. Every tile's moving patches
have one uniform displacement. However, this needs explicit scheduling of base
changes, safe handling of outgoing stream restarts, and hardware validation. It
is not safe simply to inject PUT instructions into the existing schedule.

Keeping those transfers in separate globally synchronized phases would also
restore whole-phase base relocation, but sacrifices exchange overlap and adds a
barrier. The better comparison is a complete sequence including setup, not just
a patch count or transfer horizon.
