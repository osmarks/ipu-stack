# MLP reduction epilogue investigation

Source: `artifacts/mid-cast-order-20260913/bs1-final/model.capnp` and its saved
27-layer BS1 recipe. Queries are saved in `artifacts/reduction-epilogue-20260913/`.
This is a feasibility/profile investigation, not a measured fused-kernel result.

## Actual data flow

The device has 1,472 compute tiles. This plan uses 1,458 for both up-projection
reduction and bias-GeLU. There is no need to gather the epilogue onto a small
set of reduction roots.

The up-projection grid is 9 row partitions, 27 column partitions, and 6 K
partitions. Its result divides each column partition among six owners, giving
9 x 162 output owners. The final reduction is a single stage over six FP16
partials, in AMP-left physical order. Native output shards hold 79–82 rows of
16 or 32 columns. The profile has exactly one reduction call per active tile:

| Physical elements | Cycles | Tiles |
|---:|---:|---:|
| 2624 | 7218 | 535 |
| 2560 | 7092 | 321 |
| 2528 | 7092 | 107 |
| 1312 | 3816 | 275 |
| 1280 | 3690 | 165 |
| 1264 | 3690 | 55 |

Their element counts sum to 3,137,616 = 729 x 4304: the reduction does not do
extra padded-element work in this case.

The result is then redistributed into row-major ownership: 729 rows x two
column partitions. Each of the 1,458 tiles runs one row of 2,152-element
`bias_gelu_f16`, followed by a separate FP8 cast. Bias addition is already fused
with GeLU. The cast-order work did not fuse this encoder GeLU's FP8 output;
the profile's `bias_gelu_f8` belongs to the one-time MAP head instead.

## Timings and overlap

| Work | Longest tile call | First start–last end |
|---|---:|---:|
| Up-projection final reduction | 7,218 | 304,512–320,436 |
| Add preparation exchange | 15,246 | 308,202–323,526 |
| Bias-GeLU | 8,556 | 323,346–332,424 |
| Down-projection input cast | 5,496 | 331,902–337,920 |

These intervals overlap; their durations cannot be added to predict savings.
In particular, removing the add-preparation exchange does not save its entire
15k-cycle duration from the critical path. The native reduction shards also
have more uneven work sizes than the GeLU shards, so fusion inherits that tail.

## What fusion can remove

- The activation redistribution from the native reduction layout to the
  wide-row GeLU layout: 6,275,232 logical FP16 destination bytes per layer.
- Most bias replication. Current bias operands occupy 2,152 FP16 elements on
  every active tile: also 6,275,232 bytes. Native epilogues need only one 16/32
  element column slice per tile, reused across its rows: 9 x 4304 x 2 = 77,472
  bytes. This is an 81x reduction in destination storage/payload, not an 81x
  reduction in sender instructions—multicast already shares transmissions.
- The reduced FP16 intermediate's store/reload between reduction and GeLU,
  and separate worker setup for the epilogue.
- With native FP8 output, another intermediate and the standalone cast; the
  subsequent down-GEMM redistribution still remains.

The exchange gathering the six partial sums is necessary and remains.

## Kernel and representation constraints

Calling the existing row-oriented BiasGeLU kernel on native 82x32 shards is
not a good substitute. Its current model predicts 33,678 cycles there, because
it repartitions workers and resets row traversal on every short row. A fused
kernel should stream physical panels, reusing the bias instead.

FP16 AMP-left stores columns in 16-column panels, then rows, then columns
within the panel. Bias addressing is therefore a repeated 16-element pattern.
The existing reducer processes eight elements per worker iteration with a
48-element worker stride; within one panel, the bias offset stays constant.
Panel transitions need explicit handling, but no global row-major conversion.

Register pressure needs attention: the reducer holds two four-half sums in
`a0:3`, while MIX BiasGeLU uses all eight ARF registers. Its polynomial cannot
simply be pasted before both stores. Options are processing one quad at a time,
or retaining the second quad in main registers while processing the first.
The existing MIX/TAS polynomial can be shared, rather than introducing another
numerical approximation. Only the final reduction stage may apply the epilogue;
applying GeLU to independent partials or intermediate stages is incorrect.

FP8 AMP-left output groups 32 columns rather than 16. Adjacent FP16 panels need
interleaving in the output addresses, and 16-column owners need bounded padding
or compatible redistribution. Merely changing the reducer output precision
would be wrong. F16 output is the simpler first kernel comparison; FP8 output
is the fuller opportunity.

At mid level, retain `Sum` and attach a supported final-stage epilogue through
the common producer-fusion machinery. Match single-use Sum -> copies -> BiasGeLU
(and an optional cast), bring the bias to the selected reduction ownership,
and price the entire replacement. Keep the existing separate path. Low should
only realize that choice and dispatch the epilogue on the last stage, without
recognizing an MLP or independently deciding to fuse it.

Conclusion: a plausible worthwhile optimization, especially for traffic and
scratch. It requires a native-panel epilogue, not a missing switch for the
existing BiasGeLU kernel. A whole-sequence hardware comparison is still needed
to establish the speedup; the profile does not justify promising removal of an
entire exchange phase's runtime.
