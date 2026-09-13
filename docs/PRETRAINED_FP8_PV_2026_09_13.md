# Pretrained SigLIP with fused FP8 PV

All six images pass the full 27-layer vision tower and MAP head on hardware
against the original pretrained FP32 Hugging Face model. These checks use the
existing calibrated GEMM policy and FP16 input projection, not the uniform −4
policy used for randomized performance fixtures.

The pinned checkpoint, original parameters, preprocessing, calibration/held-out
split and independent reference are described in
[the original validation](SIGLIP_PRETRAINED_VALIDATION_2026_09_12.md).
No new calibration or weight reconstruction was performed for these images.
Weights remain resident across the six distinct image invocations.

| Image | FP16 PV cosine | Fused FP8 PV cosine |
|---|---:|---:|
| authors | 0.996796693 | 0.996772244 |
| siglip | 0.997325027 | 0.997721873 |
| caffeine | 0.997113995 | 0.997597634 |
| robosign (held out) | 0.996104172 | 0.996237005 |
| fried_fish (held out) | 0.997151224 | 0.996848738 |
| cow_beach2 (held out) | 0.997084336 | 0.997553684 |

Maximum absolute embedding error is 0.248456 for FP16 PV and 0.303112 for FP8
PV. Every cosine exceeds 0.99. A higher cosine for some quantized outputs does
not establish improved model accuracy; this is a small numerical check.

Both builds use the normal baseline, zero local-search steps, B1024 exchanges
and no profiling. The FP8 build enables PV scale −4 in both encoder and MAP
attention; QK stays FP16. The planner selects encoder PV grid 7×2×6 for FP8
versus 13×1×7 for FP16. These calibrated layouts differ from the saved randomized
7.111 ms performance recipe; that timing must not be attributed to this test.

Artifacts are under `artifacts/pretrained-pv-20260913/`:

- `calibrated-f16/` and `calibrated-pv/`: successful normal-baseline packages.
- Matching `.sh`, `.log`, and `-state.json` files reproduce each run.
- `results.json`: per-image cosines and maximum errors.
- `capacity-f16*` and `capacity-pv*`: earlier capacity-baseline checks, also
  passing all six images, retained separately.
- `*-b1024.log` and `*-b256.log`: original normal-baseline compiler failures.

## Self-multicast packetization bug

The normal baselines initially failed with `unencodable initial send control`.
A concrete case was a 28-word self-multicast from tile 745. Its self-receive
neutral control occurred one event after the sender's payload start. The first
SEND must establish the outgoing source; SENDPIC can only continue an already
established stream. Moving the whole transfer later preserves the collision.
The offset search considers existing tile activity, but cannot remove a
conflict between the two roles of the same new transfer.

Packet formation now checks the primitive's actual receive-control timing
against its send start. Intrinsically conflicting self-receive packets are
split into smaller contiguous packets and checked again. The concrete 28-word
case becomes two 14-word packets. This preserves all receivers, every Repeat
source address, source offsets, SRAM access ranges and original byte order.
It introduces no new exchange instruction or unverified timing relaxation.
Other packet sizes are retained. Ordinary/paired width selection still rejects
paired alternatives whose primitive controls cannot be encoded.

This runs before cache fingerprinting, and is shared with explicit schedule
replay and validation. Tests cover four source locations, lengths 1–64,
ordinary and valid paired primitives, Repeat address preservation and cache
replay after address relocation. The exchange library suite passes 52 tests
(two ignored); compiler exchange tests pass 29 (two ignored), including the
new packet test. Its additional cache assertions also pass separately.

Exact hardware readback passes independently of model cosine tolerance:

- `loopback28.json` / `.log`: the 28-word ordinary regression, 196 checked words.
- `loopback-paired29.json` / `.log`: 29 paired items, 464 checked words.

After the fix, both previously failing normal-baseline builds complete and
produce the real-image results above. Temporary diagnostic instrumentation
was removed.
