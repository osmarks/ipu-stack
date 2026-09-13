# Full-model FP8 QK evaluation

Artifacts and executable build scripts are under
`artifacts/full-qk-20260913/`. The production default is unchanged.

## Controlled setup

Full 27-layer SigLIP, 378x378 input, B1024 exchanges, resident parameters.
Only encoder QK switches from FP16 to F143 at scale -4 with FP16 accumulation;
encoder PV remains FP8 and MAP QK stays FP16. The starting recipes preserve
GEMM grids and external tensor layouts. `prepare.py` edits the explicit
encoder attention recipe and its saved alternatives; it does not weaken the
compiler's checkpoint context validation.

Both batch sizes run eight additional ordinary local-search steps. Independent
cast sites on encoder operation 12 are V=0, Q=1, K=2. Additional fixed-recipe
Q/K combinations check the interaction independently of the search shortlist.
Profiles instrument the first repeated layer and use the renderer's cropped
execution interval.

Controls:

- BS1: `artifacts/mid-cast-order-20260913/bs1-final/`, 10,131,456 cycles.
- BS2: `artifacts/full-qk-20260913/bs2-baseline/`, refreshed with current cast
  motion, 16,746,138 cycles. The older 16,985,046-cycle result is not the
  matched control.

## Real inputs and calibration

Real-image checks use the pinned original SigLIP fixture and existing
`role-shared.json` projection/MLP scales, including FP16 image projection.
Their layouts differ from the randomized performance recipes; their execution
time must not be attributed to those recipes.

`qk_ranges.py` measures Q/K outputs after calibrated projection quantization
on the original three calibration images. The largest encoder value is
14.12747, giving shared encoder QK exponent -4 (finite range +/-15).
The MAP key maximum is 23.039, another reason not to apply this exponent to
MAP QK. The other three images remain held-out hardware checks.

All six BS1 images pass with both Q and K cast early:

| Image | FP32-reference cosine |
|---|---:|
| authors | 0.994567066 |
| siglip | 0.996543687 |
| caffeine | 0.997382663 |
| robosign (held out) | 0.995361503 |
| fried_fish (held out) | 0.996956592 |
| cow_beach2 (held out) | 0.997177897 |

The same parameters remain resident for all six invocations. `real-bs1/`
retains the package, state, log, fixture manifest, and raw hardware output.
`batch_fixture.py` pairs the six inputs into three BS2 cases, concatenates the
independent FP32 reference embeddings, and replicates the learned MAP probe.
The runtime verifier checks the minimum cosine across individual embeddings,
not one cosine over a concatenated batch. Parameter files are shared by path.

## Hardware performance

Controlled FP8-QK cast choices retain all other decisions from the matched
FP16-QK control. Both outputs and parameters survive a second resident
inference. All times below include the complete 27-layer model and MAP head.

| QK implementation / cast placement | BS1 cycles | BS2 cycles |
|---|---:|---:|
| FP16 QK control | 10,131,456 | 16,746,138 |
| FP8 QK, early Q only | 10,142,796 | pending |
| FP8 QK, early K only | 10,462,992 | pending |
| FP8 QK, early Q and K | 10,254,720 | 16,600,566 |
| FP8 QK, eight-step search | 10,099,464 | pending |

The BS1 search selected early Q but retained late K. It also selected early
casting at the MLP up-projection (operation 18), so the 0.32% improvement over
the initial control is not attributable solely to QK. A separate FP16-QK
replay with that MLP decision takes 10,088,124 cycles. FP8 QK is therefore
0.11% slower with either MLP cast choice; the search gain came from the MLP,
not QK quantization.

In the controlled BS1 early-Q profile, the largest QK kernel falls from
12,402 to 7,674 cycles (38.1% less). Its FP8 inner panel is padded to 96 rather
than FP16's 80. Despite that padding, arithmetic is much faster. Casting and
preparation offset the saving: the complete model is 0.11% slower. Kernel
phase totals overlap and must not be summed to infer wall time.

The default precision policy is unchanged. These results do not justify
unconditionally enabling FP8 QK from its standalone kernel speed.

## BS2 real-weight limitation

The calibrated BS2 control, with QK still FP16, does not fit. At a 96 KiB
exchange budget it needs 103,396 bytes on the worst tile; with 112 KiB allowed,
placement fails for a 2,048-byte standard allocation on tile 740. This is a
failure to place the real-weight baseline, not a failed cosine check. No BS2
real-weight accuracy result is claimed. Both logs are retained in
`real-bs2-baseline/` (`budget96.log` and `build.log`).

## Artifacts

- `bs1-search/model.html` and its adjacent data directory: searched BS1 profile.
- `bs2-both/model.html`: controlled early-Q-and-K BS2 profile.
- `bs1-q/attention.json` and `bs1-baseline-attention.json`: kernel comparison.
- `results.json`: hardware status, numerical checks, cycles, and saved decisions.
- Each build directory retains its build script, compiler log, package and
  search state; performance builds also retain profile input and extraction.

