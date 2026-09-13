# Full-model FP8 QK evaluation

Work in progress. Artifacts and executable build scripts are under
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

Remaining at this checkpoint: finish both searches and controlled cast-order
runs, collect/render their profiles, and complete calibrated BS2 validation.
