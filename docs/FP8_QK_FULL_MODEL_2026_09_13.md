# Full-model FP8 QK evaluation

Artifacts and executable build scripts are under
`artifacts/full-qk-20260913/`. The production default is unchanged.
With matched MLP decisions, FP8 QK is
0.11% slower for BS1 and 0.85% faster for BS2. Real-weight/image checks pass
for both batch sizes against the original FP32 model.

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

The package-planning portions took 559.925 seconds for BS1 and 2,071.589
seconds for BS2 with 16 Rayon threads. The eight-step BS2 budget dispatched
batches of 8, 7, 6 and 3 candidates: the limit counts ordered attempts through
each accepted winner, not all speculative work already dispatched in parallel.
These are observed concurrent-run times, not isolated compiler benchmarks.

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
FP16-QK control. Each package passes two inferences with parameters remaining
resident. All times below include the complete 27-layer model and MAP head.

| QK implementation / cast placement | BS1 cycles | BS2 cycles |
|---|---:|---:|
| FP16 QK control | 10,131,456 | 16,746,138 |
| FP8 QK, both casts late | 10,303,668 | 16,880,580 |
| FP8 QK, early Q only | 10,142,796 | 16,722,798 |
| FP8 QK, early K only | 10,462,992 | 16,766,628 |
| FP8 QK, early Q and K | 10,254,720 | 16,600,566 |
| FP8 QK, eight-step search | 10,099,464 | 16,065,444 |

The BS1 search selected early Q but retained late K. It also selected early
casting at the MLP up-projection (operation 18), so the 0.32% improvement over
the initial control is not attributable solely to QK. A separate FP16-QK
replay with that MLP decision takes 10,088,124 cycles. FP8 QK is therefore
0.11% slower with either MLP cast choice; the search gain came from the MLP,
not QK quantization.

The BS2 search first changed the MLP down-projection (operation 21) from a
20x6x12 compute grid with a 1x12 result grid to 8x8x23 with a 23x1 result
grid, opening output boundary 361. A matched FP16-QK replay takes 16,202,412
cycles: 3.25% faster than the starting control. The search then selected early
K followed by early Q. The final recipe differs from the MLP control only in
encoder QK precision and those two cast choices; the attention grids and all
other plans remain identical. The final hardware run takes 16,065,444 cycles
(10.710296 ms): 0.85% faster than the matched 16,202,412-cycle FP16-QK
control, and 4.06% faster than the original 16,746,138-cycle starting plan.

The best BS1 result from this experiment retains FP16 QK and uses the earlier
MLP cast: 10,088,124 cycles (6.725416 ms), 0.43% faster than the starting plan.
All four fixed FP8-QK cast placements produce identical randomized-reference
cosines within each batch size (BS1 0.994179673; BS2 0.993945443), as expected
when the cast moves without changing its arithmetic. The final searched
BS1/BS2 recipes reach 0.994366413 and 0.994154827 respectively; their MLP
changes alter rounding. These randomized checks are separate from the
pretrained checks.

In the controlled BS1 early-Q profile, the largest QK kernel falls from
12,402 to 7,674 cycles (38.1% less). Its FP8 inner panel is padded to 96 rather
than FP16's 80. Despite that padding, arithmetic is much faster. Casting and
preparation offset the saving: the complete model is 0.11% slower. Kernel
phase totals overlap and must not be summed to infer wall time.

For BS2 the corresponding maximum QK kernel drops from 22,998 to 14,058
cycles. Even before preparation, the kernel saving multiplied by 27 layers
is only about 1.4% of the complete baseline runtime (about 1.3% for BS1).
This is an approximate arithmetic contribution, not a prediction of the
whole preparation/exchange/compute sequence.

The default precision policy is unchanged. These results do not justify
unconditionally enabling FP8 QK from its standalone kernel speed.

## BS2 real-weight validation

All six images also pass in three batch-two invocations with resident
parameters, encoder FP8 QK/PV, both Q/K casts early, and FP16 MAP QK.
The verifier takes the minimum cosine of the two individual embeddings.

| Image pair | Minimum FP32-reference cosine |
|---|---:|
| authors + siglip | 0.996363787 |
| caffeine + robosign | 0.994959132 |
| fried_fish + cow_beach2 | 0.996347857 |

Maximum absolute error across these checks is 0.392955. As with BS1, the
pretrained test uses calibrated GEMM scales and FP16 input projection; it is
not the randomized performance package.

The fresh calibrated BS2 control, with QK still FP16, initially failed. At a
96 KiB exchange budget it needed 103,396 bytes on the worst tile; with
112 KiB allowed, placement failed for a 2,048-byte standard allocation on
tile 740. Both failed logs remain in `real-bs2-baseline/` (`budget96.log` and
`build.log`).

`prepare_real_bs2.py` instead transfers the successful BS2 MLP-control geometry
to the calibrated precision policy. It preserves batch-two layouts, sets each
GEMM's operand precision and parameter-input precision from the established
calibration, retains FP16 image projection, and enables early encoder Q/K
casts. It constructs the matching checkpoint context; the compiler's exact
context check and normal lowering, scheduling, placement, and validation are
unchanged. This recipe fits with the 112 KiB exchange budget and passes the
hardware checks above. No allocator change or numerical-threshold relaxation
was needed. Package/state/log are in `real-bs2/`; input manifests, independent
FP32 references and raw device outputs are in `fixture-bs2/`.

## Artifacts

- `bs1-mlp-control/model.html`: improved FP16-QK BS1 control.
- `bs1-search/model.html` and its adjacent data directory: searched FP8-QK BS1 profile.
- `bs2-both/model.html`: controlled early-Q-and-K BS2 profile.
- `bs2-mlp-control/model.html`: improved MLP layout with FP16 QK.
- `bs2-search/model.html`: final searched BS2 FP8-QK profile.
- `bs1-q/attention.json` and `bs1-baseline-attention.json`: kernel comparison.
- `results.json`: hardware status, numerical checks, cycles, and saved decisions.
- Each build directory retains its build script, compiler log, package and
  search state; performance builds also retain profile input and extraction.

