# Profile storage placement

Cycle samples now use auxiliary requests in the normal tensor allocator. Each
execution tile requests its own sample count, including inactive tiles. The
allocator chooses addresses with the same lifetime ordering and fragmentation
search used for tensors. Buffers remain live through host readback; host
support allocation excludes them. Package executable reservations already
exclude their entire memory elements from these data requests.

This removes the former uniform 896-byte reservation at 0x80000, which split a
useful allocation interval on every tile and prevented both saved BS2 recipes
from building with profiling. Provisional addresses are used only for code and
host-support sizing; final instrumentation, readback, and exact memory reports
use the chosen per-tile addresses. Placement optimization retains the requests.

## Validation

Both full 27-layer SigLIP BS2 packages build with profiling, the saved recipes,
B1024 exchange scheduling, and the existing 96 KiB exchange-table budget. Each
passed two resident inference checks against the FP32 reference. These are
randomized benchmark weights/inputs, not the separately validated pretrained
image fixture.

| PV arithmetic | Cropped cycles | Time per batch | Minimum cosine |
|---|---:|---:|---:|
| FP16 | 17,950,902 | 11.967268 ms | 0.993905451 |
| FP8 | 17,462,346 | 11.641564 ms | 0.994151272 |

Times use the renderer's default crop and include the full 27-layer model, with
only the first Repeat iteration profiled in detail. They exclude host I/O.

Artifacts: `artifacts/profile-allocation-20260913/bs2-{control,pv}/` contains
`model.ipuexe`, `model.profile.capnp`, `model.html`, `query.json`, and `memory/`.
Build scripts and logs are in the parent directory.

Each package has 1,472 buffers, sized 380–896 bytes. The FP16 package uses 240
distinct addresses; FP8 uses 236. Exact memory records were checked for overlap
against all other allocations, including executable support and host aperture.

Validation: 286 codegen tests passed, 5 ignored. Allocator regression covers
auxiliary/tensor placement, inactive execution tiles, and retaining samples
through readback rather than borrowing the host aperture. Profile binding
regression covers physical/logical tile reordering and unequal sample counts.

## Cost-model investigation

Compared the prior BS1 FP8-PV recipe (`producer-fp8-20260913/pv-early-full`) with
its early-downprojection-cast successor (`fp8-pv-batches-20260913/bs1-step64`).
Estimated cycles improved 0.34%, while measured cycles worsened 1.24%.

Across the first layer's 25 exchanges, scheduled event horizons sum to 157,023
cycles before and 159,253 after. Measured time after the last tile's profile
entry sums to 165,210 and 169,890 respectively. Thus the scheduler horizons
account for 2,230 of the additional 4,680 exchange cycles; the residual overhead
increases by 2,450. Almost all that residual increase is in exchange phase 27
(295 to 2,786 cycles beyond its event horizon). These entries precede generated
exchange setup, so this residual is not evidence of inaccurate transfer timing.

`scheduled_program_cycles` prices event horizons plus a fixed phase overhead.
It omits the generated cross-phase row-sharing and Repeat patch programs.
Those are a concrete accounting limitation, but this comparison does not
isolate their exact contribution from other setup, nor do they explain all
compute/preparation differences. No costing constants were changed to fit
this one comparison. A future improvement should share patch-work accounting
with final row assembly, preserving tile-local overlap, rather than bolt another
approximation onto the earlier layout model.
