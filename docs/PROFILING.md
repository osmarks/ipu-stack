# Profile queries

`ipu-profile` is the programmatic analytics interface for cycle profiles.
`ipu-stack profile-query` exposes it as bounded text or JSON, so Python and
shell tools do not need to decode the Cap'n Proto schema or load the HTML
renderer.

The default query attributes phase-critical time by kernel:

```sh
cargo run -q -p ipu-cli -- profile-query profile.capnp
```

Useful drilldowns include:

```sh
# Scheduled steps within the GEMM accumulation kernel.
cargo run -q -p ipu-cli -- profile-query profile.capnp \
  --kernel gemm_f32_accumulate --group-by phase --limit 20

# Operations active at one normalized graph offset, with two complete samples.
cargo run -q -p ipu-cli -- profile-query profile.capnp \
  --at 1000000 --group-by operation --samples 2 --json

# The slowest samples for one block coordinate.
cargo run -q -p ipu-cli -- profile-query profile.capnp \
  --kind compute --metadata output_block_row=12 --metadata inner_block=31 \
  --group-by tile --sort-by maximum-cycles --samples 8 --json

# Exchange phases preceding one kernel.
cargo run -q -p ipu-cli -- profile-query profile.capnp \
  --kind exchange --metadata next_kernel=gemm_f32_accumulate --group-by phase

# Compare accumulation cost by inner block number.
cargo run -q -p ipu-cli -- profile-query profile.capnp \
  --kernel gemm_f32_accumulate --group-by metadata --metadata-key inner_block
```

Filters on `--tile`, `--phase`, and `--metadata` are repeatable. Repeated
metadata filters are combined with AND. `--kernel` is exact;
`--operation-contains` is a case-sensitive substring filter. `--at` uses the
same offset as the HTML timeline: each tile's first counter sample is offset
zero, and counter wrap is handled with wrapping subtraction.

Each aggregate reports:

- `phaseCycles`: the sum of the maximum matching tile duration in each phase;
  this estimates serialized graph-time contribution.
- `workCycles`: the sum of every matching tile duration.
- `averageActiveTiles`: `workCycles / phaseCycles`.
- sample mean, p50, p95, and maximum duration.
- participating phase, tile, and sample counts.

`--samples N` adds the `N` longest matching samples, including tile, normalized
offset, phase, operation, kernel, and all semantic metadata. It defaults to
zero so broad queries remain bounded.

Python can consume the stable JSON output without an in-process binding:

```python
import json
import subprocess

result = subprocess.run(
    [
        "target/release/ipu-stack",
        "profile-query",
        "profile.capnp",
        "--group-by", "kernel",
        "--json",
    ],
    check=True,
    capture_output=True,
    text=True,
)
profile = json.loads(result.stdout)
```

Rust callers can use `ipu_profile::query(&ProfileReport, &Query)` directly.
