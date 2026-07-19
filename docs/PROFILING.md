# Profile queries

`ipu-profile` is the programmatic analytics interface for cycle profiles.
`ipu-stack profile-query` exposes it as bounded text or JSON, so Python and
shell tools do not need to decode the Cap'n Proto schema or load the HTML
renderer.

## Collection granularity

Set `IPU_PROFILE_OUTPUT` to enable profiling and select one of three levels with
`IPU_PROFILE_GRANULARITY`:

| Value | Intervals on each tile | Intended use |
| --- | --- | --- |
| `graph` | One for the complete graph | Low-overhead benchmark timing |
| `phase` | Compute phases plus separate sync/exchange intervals | Coarse semantic analysis; default |
| `step` | Sync/exchange intervals plus every lowered kernel call | Detailed code-generation diagnostics |

`phase` retains operation and kernel names plus compact planner metadata such as
layer, block, shape, and transfer-byte counts. `step` additionally retains
individual operands, arguments, and transfer details. Every setting samples all
tiles. Instrumentation itself runs workers and device barriers at interval
boundaries, so compare performance using an unprofiled run. The legacy
`IPU_PROFILE_AGGREGATE` variable selects `graph` for compatibility.

Exchange epochs produce a `synchronization` interval ending immediately after
the supervisor sync instruction returns, followed by an `exchange` interval
covering the worker barrier and exchange plan. Compute phases use `compute` on
tiles with a scheduled kernel and `idle` on the others. Idle tiles normally
reach the following synchronization early, so their wait for active compute
tiles is attributed to that synchronization rather than to idle compute.

For example, collect and render the coarse semantic view of the MLP:

```sh
IPU_PROFILE_GRANULARITY=phase \
IPU_PROFILE_OUTPUT=profiles/mlp-512x2048x8-phase.capnp \
  cargo run --release -q -p ipu-runtime --bin ipu-mlp-e2e
cargo run --release -q -p ipu-cli -- profile-render \
  profiles/mlp-512x2048x8-phase.capnp \
  -o profiles/mlp-512x2048x8-phase.html
```

The default query attributes phase-critical time by kernel:

```sh
cargo run --release -q -p ipu-cli -- profile-query profile.capnp
```

Useful drilldowns include:

```sh
# Scheduled steps within the GEMM accumulation kernel.
cargo run --release -q -p ipu-cli -- profile-query profile.capnp \
  --kernel gemm_f32_accumulate_large_rows --group-by phase --limit 20

# Operations active at one normalized graph offset, with two complete samples.
cargo run --release -q -p ipu-cli -- profile-query profile.capnp \
  --at 1000000 --group-by operation --samples 2 --json

# The slowest samples for one block coordinate.
cargo run --release -q -p ipu-cli -- profile-query profile.capnp \
  --kind compute --metadata output_block_row=12 --metadata inner_block=31 \
  --group-by tile --sort-by maximum-cycles --samples 8 --json

# Exchange phases preceding one kernel.
cargo run --release -q -p ipu-cli -- profile-query profile.capnp \
  --kind exchange --metadata next_kernel=gemm_f32_accumulate_large_rows --group-by phase

# Compare accumulation cost by inner block number.
cargo run --release -q -p ipu-cli -- profile-query profile.capnp \
  --kernel gemm_f32_accumulate_large_rows --group-by metadata --metadata-key inner_block
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
