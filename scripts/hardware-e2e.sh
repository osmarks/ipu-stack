#!/usr/bin/env bash
set -euo pipefail

: "${POPLAR_SDK_ENABLED:?set POPLAR_SDK_ENABLED to the Poplar SDK directory}"
: "${IPU_CONFIG:?set IPU_CONFIG to an IPUCFG1 configuration capture}"

root=$(cd "$(dirname "$0")/.." && pwd)
tests=(
  generated_host_exchange_graph_runs_end_to_end
  generated_multi_epoch_host_exchange_runs_end_to_end
  generated_remote_tile_d2h_runs_end_to_end
  randomized_exchange_graphs_run_end_to_end
)
for test in "${tests[@]}"; do
  cargo test --manifest-path "$root/Cargo.toml" -p ipu-runtime \
    --test hardware_e2e "$test" -- --ignored --nocapture --test-threads=1
done
