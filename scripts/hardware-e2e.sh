#!/usr/bin/env bash
set -euo pipefail

: "${POPLAR_SDK_ENABLED:?set POPLAR_SDK_ENABLED to the Poplar SDK directory}"
: "${IPU_CONFIG:?set IPU_CONFIG to an IPUCFG1 configuration capture}"

root=$(cd "$(dirname "$0")/.." && pwd)
tests=(
  generated_compute_exchange_graph_runs_end_to_end
  generated_host_exchange_graph_runs_end_to_end
  generated_multi_epoch_host_exchange_runs_end_to_end
  generated_remote_tile_h2d_and_d2h_run_end_to_end
  generated_remote_tile_d2h_runs_end_to_end
  generated_distinct_multi_source_d2h_runs_end_to_end
  randomized_exchange_graphs_run_end_to_end
)
failed=()
for test in "${tests[@]}"; do
  if ! cargo test --manifest-path "$root/Cargo.toml" -p ipu-runtime \
    --test hardware_e2e "$test" -- --ignored --nocapture --test-threads=1; then
    failed+=("$test")
  fi
done

if ((${#failed[@]})); then
  printf 'hardware acceptance failures:\n' >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
