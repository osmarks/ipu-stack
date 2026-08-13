#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CONFIGURATION [OUTPUT.json]" >&2
  exit 2
fi

configuration=$1
output=${2:-ipu21-cost-calibration.json}
calibration_dir=$(mktemp -d)
trap 'rm -rf -- "$calibration_dir"' EXIT

cargo build --release -p ipu-tests -p ipu-cli

profiles=()
run_gemm() {
  local rows=$1
  local inner=$2
  local columns=$3
  local profile="$calibration_dir/gemm-${rows}-${inner}-${columns}.ipuprofile"
  target/release/ipu-trivial-test "$configuration" \
    --workload gemm-benchmark \
    --benchmark-rows "$rows" \
    --benchmark-inner "$inner" \
    --benchmark-columns "$columns" \
    --profile-output "$profile"
  profiles+=("$profile")
}

# Cover retained output widths, K blocking, small-row specialization, direct
# interleaved loads, streamed panels, and local staging without enumerating a
# general matrix-shape space.
run_gemm 8192 64 32
run_gemm 8192 64 64
run_gemm 4096 128 128
run_gemm 2048 256 64

mlp_profile="$calibration_dir/siglip-mlp.ipuprofile"
target/release/ipu-trivial-test "$configuration" \
  --workload siglip-mlp-benchmark \
  --tiles 1472 \
  --profile-output "$mlp_profile"
profiles+=("$mlp_profile")

build_id=$(git rev-parse HEAD:device)
target/release/ipu-stack profile-calibrate \
  "${profiles[@]}" \
  --build-id "$build_id" \
  --output "$output"
