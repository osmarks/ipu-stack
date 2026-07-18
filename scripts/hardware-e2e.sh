#!/usr/bin/env bash
set -euo pipefail

: "${POPLAR_SDK_ENABLED:?set POPLAR_SDK_ENABLED to the Poplar SDK directory}"
: "${IPU_CONFIG:?set IPU_CONFIG to an IPUCFG1 configuration capture}"

root=$(cd "$(dirname "$0")/.." && pwd)
cargo test --manifest-path "$root/Cargo.toml" -p ipu-runtime \
  --test hardware_e2e -- --nocapture --test-threads=1
