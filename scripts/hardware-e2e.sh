#!/usr/bin/env bash
set -euo pipefail

: "${IPU_CONFIG:?set IPU_CONFIG to an IPUCFG1 configuration capture}"
: "${POPLAR_SDK_ENABLED:?set POPLAR_SDK_ENABLED to the Poplar SDK root}"

root=$(cd "$(dirname "$0")/.." && pwd)
cargo run --manifest-path "$root/Cargo.toml" -p ipu-tests --bin ipu-trivial-test -- \
  "$IPU_CONFIG" \
  --sdk "$POPLAR_SDK_ENABLED" \
  --device "${IPU_DEVICE:-/dev/ipu0}" \
  --package "${IPU_TEST_PACKAGE:-/tmp/ipu-trivial.ipuexe}"
