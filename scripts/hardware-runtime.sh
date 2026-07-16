#!/usr/bin/env bash
set -euo pipefail

: "${POPLAR_SDK_ENABLED:?set POPLAR_SDK_ENABLED to the Poplar directory}"
: "${IPU_CONFIG:?set IPU_CONFIG to an IPUCFG1 configuration capture}"
: "${IPU_TILE_COUNT:?set IPU_TILE_COUNT to the package tile count}"

root=$(cd "$(dirname "$0")/.." && pwd)
output=${IPU_TEST_OUTPUT:-/tmp/ipu-stack-runtime}

cargo run -q --manifest-path "$root/Cargo.toml" -p ipu-cli -- \
  kernel-compile "$root/device/runtime.S" "$output" --name runtime \
  --sdk "$POPLAR_SDK_ENABLED"
cargo run -q --manifest-path "$root/Cargo.toml" -p ipu-cli -- \
  package-runtime-fixture "$output/runtime.o" -o "$output/runtime.ipuexe" \
  --tiles "$IPU_TILE_COUNT"
cargo run -q --manifest-path "$root/Cargo.toml" -p ipu-cli -- \
  load "$output/runtime.ipuexe" \
  "$POPLAR_SDK_ENABLED/bin/ipu/tile_bootloader_ipu2.elf" "$IPU_CONFIG"
