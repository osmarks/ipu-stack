# Cross-component cleanup coverage

The earlier Rust cleanup is committed through `59862e8`; its full workspace
check passed 394 tests with 7 ignored. This broader pass remains in progress.
Generated artifacts and historical benchmark outputs are reference material,
not sources to rewrite.

| Component | Current coverage | Remaining examination |
| --- | --- | --- |
| Compiler/planner, storage, exchange | Shared fusion contracts, span geometry, endpoint accounting and package assembly reviewed in the earlier pass | Broader candidate generation and lowering control flow |
| Package/driver/runtime/CLI | Shared binding extents, host capture, logging and profile interning reviewed in the earlier pass | Remaining loader/protocol and command dispatch paths |
| ELF toolchain | Source hashing and compilation/cache entry points inspected | Linker and relocation handling |
| Device kernels/runtime | Source inventory taken | Compare kernel families and shared assembly support; preserve independent numerical references |
| Profile viewers | HTML entry points located | Interaction, data loading and rendering logic across viewers |
| Calibration tools | Shared Torch-only F143 module; shared Hessian accumulation and scale selection; shared calibration loop with exception-safe hook removal | Broader fixture export and placement tools |
| Experiment scripts | Gather/packing/frontier scripts sampled | Remaining sweeps, analyses and their common file/command handling |
| Diagnostic harnesses | Shared bindings and logical packing reviewed | Standalone kernel fixtures and orchestration |

Calibration validation: four tests from `tools/test*calibration*.py`; deterministic
comparison against the pre-extraction quantization functions produced identical
FP8 projections, reconstructed weights, bias corrections and objectives. Both
successful collection and injected-forward-failure hook cleanup were checked.
The core numerical module was also imported without loading Transformers.
