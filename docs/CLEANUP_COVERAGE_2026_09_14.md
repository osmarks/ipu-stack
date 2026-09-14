# Cross-component cleanup coverage

The earlier Rust cleanup is committed through `59862e8`; its full workspace
check passed 394 tests with 7 ignored. This broader pass remains in progress.
Generated artifacts and historical benchmark outputs are reference material,
not sources to rewrite.

| Component | Current coverage | Remaining examination |
| --- | --- | --- |
| Compiler/planner, storage, exchange | Shared fusion contracts, span geometry, endpoint accounting and package assembly reviewed; forward/reverse view adapters consolidated in low; identity-copy predicates unified | Broader candidate generation and lowering control flow |
| Package/driver/runtime/CLI | Shared binding extents, host capture, logging and profile interning reviewed; redundant runtime error wrapper removed | Remaining loader/protocol and command dispatch paths |
| ELF toolchain | Hashing/cache and linker paths inspected; instruction field relocations consolidated; relocation bounds constrained to their section | No additional duplication selected from this inspection |
| Device kernels/runtime | Dense packing supervisor loops, GEMM weight loads and runtime word copies consolidated; cast and normalization families inspected | Remaining kernel/runtime families; preserve independent numerical references |
| Profile viewers | HTML entry points located | Interaction, data loading and rendering logic across viewers |
| Calibration tools | Shared Torch-only F143 module; shared Hessian accumulation and scale selection; shared calibration loop with exception-safe hook removal; offline placement tool inspected (independent constraints intentionally retained) | No additional duplication selected in the pretrained exporter |
| Experiment scripts | Gather/packing affine detection consolidated; frontier, batch/MLP sweep orchestration and SDK summary inspected | Deeper sweep provenance/error handling |
| Diagnostic harnesses | Shared bindings/logical packing reviewed; six standalone kernel binaries share locked runtime loading and use HostSession::finish for deferred output completion; four share timestamp bindings | Remaining fixture assembly; terminal-state polling now shares the model diagnostic checker |

Calibration validation: four tests from `tools/test*calibration*.py`; deterministic
comparison against the pre-extraction quantization functions produced identical
FP8 projections, reconstructed weights, bias corrections and objectives. Both
successful collection and injected-forward-failure hook cleanup were checked.
The core numerical module was also imported without loading Transformers.

ELF validation: five unit tests and workspace checks pass. A synthetic ELF
fixture with two retained sections linked a valid cross-section reference to
the expected bytes; moving the relocation offset beyond its originating
section was rejected. Field tests cover opcode preservation, maximum values,
misalignment, truncation, and `usize::MAX` offsets.

Device packing validation: the SDK compiled both old and new block-major and
transposed-right kernels with identical allocated section bytes and normalized
relocations. All 22 kernel selection tests passed. AMP-left packing retains its
different destination stride and worker frame.

GEMM/runtime validation: the SDK produced identical allocated section bytes
and normalized relocations for FP32 GEMM, FP16 GEMM, all four FP16/FP8 ×
standard/interleaved weight dispatchers, and the static runtime. The shared
weight loader preserves its unrolled order and both instruction widths.
The full codegen suite also passed: 295 tests, 5 ignored.

Exchange experiment validation: ten reconstruction/geometry tests pass.
On 2,000 deterministic copy sequences, both the optimistic loop count and the
forward-only task descriptors match their previous independent implementations.
Both changed scripts and their tests pass Ruff checks.

Diagnostic completion validation: all ipu-tests targets compile. Hardware
unpack checks passed 14 cases / 84,208 bytes; FP8 cast checks passed 1,404
cases / 7,517,232 bytes, including scales, tails, padding and shifted overlap.
The softmax and kernel-equivalence binaries retain their additional context
halt checks after the shared output handshake.

Diagnostic setup validation: all diagnostic targets compile after the shared
loader extraction; 14 unpack cases / 84,208 bytes pass on hardware. The runtime
can only be borrowed from its owner, which drops it before releasing the lock.
Pretrained fixture export and baseline/candidate entry points were also sampled;
no further consolidation was selected from those portions in this pass.

Completion diagnostic validation: model, softmax and kernel-equivalence
paths now share the existing model terminal-state checker, including the
completion marker, InvalidProgramCounter exception and halted workers. Four
softmax hardware cases passed after extraction; all diagnostic targets compile.

View adapter cleanup: logical-range to shard-extent translation now belongs
to low expansion and is shared by forward and reversed views; mid/view.rs
is removed. The low expansion suite passed (see /tmp/ipu-view-cleanup-tests.log).

Runtime cleanup: removed the single-variant wrapper around DriverError and
its thiserror dependency; RuntimeError remains a public alias. Workspace
compilation passes. No callers pattern-matched the old wrapper in this tree.

Identity mapping validation: an omitted-offset versus explicit-zero-offset
copy regression failed before consolidation (exchange row estimate 18,436
versus 4,612 bytes) and passes afterward. Cast motion, producer tracing and
costing now use the same predicate. Full codegen suite: 296 passed, 5 ignored.
