# Compiler refactor continuation

User priorities (2026-09-05): smaller, clearer compiler; general mid-level
copies/rearrangements and views; sane kernel contracts; fewer layer violations.
Audits are leads, not requirements. Keep existing two-pass exchange scheduling.
Commit tested chunks frequently. Preserve the user-generated callgraph artifacts.

## Validated checkpoint

The first checkpoint includes the accumulated previous refactors and runtime
fixes, plus shared axis-factor view geometry and kernel specialization keys.
Workspace release tests (140 total, including doctests) and strict Clippy pass.
Five hardware workloads pass: 64-tile GEMM and batched GEMM, canonical batch-one
MLP, projected attention, and attention smoke. All five tile images are identical
to the preceding `/tmp/completion-final-*` packages. Logs/packages:
`/tmp/views-kernels-*`; test log `/tmp/views-kernels-workspace.log`.

Keep the configuration replay CCSR readback after every MMIO write. It fixed the
observed startup/run instability. Autoreset has been removed. Completion checking
accepts only the known terminal InvalidProgramCounter at the named completed
symbol with the completion word set; it still rejects other faults. The debug
interface's post-host-exchange visibility is not claimed to be fixed.

## Next work

1. Kernel responsibilities are now split into ABI, specialization collection,
   geometry, build recipes, and placed-call materialization. Four kernel tests
   pass after extraction; duplicate rearrangement symbol insertion is removed.
2. Decouple physical storage-span geometry from LowShard so mid planning can use
   actual relative spans. Low conversion currently calls the cycle estimator to
   reconsider direct word exchange versus staging; move this policy to a shared
   mid-level copy plan, rather than duplicating its heuristic again.
3. Extend views beyond the axis-factor primitive as useful. Graph SplitHeads is
   still semantic syntax; attention candidate layouts and cost paths remain
   specialized. Do not mistake renaming those paths for generalization.
4. Kernel attention-stage compilation currently assumes one common configuration
   and at most two query/key sizes. This deserves correction before claiming the
   kernel design supports arbitrary additional operations/configurations.

No subagents were used. Work is on `refactor/compiler-views-kernels`.
