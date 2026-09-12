# Placement constraint dumps

Set `IPU_STACK_PLACEMENT_DUMP=/path/to/directory` when running a build to save
each tile problem that fails both production allocation orders. Also set
`IPU_STACK_PLACEMENT_DUMP_ALL=1` to include successful problems. This is opt-in
diagnostic output; it does not change allocation or reject a build if writing
the dump fails. Failures to write are logged. Content-derived filenames avoid
repeated copies of identical problems during local search. Files are ready to
consume after the producing build exits.

Each version-1 JSON file is independent of the compiler graph:

* `ranges`: available half-open byte-address intervals, after reservations.
* `region_boundary`, `region0_element_bytes`, `region1_element_bytes`: address
  regions and effective memory-element sizes. Allocations cannot cross the
  region boundary. Standard allocations may use either region; interleaved
  allocations may only use region 1, above `region_boundary + interleaved_offset`.
* `host_scratch_range`: available only when `first != 0` and `last != 4294967295`.
* `requests`: allocation groups, including aliases and contiguous Repeat members.
  Lifetimes are **inclusive** at both ends. `bytes` includes access tails.
  `alignment` constrains the base address. `assignments` lists root IDs and
  byte offsets within the group.
* A non-null `region1_stride` replaces member offsets with `index * stride`
  and the total size with `member_count * stride` in region 1. Such groups also
  require base alignment to the region's element size in either region.
* `conflicts` lists roots whose allocation groups must occupy disjoint effective
  memory elements. The constraint applies to the entire group span, not merely
  to the named member, and is separate from ordinary lifetime interference.
  Element boundaries are absolute address multiples of the relevant size.
* `placed` records whether the existing allocator succeeded.
  `assigned_root_spans` contains its complete or partial placement; all roots
  in a group map to the same whole-group span. A partial placement is diagnostic
  information, not a constraint on a replacement solver.

The root IDs are identifiers only; they need not be consecutive. An independent
solver needs no tensor shapes or operator semantics to solve this placement
problem. It must preserve every constraint above, and distinguish a proved
infeasible instance from a search timeout.

Historical batch-two failures and their analysis are described in
[BATCH2_FRAGMENTATION_2026_09_12.md](BATCH2_FRAGMENTATION_2026_09_12.md).
