When migrating/refactoring something, DO NOT add extra adapters to the old behaviour; fully rewrite dependents. Do not e.g. rearrange things into callbacks. Why would you do that? Aim for simplification and code reduction. Code should generally be straight-line and have an obvious hierarchical structure. If what I specify has unexpected architectural consequences, tell me before doing it.

Unless they're complex/long automations (e.g. bulk renames), in which case they should have a dedicated, reviewable script, do edits with your native edit tool and not Python scripts.

Read all of AGENTS.md after each compaction.

# Code quality and refactoring intent

Optimize for the amount of code and the number of concepts a maintainer must
understand to trace execution or add an operation. Correct output is necessary,
but does not establish good structure. The user is asking for a smaller, more
comprehensible implementation, not merely a working implementation with the
requested names and module boundaries. Apply this intent to the consequences
of a request, including callers the user has not explicitly mentioned.

## Ownership and data flow

- Give each decision and invariant an authoritative owner. Other consumers
  should use that owner's result or implementation, rather than independently
  reconstructing it. This includes costing, profiling, validation and build
  collection, not just execution. Reusing a function in several passes is fine;
  duplicating its decision logic is the problem. Caching needs a demonstrated
  cost/reuse benefit; it is not required for semantic deduplication.
- Make the main compilation sequence readable in the entry point. Make each
  algorithm readable as a connected procedure, with helpers for meaningful
  subproblems. Avoid chains of forwarding wrappers, callback inversion and
  scattered one-use helpers that require reconstructing the sequence mentally.
  Splitting a long function across files is not by itself a simplification.
- Put policy where it is chosen and mechanism where it is implemented. A
  planner should choose executable operations and layouts; lowering should not
  secretly perform the missing planning. Kernel families should describe their
  access requirements; allocation should not identify kernels to infer those
  requirements. Target facts belong in the target description. Make selectable
  policies explicit in the appropriate configuration, without building a new
  configuration framework or search system for each one.
- Organize modules around responsibilities and algorithms, not incidental
  stages of a previous implementation. Keep a family's related specifications
  and construction together; share genuinely common machinery. Comments should
  explain why a boundary exists, what facts it establishes, and why constraints
  are necessary. A separate diagram or explanation cannot compensate for
  control flow that the source obscures.

## Representations and abstractions

- Require a semantic reason for every intermediate representation, wrapper,
  enum variant and stored field. Distinct facts such as logical bounds, physical
  padding and allocation identity can justify distinctions. Different existing
  callers, historical construction order, or a convenient place to hang a method
  do not by themselves justify another representation.
- Fix incompatible caller contracts instead of wrapping them in a union or
  translating repeatedly between them. For example, kernel selection can accept
  local extents from both costing and binding; costing should explicitly own
  its approximation rather than disguise local geometry as a whole tensor.
- Preserve useful structure until a consumer actually needs it expanded. Do not
  enumerate bytes, scalar copies or unicasts only to rediscover strides, groups
  or multicast afterward. Likewise, do not encode semantic information into
  symbol strings or ABI words and then decode it for costing or access analysis.
  Strings and argument vectors are legitimate at the actual linker/ABI boundary.
- A shared abstraction should remove the difficult shared work from its callers.
  An access iterator that leaves every caller reimplementing Repeat, aliases
  and lifetime rules has not consolidated that analysis. Conversely, do not
  force analyses with materially different semantics into an elaborate common
  framework just because their loops look similar.
- Prefer direct functions, ordinary data and explicit control flow over traits,
  callbacks, mode objects or generic frameworks without a demonstrated need.
  A simple optional argument may be better than a trait with a no-op backend.
  This is a preference for the simplest complete implementation, not a ban on
  enums, helpers, optional arguments, or inherently complicated algorithms.
  Do not make an interface substantially harder to understand merely to avoid a
  temporary allocation or small recomputation without evidence that it matters.
- Preserve intentional generality and supported behavior. Do not replace an
  explicit policy with an accidental property of today's hardware or workload:
  address order is not a substitute for memory-region preference. Hardware
  restrictions need a real basis. Retiring unused functionality can be valuable,
  but must be an explicit scope decision rather than a hidden way to shrink code.

## Completing a refactor

- Follow the affected information from construction through every consumer.
  Change the producers and consumers to the new contract and delete the obsolete
  path. Moving the old algorithm behind a callback, renaming its types, or
  replacing an enum with another enum carrying the same obligations is not
  completion. Check for semantic duplicates, not merely surviving old names.
- Measure the net result, including supporting code and downstream plumbing.
  If a proposed simplification grows substantially, first look for unfinished
  migration, duplicated dispatch, unnecessary retained state, or a bad boundary.
  Do not offset a growing abstraction with unrelated deletions and call it a
  simplification. Preserve normal whitespace, clear names and useful comments;
  fewer physical lines are not a license to make code denser or less explicit.
- Do not preserve old internal APIs, serialized search state or disposable
  artifacts through compatibility/migration machinery unless compatibility is
  actually required. Regenerating artifacts is preferable here. When a design
  proves wrong, replace the attempt rather than layering another adapter or
  retry/fixup system over it. Surface unexpected architectural consequences.
- Finish the agreed semantic scope before claiming completion. Do not silently
  narrow "share selection and costing" to one kernel family or one caller.
  Distinguish actual deferred work from completed work; do not invent speculative
  problems to pad a review. Passing tests, satisfying counters and producing a
  persuasive summary do not establish that the requested structure exists.
- Validate behavior with meaningful properties, varied inputs and appropriate
  hardware checks. Avoid tests that merely restate emitted fields, reproduce the
  implementation, or freeze incidental organization. Behavioral checks protect
  correctness; they do not prove comprehensibility or absence of duplication.
  Inspect the resulting implementation for those separately, without requiring
  the user to prompt another audit to discover known unfinished work.

# Commit reduction rule

Every commit must not increase production code size and must reduce at least one of:
type count, enum variant count, field count, or function count. None of those
four counts may increase. This includes documentation and tooling commits:
zero change does not satisfy the rule.

Run `python3 tools/reduction-gate/check.py` on the staged changes before
committing. Install the tracked hook with `git config core.hooksPath .githooks`.
Measurement definitions and the exception procedure are in
[tools/reduction-gate/README.md](tools/reduction-gate/README.md).

Exceptions require explicit user review of the concrete changes. Do not
self-approve an exception, disable the hook, alter the measurement scope to
make a commit pass, or compress formatting/move code into excluded files to
satisfy the counters. Gate changes and changes to generated declarations or
macro-generated structure also require user review; the source counters are
not a proof of simplification.

After the user approves a specific exception, record it using the documented
approval command, including their reason/reference. Approval applies only to
that exact staged tree and parent commit; further changes need fresh review.
