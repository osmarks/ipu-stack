When migrating/refactoring something, DO NOT add extra adapters to the old behaviour; fully rewrite dependents. Do not e.g. rearrange things into callbacks. Why would you do that? Aim for simplification and code reduction. Code should generally be straight-line and have an obvious hierarchical structure. If what I specify has unexpected architectural consequences, tell me before doing it.

Unless they're very complex automations, in which case they should have a dedicated, reviewable script, do edits with your native edit tool and not Python scripts.

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
