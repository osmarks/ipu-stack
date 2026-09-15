# Commit reduction gate

Install once per checkout:

```sh
git config core.hooksPath .githooks
```

The pre-commit hook compares HEAD with `git write-tree` (the index). Unstaged
edits do not affect the measurements. It requires no increase in production code lines,
no increase in types, variants, fields, or functions, and a decrease in at
least one of those four declaration counts. Even documentation-only commits
need an exception. Missing dependencies and parse errors block commits.

Run manually with `python3 tools/reduction-gate/check.py`. Python 3, Cargo,
and the locked Rust dependencies are required. The counter builds under
`target/reduction-gate`; unchanged blobs are counted once per invocation.

## Measurement contract

Source scope matches the September 15 refactor accounting: `.rs`, `.cpp`,
`.S`, `.inc`, `.h`, `.def` under `crates/` and `device/`, excluding
`crates/ipu-tests/`, `tests/`, `benches/`, `tests.rs`, `test_support.rs`,
`test_*`, `*_test`, `*_tests`, and `*_bench` files.

Rust size counts physical lines containing tokens, excluding documentation
and items provably disabled when `cfg(test)` is false. Unknown platform and
feature conditions remain included. Device size counts nonblank,
non-comment lines, including preprocessor directives. This is a source-size
metric, not generated machine code size. Keep normal formatting.

The four declaration counts cover handwritten Rust syntax:

- Types: structs, enums, unions, aliases, traits, and associated/foreign types.
- Variants: enum variants.
- Fields: named and tuple fields, including enum-variant payloads.
- Functions: free/nested functions, methods, trait declarations/defaults, and
  foreign declarations. Closures and function-pointer types do not count.

Test-only items, fields, and variants are excluded. Macro expansions and
generated code are not counted; changes to macro definitions, includes,
crate-root `build.rs`, `.capnp`, or `.def` inputs require review. Third-party procedural
macros are not expanded. The metric does not count C++ declarations or
assembly entry points, HTML/JS, `.hpp` files, documentation, or Python tools.
These limits preserve the existing audit scope; moving code outside that
scope is not a valid reduction. Numerical compliance does not prove improved
design. Changes to the gate or its rules also require explicit review.

## User-reviewed exceptions

Prepare and stage the complete change, run the check, and present its changes
and failing metrics to the user. Only after explicit approval, record their
review/reason and commit:

```sh
python3 tools/reduction-gate/check.py --approve 'User approved <change>: <reason/reference>'
git commit
```

The local approval records the exact parent and staged tree, and cannot
authorize another snapshot. It is a workflow record, not authentication:
agents must never self-approve, use `--no-verify`, or disable the hook. Git
hooks are local; other checkouts must install this hook too. Amended commits
are checked against current HEAD, so a message-only amend needs an exception.
