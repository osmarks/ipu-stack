# Advisory commit metrics

Install once per checkout:

```sh
git config core.hooksPath .githooks
```

The pre-commit hook compares HEAD with `git write-tree` (the index). Unstaged
edits do not affect the measurements. It reports production code lines and
type, variant, field and function counts before and after each commit.
All deltas are advisory; there are no thresholds or approval exceptions.
Missing dependencies and parse errors produce a warning without blocking a commit.

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
generated code are not counted. Changes to macro definitions, includes,
crate-root `build.rs`, `.capnp`, or `.def` inputs can change generated structure
without appearing in these counts. Third-party procedural
macros are not expanded. The metric does not count C++ declarations or
assembly entry points, HTML/JS, `.hpp` files, documentation, or Python tools.
These limits preserve the existing audit scope; moving code outside that
scope is not a valid reduction. Numerical compliance does not prove improved
design. Git hooks are local; other checkouts must install the hook too.
