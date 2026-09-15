#!/usr/bin/env python3
"""Check the index, never the working tree. Exceptions bind parent + tree."""

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

METRICS = ("lines", "types", "variants", "fields", "functions")
ROOT = Path(__file__).resolve().parent


def run(*args, **kwargs):
    return subprocess.check_output(args, **kwargs)


def production(name):
    path = PurePosixPath(name)
    return (
        name.startswith(("crates/", "device/"))
        and path.suffix in {".rs", ".cpp", ".S", ".inc", ".h", ".def"}
        and not name.startswith("crates/ipu-tests/")
        and not {"tests", "benches"}.intersection(path.parts)
        and path.name not in {"tests.rs", "test_support.rs"}
        and not path.stem.startswith("test_")
        and not path.stem.endswith(("_tests", "_test", "_bench"))
    )


def protected(name):
    return (
        name == "AGENTS.md"
        or name.startswith((".githooks/", "tools/reduction-gate/"))
        or (name.startswith("crates/") and name.count("/") == 2 and name.endswith("/build.rs"))
        or name.endswith((".capnp", ".def"))
    )


def listing(tree):
    if tree == "unborn":
        return {}
    result = {}
    for entry in run("git", "ls-tree", "-rz", tree).split(b"\0"):
        if entry:
            meta, name = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            name = name.decode()
            if production(name) or protected(name):
                if kind != "blob" or mode == "120000":
                    raise ValueError(f"unsupported source entry: {name} ({mode}, {kind})")
                result[name] = oid
    return result


def violations(before, after):
    errors = []
    if after["lines"] > before["lines"]:
        errors.append("production code lines must not increase")
    for key in METRICS[1:]:
        if after[key] > before[key]:
            errors.append(f"{key} must not increase")
    if not any(after[key] < before[key] for key in METRICS[1:]):
        errors.append("at least one declaration count must decrease")
    return errors


def measure(trees):
    target = Path(run("git", "rev-parse", "--show-toplevel").decode().strip()) / "target/reduction-gate"
    subprocess.run([
        "cargo", "build", "--quiet", "--locked", "--release",
        "--manifest-path", str(ROOT / "Cargo.toml"), "--target-dir", str(target),
    ], check=True)
    keys = sorted({(PurePosixPath(name).suffix, oid)
                   for tree in trees for name, oid in tree.items() if production(name)})
    # Batch blob reads avoid a process per source file. Deduplicate unchanged
    # blobs across the two trees; no persistent cache/invalidation machinery.
    data = run("git", "cat-file", "--batch", input="".join(oid + "\n" for _, oid in keys).encode())
    inputs, offset = [], 0
    for suffix, _ in keys:
        end = data.index(b"\n", offset)
        size = int(data[offset:end].split()[2])
        source = data[end + 1:end + 1 + size].decode()
        inputs.append(json.dumps({"path": "source" + suffix, "source": source}))
        offset = end + size + 2
    output = run(str(target / "release/ipu-reduction-metrics"),
                 input="\n".join(inputs), text=True)
    rows = [json.loads(line) for line in output.splitlines()]
    if len(rows) != len(keys):
        raise ValueError("counter returned an incomplete result")
    counts = dict(zip(keys, rows))
    results = []
    for tree in trees:
        total = dict.fromkeys(METRICS, 0)
        macros = []
        for name, oid in tree.items():
            if production(name):
                row = counts[PurePosixPath(name).suffix, oid]
                for key in METRICS:
                    total[key] += row[key]
                macros.extend(row["macros"])
        results.append((total, sorted(macros)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve", metavar="REVIEW", help="record explicit user approval of this exact exception")
    args = parser.parse_args()
    parent_result = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    parent = parent_result.stdout.strip() if parent_result.returncode == 0 else "unborn"
    tree = run("git", "write-tree").decode().strip()
    old, new = listing(parent), listing(tree)
    (before, old_macros), (after, new_macros) = measure((old, new))
    for key in METRICS:
        print(f"{key:10} {before[key]:8} -> {after[key]:8} ({after[key] - before[key]:+})")
    errors = violations(before, after)
    if old_macros != new_macros:
        errors.append("macro definitions/includes changed: generated structure needs review")
    if {p: h for p, h in old.items() if protected(p)} != {p: h for p, h in new.items() if protected(p)}:
        errors.append("gate/rule/generator changes need review")
    approval = Path(run("git", "rev-parse", "--git-path", "reduction-approval.json").decode().strip())
    if args.approve is not None:
        if not args.approve.strip():
            raise ValueError("approval must identify the user's explicit review")
        approval.write_text(json.dumps({"parent": parent, "tree": tree, "review": args.approve}, indent=2) + "\n")
        print(f"Recorded user-reviewed exception for {tree} on {parent}.")
    if not errors:
        print("Reduction gate passed.")
        return 0
    for error in errors:
        print(f"FAIL: {error}")
    record = json.loads(approval.read_text()) if approval.exists() else {}
    if record.get("parent") == parent and record.get("tree") == tree and record.get("review"):
        print(f"Accepted exact-tree exception: {record['review']}")
        return 0
    print("Commit blocked. Revise the changes or obtain explicit user review of an exception.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Reduction gate failed closed: {error}", file=sys.stderr)
        sys.exit(1)
