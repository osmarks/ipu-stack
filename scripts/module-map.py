#!/usr/bin/env python3
"""Generate docs/module-map.html from staged production sources; --stage for hooks."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "scripts/module-map"
OUTPUT = ROOT / "docs/module-map.html"
spec = importlib.util.spec_from_file_location("gate", ROOT / "tools/reduction-gate/check.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def run(*args, **kwargs):
    return subprocess.check_output(args, cwd=ROOT, **kwargs)


def module(path):
    parts = Path(path).parts
    if parts[0] != "crates":
        return "device::" + Path(path).stem
    crate = parts[1].replace("-", "_")
    tail = list(parts[3:]) if parts[2] == "src" else list(parts[2:])
    tail[-1] = Path(tail[-1]).stem
    if tail[-1] in ("lib", "main", "mod"):
        tail.pop()
    return "::".join([crate] + tail)


def generate():
    # Read the index, including partial staging and deletions, in one blob batch.
    tree = run("git", "write-tree").decode().strip()
    entries = gate.listing(tree)
    paths = sorted(p for p in entries if gate.production(p))
    blobs = run("git", "cat-file", "--batch",
                input="".join(entries[p] + "\n" for p in paths).encode())
    sources, offset = {}, 0
    for path in paths:
        end = blobs.index(b"\n", offset)
        size = int(blobs[offset:end].split()[2])
        sources[path] = blobs[end + 1:end + 1 + size].decode()
        offset = end + size + 2

    # Reuse the gate's counting implementation without changing its interface.
    # The temporary parser also exposes excluded ranges for dependency analysis.
    parser = ROOT / "target/module-map/parser"
    (parser / "src/bin").mkdir(parents=True, exist_ok=True)
    for name in ("Cargo.toml", "Cargo.lock", "src/main.rs"):
        source = (ASSETS / "parser" / name).read_bytes()
        if not (parser / name).exists() or (parser / name).read_bytes() != source:
            (parser / name).write_bytes(source)
    counter = (ROOT / "tools/reduction-gate/src/main.rs").read_text()
    marker = '"macros":counter.macros}'
    if counter.count(marker) != 1:
        raise ValueError("reduction counter output changed; update module-map integration")
    counter = counter.replace(marker, '"macros":counter.macros,"excluded":counter.excluded}')
    metrics_source = parser / "src/bin/metrics.rs"
    if not metrics_source.exists() or metrics_source.read_text() != counter:
        metrics_source.write_text(counter)
    target = ROOT / "target/module-map"
    run("cargo", "build", "--quiet", "--locked", "--release", "--manifest-path",
        str(parser / "Cargo.toml"), "--target-dir", str(target))
    payload = "\n".join(json.dumps({"path": p, "source": sources[p]}) for p in paths)
    metrics = [json.loads(s) for s in run(
        str(target / "release/metrics"), input=payload, text=True).splitlines()]
    if len(metrics) != len(paths):
        raise ValueError("incomplete module LOC output")
    rows = [{"path": p, "module": module(p), "lines": m["lines"]}
            for p, m in zip(paths, metrics)]
    rust = [(p, m) for p, m in zip(paths, metrics) if p.endswith(".rs") and m["lines"]]
    payload = "\n".join(json.dumps({"source": sources[p], "excluded": m["excluded"]})
                        for p, m in rust)
    parsed = [json.loads(s) for s in run(str(target / "release/module-map-parser"),
                                       input=payload, text=True).splitlines()]
    if len(parsed) != len(rust):
        raise ValueError("incomplete module dependency output")
    mods = {module(p): {"path": p, **d} for (p, _), d in zip(rust, parsed)}
    exports = {m: {name: m for name in d["definitions"]} for m, d in mods.items()}
    crates = {m.split("::")[0] for m in mods}

    def resolve(m, path):
        bits = path.split("::") if path else []
        here = m.split("::")
        if bits and bits[0] == "crate":
            bits = [here[0]] + bits[1:]
        elif bits and bits[0] == "self":
            bits = here + bits[1:]
        elif bits and bits[0] == "super":
            while bits and bits[0] == "super":
                here, bits = here[:-1], bits[1:]
            bits = here + bits
        elif not bits or bits[0] not in crates:
            bits = here + bits
        for end in range(len(bits), 0, -1):
            prefix = "::".join(bits[:end])
            if prefix in mods:
                return prefix if end == len(bits) else exports[prefix].get(bits[end])
        return None

    # Follow explicit and wildcard reexports to their defining file modules.
    while True:
        changed = False
        for m, d in mods.items():
            for alias, path in d["imports"]:
                target = resolve(m, path)
                if target is None:
                    continue
                additions = exports[target] if alias == "*" else {alias: target}
                for name, owner in list(additions.items()):
                    if name not in exports[m]:
                        exports[m][name] = owner
                        changed = True
        if not changed:
            break

    edges = collections.defaultdict(set)
    for m, d in mods.items():
        for alias, path in d["imports"]:
            target = resolve(m, path)
            if target and target != m:
                edges[m, target].add(f'{d["path"]}: use {path}' + ('::*' if alias == '*' else ''))
        for line, path in d["paths"]:
            target = resolve(m, path) or exports[m].get(path.split("::")[0])
            if target and target != m:
                edges[m, target].add(f'{d["path"]}:{line}: {path}')

    groups, subtrees, direct = collections.Counter(), collections.Counter(), collections.Counter()
    for row in rows:
        bits = row["module"].split("::")
        groups["::".join(bits[:2]) if bits[0] == "ipu_codegen" else bits[0]] += row["lines"]
        direct[row["module"]] += row["lines"]
        for i in range(1, len(bits) + 1):
            subtrees["::".join(bits[:i])] += row["lines"]
    data = {"groups": dict(groups), "files": rows,
            "modules": [{"module": m, "own_loc": direct[m], "subtree_loc": n}
                        for m, n in sorted(subtrees.items())],
            "edges": [{"source": a, "target": b, "evidence": sorted(e)}
                      for (a, b), e in sorted(edges.items())]}
    document = (ASSETS / "template.html").read_text().replace(
        "DATA_JSON", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    OUTPUT.parent.mkdir(exist_ok=True)
    if not OUTPUT.exists() or OUTPUT.read_text() != document:
        OUTPUT.write_text(document)
    print(f"docs/module-map.html: {sum(direct.values()):,} production LOC, {len(edges):,} dependencies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="add the generated report to the index")
    args = parser.parse_args()
    generate()
    if args.stage:
        run("git", "add", "--", str(OUTPUT))
