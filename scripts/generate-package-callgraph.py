#!/usr/bin/env python3
"""Generate package build/run call graphs for the ipu-stack workspace.

The extractor prefers rustc MIR because MIR call terminators resolve associated
functions and trait methods.  If a crate does not currently compile, it falls
back to a conservative source-level extractor for that crate.  Compiler-made
closure functions are folded into their containing named Rust function.

The generated full graph contains every reachable workspace-local named
function plus every workspace-local type mentioned by those functions.  The
overview is the function-only prefix within four calls of either entry point.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import html
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


WORKSPACE_PACKAGES = (
    "ipu-codegen",
    "ipu-driver",
    "ipu-elf",
    "ipu-exchange",
    "ipu-package",
    "ipu-profile",
    "ipu-runtime",
)

CRATE_COLORS = {
    "ipu_codegen": "#d9e8fb",
    "ipu_driver": "#f9dfc5",
    "ipu_elf": "#eadcf8",
    "ipu_exchange": "#d8f0dd",
    "ipu_package": "#f7e8ae",
    "ipu_profile": "#e2e2e2",
    "ipu_runtime": "#f7d8e3",
}

CONTROL_WORDS = {
    "as",
    "assert",
    "async",
    "break",
    "const",
    "continue",
    "drop",
    "else",
    "fn",
    "for",
    "if",
    "loop",
    "match",
    "move",
    "return",
    "sizeof",
    "static",
    "while",
}

IGNORED_AMBIGUOUS_TYPES = {"Result", "Error", "IntoIter", "Reader", "Builder"}


@dataclasses.dataclass(eq=False)
class RustFunction:
    key: str
    crate: str
    module: str
    owner: str | None
    name: str
    path: Path
    line: int
    signature: str
    body: str

    @property
    def display(self) -> str:
        owner = f"::{self.owner}" if self.owner else ""
        return f"{self.module}{owner}::{self.name}"


@dataclasses.dataclass(eq=False)
class RustType:
    key: str
    crate: str
    module: str
    name: str
    kind: str
    path: Path
    line: int

    @property
    def display(self) -> str:
        return f"{self.module}::{self.name}"


@dataclasses.dataclass
class Edge:
    callers: set[str] = dataclasses.field(default_factory=set)
    provenance: set[str] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class GraphPartition:
    key: str
    title: str
    crate: str
    functions: set[str] = dataclasses.field(default_factory=set)
    types: set[str] = dataclasses.field(default_factory=set)

    @property
    def filename(self) -> str:
        return f"{self.key}.svg"


def crate_name(package: str) -> str:
    return package.replace("-", "_")


def module_name(root: Path, source: Path, crate: str) -> str:
    relative = source.relative_to(root / "crates" / crate.replace("_", "-") / "src")
    parts = list(relative.parts)
    filename = parts.pop()
    stem = Path(filename).stem
    if stem not in {"lib", "main", "mod"}:
        parts.append(stem)
    if parts and parts[0] == "bin":
        parts = parts[1:]
    return "::".join([crate, *parts])


def sanitize_rust(source: str) -> str:
    """Blank comments and literal contents while retaining offsets/newlines."""
    output = list(source)
    index = 0
    block_depth = 0
    while index < len(source):
        if block_depth:
            if source.startswith("/*", index):
                output[index : index + 2] = "  "
                block_depth += 1
                index += 2
            elif source.startswith("*/", index):
                output[index : index + 2] = "  "
                block_depth -= 1
                index += 2
            else:
                if source[index] != "\n":
                    output[index] = " "
                index += 1
            continue
        if source.startswith("//", index):
            end = source.find("\n", index)
            if end < 0:
                end = len(source)
            for cursor in range(index, end):
                output[cursor] = " "
            index = end
            continue
        if source.startswith("/*", index):
            output[index : index + 2] = "  "
            block_depth = 1
            index += 2
            continue
        if source[index] in {'"', "'"}:
            quote = source[index]
            # A lifetime such as 'a is not a character literal.
            if quote == "'" and index + 1 < len(source) and (
                source[index + 1].isalnum() or source[index + 1] == "_"
            ):
                # Treat 'a and '_ as lifetimes, but retain ordinary character
                # literals such as 'a'.
                if index + 2 >= len(source) or source[index + 2] != "'":
                    index += 1
                    continue
            cursor = index + 1
            while cursor < len(source):
                if source[cursor] == "\\":
                    if source[cursor] != "\n":
                        output[cursor] = " "
                    if cursor + 1 < len(source) and source[cursor + 1] != "\n":
                        output[cursor + 1] = " "
                    cursor += 2
                    continue
                if source[cursor] == quote:
                    break
                if source[cursor] != "\n":
                    output[cursor] = " "
                cursor += 1
            index = min(cursor + 1, len(source))
            continue
        index += 1
    return "".join(output)


def matching_brace(source: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def source_line(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def impl_owner(header: str) -> str | None:
    header = re.sub(r"^\s*impl\s*<[^>{}]*>\s*", "", header.strip())
    header = re.sub(r"^\s*impl\s+", "", header)
    header = header.split(" where ", 1)[0].strip()
    if " for " in header:
        header = header.rsplit(" for ", 1)[1]
    match = re.search(r"(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)", header)
    return match.group(1) if match else None


def enclosing_owner(impls: list[tuple[int, int, str]], offset: int) -> str | None:
    candidates = [(start, owner) for start, end, owner in impls if start < offset < end]
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def discover_source(root: Path) -> tuple[dict[str, RustFunction], dict[str, RustType]]:
    functions: dict[str, RustFunction] = {}
    types: dict[str, RustType] = {}
    for package in WORKSPACE_PACKAGES:
        crate = crate_name(package)
        source_root = root / "crates" / package / "src"
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*.rs")):
            original = path.read_text(encoding="utf-8")
            source = sanitize_rust(original)
            module = module_name(root, path, crate)

            impls: list[tuple[int, int, str]] = []
            impl_pattern = re.compile(r"\bimpl(?:\s*<[^>{}]*>)?\s+([^;{]+)\{")
            for match in impl_pattern.finditer(source):
                opening = source.find("{", match.start(), match.end())
                closing = matching_brace(source, opening)
                owner = impl_owner(source[match.start() : opening])
                if closing is not None and owner:
                    impls.append((opening, closing, owner))

            type_pattern = re.compile(
                r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?"
                r"(struct|enum|union|trait|type)\s+([A-Za-z_]\w*)"
            )
            for match in type_pattern.finditer(source):
                kind, name = match.groups()
                line = source_line(source, match.start())
                key = f"type:{crate}:{path.relative_to(root)}:{line}:{name}"
                types[key] = RustType(key, crate, module, name, kind, path, line)

            function_pattern = re.compile(
                r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?"
                r"(?:(?:async|const|unsafe|extern\s+\"[^\"]+\")\s+)*"
                r"fn\s+([A-Za-z_]\w*)\s*(?:<[^;{}]*?>\s*)?\("
            )
            for match in function_pattern.finditer(source):
                name = match.group(1)
                opening = source.find("{", match.end())
                semicolon = source.find(";", match.end())
                if opening < 0 or (semicolon >= 0 and semicolon < opening):
                    continue
                closing = matching_brace(source, opening)
                if closing is None:
                    continue
                owner = enclosing_owner(impls, match.start())
                line = source_line(source, match.start())
                key = f"fn:{crate}:{path.relative_to(root)}:{line}:{name}"
                signature = re.sub(r"\s+", " ", original[match.start() : opening].strip())
                functions[key] = RustFunction(
                    key,
                    crate,
                    module,
                    owner,
                    name,
                    path,
                    line,
                    signature,
                    source[opening + 1 : closing],
                )
    return functions, types


class Resolver:
    def __init__(self, functions: dict[str, RustFunction]):
        self.functions = functions
        self.by_leaf: dict[str, list[str]] = collections.defaultdict(list)
        self.by_owner_leaf: dict[str, list[str]] = collections.defaultdict(list)
        self.by_display: dict[str, list[str]] = collections.defaultdict(list)
        for key, function in functions.items():
            self.by_leaf[function.name].append(key)
            self.by_display[function.display].append(key)
            if function.owner:
                self.by_owner_leaf[f"{function.owner}::{function.name}"].append(key)

    @staticmethod
    def strip_generics(target: str) -> str:
        result: list[str] = []
        index = 0
        while index < len(target):
            if target.startswith("::<", index):
                depth = 1
                index += 3
                while index < len(target) and depth:
                    if target[index] == "<":
                        depth += 1
                    elif target[index] == ">":
                        depth -= 1
                    index += 1
                continue
            result.append(target[index])
            index += 1
        return "".join(result)

    @staticmethod
    def normalize_target(target: str, caller: RustFunction) -> str:
        target = Resolver.strip_generics(target.strip())
        target = re.sub(r"^<([^<> ]+)(?: as [^>]+)?>::", r"\1::", target)
        target = target.replace("Self::", f"{caller.owner}::" if caller.owner else "")
        target = target.replace("crate::", f"{caller.crate}::")
        while target.startswith("super::"):
            target = target[7:]
        return target

    def resolve(self, target: str, caller: RustFunction, method: bool = False) -> list[str]:
        target = self.normalize_target(target, caller)
        leaf = target.rsplit("::", 1)[-1]
        if leaf in CONTROL_WORDS:
            return []
        candidates: list[str] = []
        if "::" in target:
            candidates = [
                key
                for display, keys in self.by_display.items()
                if display == target or display.endswith(f"::{target}")
                for key in keys
            ]
            if not candidates:
                candidates = list(
                    self.by_owner_leaf.get("::".join(target.split("::")[-2:]), [])
                )
        else:
            candidates = list(self.by_leaf.get(leaf, []))
        if not candidates:
            return []
        same_crate = [key for key in candidates if self.functions[key].crate == caller.crate]
        same_module = [key for key in same_crate if self.functions[key].module == caller.module]
        same_owner = [
            key
            for key in same_crate
            if caller.owner and self.functions[key].owner == caller.owner
        ]
        if same_owner:
            candidates = same_owner
        elif same_module:
            candidates = same_module
        elif same_crate:
            candidates = same_crate
        if method and len(candidates) != 1:
            return []
        # Direct paths should normally be unique. Retain a bounded over-approximation
        # rather than inventing an arbitrary target for duplicate helper names.
        return sorted(set(candidates)) if len(candidates) <= 4 else []


def source_calls(function: RustFunction, resolver: Resolver) -> set[str]:
    result: set[str] = set()
    qualified = re.compile(
        r"(?<![.!])\b((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)"
        r"\s*(?:::\s*<[^;{}()]*>)?\s*\("
    )
    methods = re.compile(r"\.\s*([A-Za-z_]\w*)\s*(?:::\s*<[^;{}()]*>)?\s*\(")
    trait_calls = re.compile(r"<\s*([A-Za-z_]\w*)\s+as\s+[^>]+>::([A-Za-z_]\w*)\s*\(")
    for match in qualified.finditer(function.body):
        result.update(resolver.resolve(match.group(1), function))
    for match in methods.finditer(function.body):
        result.update(resolver.resolve(match.group(1), function, method=True))
    for match in trait_calls.finditer(function.body):
        result.update(resolver.resolve(f"{match.group(1)}::{match.group(2)}", function))
    result.discard(function.key)
    return result


def mir_call_target(line: str) -> str | None:
    marker = " -> ["
    if marker not in line or " = " not in line:
        return None
    expression = line.split(" = ", 1)[1].split(marker, 1)[0].rstrip()
    if not expression.endswith(")"):
        return None
    depth = 0
    for index in range(len(expression) - 1, -1, -1):
        if expression[index] == ")":
            depth += 1
        elif expression[index] == "(":
            depth -= 1
            if depth == 0:
                return expression[:index].strip()
    return None


def mir_definition(raw_name: str, crate: str, functions: dict[str, RustFunction]) -> str | None:
    base = raw_name.split("::{closure#", 1)[0]
    method = re.search(r">::([A-Za-z_]\w*)$", base)
    if method:
        candidates = [
            key
            for key, function in functions.items()
            if function.crate == crate and function.name == method.group(1) and function.owner
        ]
    else:
        leaf = base.rsplit("::", 1)[-1]
        candidates = [
            key
            for key, function in functions.items()
            if function.crate == crate and function.name == leaf
        ]
        qualified = base.rsplit("::", 1)[0] if "::" in base else ""
        if qualified:
            narrowed = [
                key
                for key in candidates
                if functions[key].owner == qualified or functions[key].module.endswith(f"::{qualified}")
            ]
            if narrowed:
                candidates = narrowed
    return candidates[0] if len(candidates) == 1 else None


def run_mir(root: Path, package: str) -> tuple[str | None, str | None]:
    command = [
        "cargo",
        "+nightly",
        "rustc",
        "-q",
        "-p",
        package,
        "--lib",
        "--",
        "-Zunpretty=mir",
    ]
    process = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        summary = next(
            (line.strip() for line in process.stderr.splitlines() if line.startswith("error")),
            f"cargo exited {process.returncode}",
        )
        return None, summary
    return process.stdout, None


def mir_calls(
    mir: str,
    crate: str,
    functions: dict[str, RustFunction],
    resolver: Resolver,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = collections.defaultdict(set)
    current_raw: str | None = None
    current_key: str | None = None
    for line in mir.splitlines():
        if line.startswith("fn "):
            current_raw = line[3:].split("(", 1)[0]
            current_key = mir_definition(current_raw, crate, functions)
            if current_key is not None:
                result.setdefault(current_key, set())
            continue
        if line.startswith("const ") or (line == "}" and current_raw is not None):
            if line == "}":
                current_raw = None
                current_key = None
            continue
        if current_key is None:
            continue
        target = mir_call_target(line)
        if target is None:
            continue
        caller = functions[current_key]
        for callee in resolver.resolve(target, caller):
            if callee != current_key:
                result[current_key].add(callee)
    return result


def type_uses(
    function: RustFunction,
    types: dict[str, RustType],
) -> set[str]:
    text = f"{function.signature}\n{function.body}"
    by_name: dict[str, list[str]] = collections.defaultdict(list)
    for key, item in types.items():
        by_name[item.name].append(key)
    result: set[str] = set()
    for name, candidates in by_name.items():
        if name in IGNORED_AMBIGUOUS_TYPES or not re.search(rf"\b{re.escape(name)}\b", text):
            continue
        same_module = [key for key in candidates if types[key].module == function.module]
        same_crate = [key for key in candidates if types[key].crate == function.crate]
        if len(same_module) == 1:
            result.add(same_module[0])
        elif len(same_crate) == 1:
            result.add(same_crate[0])
        elif len(candidates) == 1:
            result.add(candidates[0])
    return result


def find_function(functions: dict[str, RustFunction], suffix: str) -> str:
    matches = [key for key, function in functions.items() if function.display.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one function ending {suffix!r}, found {len(matches)}")
    return matches[0]


def reachable(
    roots: Iterable[str],
    calls: dict[str, set[str]],
) -> tuple[set[str], dict[str, int]]:
    seen: set[str] = set()
    depth: dict[str, int] = {}
    queue = collections.deque((root, 1) for root in roots)
    while queue:
        function, current_depth = queue.popleft()
        if function in seen:
            depth[function] = min(depth[function], current_depth)
            continue
        seen.add(function)
        depth[function] = current_depth
        for callee in calls.get(function, set()):
            if callee not in seen:
                queue.append((callee, current_depth + 1))
    return seen, depth


def dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def relative_source(root: Path, path: Path, line: int) -> str:
    return f"{path.relative_to(root)}:{line}"


def write_dot(
    output: Path,
    title: str,
    root: Path,
    functions: dict[str, RustFunction],
    types: dict[str, RustType],
    selected_functions: set[str],
    selected_types: set[str],
    calls: dict[str, set[str]],
    edge_provenance: dict[tuple[str, str], set[str]],
    build_reachable: set[str],
    run_reachable: set[str],
    type_edges: dict[str, set[str]],
    include_types: bool,
) -> None:
    function_ids = {key: f"f{index}" for index, key in enumerate(sorted(selected_functions))}
    type_ids = {key: f"t{index}" for index, key in enumerate(sorted(selected_types))}
    lines = [
        "digraph ipu_stack_package_flow {",
        '  graph [rankdir=LR, bgcolor="white", fontname="DejaVu Sans", fontsize=18,',
        f'         label="{dot_escape(title)}", labelloc=t, labeljust=l, pad=0.25, '
        'nodesep=0.35, ranksep=0.9, newrank=true, overlap=false, splines=polyline, outputorder=edgesfirst];',
        '  node [fontname="DejaVu Sans", fontsize=9, style="rounded,filled", color="#52606d", penwidth=0.8];',
        '  edge [fontname="DejaVu Sans", fontsize=7, color="#52606d", arrowsize=0.55, penwidth=0.8];',
        '  build_entry [label="package build entry", shape=octagon, fillcolor="#cfe2ff", penwidth=1.4];',
        '  run_entry [label="ipu_cli::main\\nCommand::HostRun arm", shape=octagon, fillcolor="#d5f5e3", penwidth=1.4];',
    ]

    for crate in sorted({functions[key].crate for key in selected_functions} | {types[key].crate for key in selected_types}):
        color = CRATE_COLORS.get(crate, "#eeeeee")
        lines.append(f'  subgraph cluster_{crate} {{ label="{crate}"; color="{color}"; style="rounded";')
        for key in sorted(selected_functions, key=lambda item: functions[item].display):
            function = functions[key]
            if function.crate != crate:
                continue
            source = relative_source(root, function.path, function.line)
            tooltip = f"{function.signature} — {source}"
            if key in build_reachable and key in run_reachable:
                border = "#7d3c98"
            elif key in build_reachable:
                border = "#2166ac"
            else:
                border = "#238b45"
            lines.append(
                f'    {function_ids[key]} [label="{dot_escape(function.display)}", shape=box, '
                f'fillcolor="{color}", color="{border}", tooltip="{dot_escape(tooltip)}"];'
            )
        if include_types:
            for key in sorted(selected_types, key=lambda item: types[item].display):
                rust_type = types[key]
                if rust_type.crate != crate:
                    continue
                source = relative_source(root, rust_type.path, rust_type.line)
                lines.append(
                    f'    {type_ids[key]} [label="{dot_escape(rust_type.display)}\\n«{rust_type.kind}»", '
                    f'shape=ellipse, style="filled,dashed", fillcolor="{color}", color="#7b8794", '
                    f'tooltip="{dot_escape(source)}"];'
                )
        lines.append("  }")

    build_roots = [key for key in selected_functions if functions[key].display.endswith("::build_package")]
    write_roots = [key for key in selected_functions if functions[key].display.endswith("::Application::write")]
    run_suffixes = (
        "::Application::read",
        "::Runtime::open",
        "::Runtime::load",
        "::Runtime::host_session",
        "::HostSession::start",
        "::HostSession::invoke",
    )
    for key in build_roots + write_roots:
        lines.append(f"  build_entry -> {function_ids[key]} [color=\"#2166ac\", penwidth=1.4];")
    for key in selected_functions:
        if functions[key].display.endswith(run_suffixes):
            lines.append(f"  run_entry -> {function_ids[key]} [color=\"#238b45\", penwidth=1.4];")

    for caller in sorted(selected_functions):
        for callee in sorted(calls.get(caller, set())):
            if callee not in selected_functions:
                continue
            paths = edge_provenance.get((caller, callee), {"source"})
            approximate = paths == {"source"}
            if caller in build_reachable and callee in build_reachable and caller in run_reachable and callee in run_reachable:
                color = "#7d3c98"
            elif caller in build_reachable and callee in build_reachable:
                color = "#2166ac"
            elif caller in run_reachable and callee in run_reachable:
                color = "#238b45"
            else:
                color = "#52606d"
            style = "dotted" if approximate else "solid"
            tooltip = "source fallback (conservative)" if approximate else "rustc MIR call edge"
            lines.append(
                f'  {function_ids[caller]} -> {function_ids[callee]} [color="{color}", '
                f'style="{style}", tooltip="{tooltip}"];'
            )
        if include_types:
            for type_key in sorted(type_edges.get(caller, set())):
                if type_key in selected_types:
                    lines.append(
                        f'  {function_ids[caller]} -> {type_ids[type_key]} '
                        '[style=dashed, color="#9aa5b1", arrowhead=none, constraint=false, tooltip="uses workspace type"];'
                    )

    lines.extend(
        [
            '  legend [shape=note, style="filled", fillcolor="#ffffff", color="#9aa5b1", fontsize=8,',
            '          label="Solid edge: rustc MIR-resolved call\\nDotted edge: source fallback\\nDashed edge: function uses workspace type\\nBlue: build path   Green: run path   Purple: shared"];',
            "}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(dot_path: Path, svg_path: Path, engine: str) -> None:
    subprocess.run([engine, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)


def file_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def make_partitions(
    functions: dict[str, RustFunction],
    types: dict[str, RustType],
    selected_functions: set[str],
    selected_types: set[str],
    max_primary_nodes: int = 100,
) -> tuple[dict[str, GraphPartition], dict[str, str], dict[str, str]]:
    """Partition primary nodes by source module, then cap oversized modules."""
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = collections.defaultdict(list)
    for key in selected_functions:
        function = functions[key]
        grouped[(function.crate, function.module)].append(("function", key))
    for key in selected_types:
        rust_type = types[key]
        grouped[(rust_type.crate, rust_type.module)].append(("type", key))

    partitions: dict[str, GraphPartition] = {}
    function_partition: dict[str, str] = {}
    type_partition: dict[str, str] = {}
    for (crate, module), nodes in sorted(grouped.items()):
        nodes.sort(
            key=lambda item: (
                item[0],
                functions[item[1]].display if item[0] == "function" else types[item[1]].display,
            )
        )
        chunks = [nodes[index : index + max_primary_nodes] for index in range(0, len(nodes), max_primary_nodes)]
        for chunk_index, chunk in enumerate(chunks, start=1):
            suffix = "" if len(chunks) == 1 else f"-part-{chunk_index}"
            key = f"{file_slug(module)}{suffix}"
            title = module if len(chunks) == 1 else f"{module} — part {chunk_index}/{len(chunks)}"
            partition = GraphPartition(key=key, title=title, crate=crate)
            for kind, node_key in chunk:
                if kind == "function":
                    partition.functions.add(node_key)
                    function_partition[node_key] = key
                else:
                    partition.types.add(node_key)
                    type_partition[node_key] = key
            partitions[key] = partition
    return partitions, function_partition, type_partition


def call_edge_attributes(
    caller: str,
    callee: str,
    edge_provenance: dict[tuple[str, str], set[str]],
    build_reachable: set[str],
    run_reachable: set[str],
) -> tuple[str, str, str]:
    paths = edge_provenance.get((caller, callee), {"source"})
    approximate = paths == {"source"}
    if caller in build_reachable and callee in build_reachable and caller in run_reachable and callee in run_reachable:
        color = "#7d3c98"
    elif caller in build_reachable and callee in build_reachable:
        color = "#2166ac"
    elif caller in run_reachable and callee in run_reachable:
        color = "#238b45"
    else:
        color = "#52606d"
    style = "dotted" if approximate else "solid"
    tooltip = "source fallback (conservative)" if approximate else "rustc MIR call edge"
    return color, style, tooltip


def write_partition_dot(
    output: Path,
    root: Path,
    partition: GraphPartition,
    functions: dict[str, RustFunction],
    types: dict[str, RustType],
    all_functions: set[str],
    calls: dict[str, set[str]],
    edge_provenance: dict[tuple[str, str], set[str]],
    build_reachable: set[str],
    run_reachable: set[str],
    type_edges: dict[str, set[str]],
    function_partition: dict[str, str],
    type_partition: dict[str, str],
) -> None:
    function_ids = {key: f"f{index}" for index, key in enumerate(sorted(partition.functions))}
    type_ids = {key: f"t{index}" for index, key in enumerate(sorted(partition.types))}
    external_functions = {
        callee
        for caller in partition.functions
        for callee in calls.get(caller, set())
        if callee in all_functions and callee not in partition.functions
    }
    external_types = {
        type_key
        for caller in partition.functions
        for type_key in type_edges.get(caller, set())
        if type_key not in partition.types
    }
    external_function_ids = {key: f"xf{index}" for index, key in enumerate(sorted(external_functions))}
    external_type_ids = {key: f"xt{index}" for index, key in enumerate(sorted(external_types))}
    color = CRATE_COLORS.get(partition.crate, "#eeeeee")
    lines = [
        "digraph ipu_stack_package_partition {",
        '  graph [rankdir=LR, bgcolor="white", fontname="DejaVu Sans", fontsize=17,',
        f'         label="{dot_escape(partition.title)} — package build/run detail", labelloc=t, labeljust=l, '
        'pad=0.2, nodesep=0.3, ranksep=0.75, newrank=true, overlap=false, splines=polyline, outputorder=edgesfirst];',
        '  node [fontname="DejaVu Sans", fontsize=9, style="rounded,filled", color="#52606d", penwidth=0.8];',
        '  edge [fontname="DejaVu Sans", fontsize=7, color="#52606d", arrowsize=0.55, penwidth=0.8];',
    ]

    if any(functions[key].display.endswith(("::build_package", "::Application::write")) for key in partition.functions):
        lines.append('  build_entry [label="package build entry", shape=octagon, fillcolor="#cfe2ff", penwidth=1.4];')
    run_suffixes = (
        "::Application::read",
        "::Runtime::open",
        "::Runtime::load",
        "::Runtime::host_session",
        "::HostSession::start",
        "::HostSession::invoke",
    )
    if any(functions[key].display.endswith(run_suffixes) for key in partition.functions):
        lines.append('  run_entry [label="ipu_cli::main\\nCommand::HostRun arm", shape=octagon, fillcolor="#d5f5e3", penwidth=1.4];')

    lines.append(f'  subgraph cluster_primary {{ label="primary nodes"; color="{color}"; style="rounded";')
    for key in sorted(partition.functions, key=lambda item: functions[item].display):
        function = functions[key]
        source = relative_source(root, function.path, function.line)
        tooltip = f"{function.signature} — {source}"
        if key in build_reachable and key in run_reachable:
            border = "#7d3c98"
        elif key in build_reachable:
            border = "#2166ac"
        else:
            border = "#238b45"
        lines.append(
            f'    {function_ids[key]} [label="{dot_escape(function.display)}", shape=box, '
            f'fillcolor="{color}", color="{border}", tooltip="{dot_escape(tooltip)}"];'
        )
    for key in sorted(partition.types, key=lambda item: types[item].display):
        rust_type = types[key]
        source = relative_source(root, rust_type.path, rust_type.line)
        lines.append(
            f'    {type_ids[key]} [label="{dot_escape(rust_type.display)}\\n«{rust_type.kind}»", '
            f'shape=ellipse, style="filled,dashed", fillcolor="{color}", color="#7b8794", '
            f'tooltip="{dot_escape(source)}"];'
        )
    lines.append("  }")

    if external_functions or external_types:
        lines.append('  subgraph cluster_external { label="outgoing references — click to open target partition"; color="#d9d9d9"; style="rounded,dashed";')
        for key in sorted(external_functions, key=lambda item: functions[item].display):
            function = functions[key]
            target = function_partition[key]
            lines.append(
                f'    {external_function_ids[key]} [label="{dot_escape(function.display)}", shape=box, '
                f'fillcolor="#f5f5f5", color="#9aa5b1", style="rounded,filled,dashed", '
                f'URL="{dot_escape(target)}.svg", target="_top", tooltip="open target partition"];'
            )
        for key in sorted(external_types, key=lambda item: types[item].display):
            rust_type = types[key]
            target = type_partition[key]
            lines.append(
                f'    {external_type_ids[key]} [label="{dot_escape(rust_type.display)}\\n«{rust_type.kind}»", shape=ellipse, '
                f'fillcolor="#f5f5f5", color="#9aa5b1", style="filled,dashed", '
                f'URL="{dot_escape(target)}.svg", target="_top", tooltip="open defining partition"];'
            )
        lines.append("  }")

    for key in partition.functions:
        if functions[key].display.endswith(("::build_package", "::Application::write")):
            lines.append(f"  build_entry -> {function_ids[key]} [color=\"#2166ac\", penwidth=1.4];")
        if functions[key].display.endswith(run_suffixes):
            lines.append(f"  run_entry -> {function_ids[key]} [color=\"#238b45\", penwidth=1.4];")

    for caller in sorted(partition.functions):
        for callee in sorted(calls.get(caller, set())):
            if callee not in all_functions:
                continue
            target_id = function_ids.get(callee, external_function_ids.get(callee))
            if target_id is None:
                continue
            edge_color, style, tooltip = call_edge_attributes(
                caller, callee, edge_provenance, build_reachable, run_reachable
            )
            lines.append(
                f'  {function_ids[caller]} -> {target_id} [color="{edge_color}", style="{style}", '
                f'tooltip="{tooltip}"];'
            )
        for type_key in sorted(type_edges.get(caller, set())):
            target_id = type_ids.get(type_key, external_type_ids.get(type_key))
            if target_id is not None:
                lines.append(
                    f'  {function_ids[caller]} -> {target_id} '
                    '[style=dashed, color="#9aa5b1", arrowhead=none, constraint=false, tooltip="uses workspace type"];'
                )
    lines.extend(
        [
            '  legend [shape=note, style="filled", fillcolor="#ffffff", color="#9aa5b1", fontsize=8,',
            '          label="Solid: MIR-resolved call   Dotted: source fallback\\nDashed, no arrow: uses type   Dashed node: link to another partition\\nBlue: build   Green: run   Purple: shared"];',
            "}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def partition_dependencies(
    partitions: dict[str, GraphPartition],
    function_partition: dict[str, str],
    type_partition: dict[str, str],
    calls: dict[str, set[str]],
    type_edges: dict[str, set[str]],
) -> dict[tuple[str, str], tuple[int, int]]:
    counts: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for source_key, partition in function_partition.items():
        for callee in calls.get(source_key, set()):
            target = function_partition.get(callee)
            if target is not None and target != partition:
                counts[(partition, target)][0] += 1
        for type_key in type_edges.get(source_key, set()):
            target = type_partition.get(type_key)
            if target is not None and target != partition:
                counts[(partition, target)][1] += 1
    return {key: (value[0], value[1]) for key, value in counts.items()}


def write_partition_map_dot(
    output: Path,
    partitions: dict[str, GraphPartition],
    dependencies: dict[tuple[str, str], tuple[int, int]],
) -> None:
    partition_modules = {
        key: re.sub(r" — part \d+/\d+$", "", partition.title)
        for key, partition in partitions.items()
    }
    module_partitions: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for key, partition in partitions.items():
        module_partitions[(partition.crate, partition_modules[key])].append(key)
    module_dependencies: dict[tuple[tuple[str, str], tuple[str, str]], list[int]] = collections.defaultdict(
        lambda: [0, 0]
    )
    for (source, target), (call_count, type_count) in dependencies.items():
        source_module = (partitions[source].crate, partition_modules[source])
        target_module = (partitions[target].crate, partition_modules[target])
        if source_module != target_module:
            module_dependencies[(source_module, target_module)][0] += call_count
            module_dependencies[(source_module, target_module)][1] += type_count
    node_ids = {key: f"p{index}" for index, key in enumerate(sorted(module_partitions))}
    lines = [
        "digraph ipu_stack_package_partition_map {",
        '  graph [rankdir=LR, bgcolor="white", fontname="DejaVu Sans", fontsize=18,',
        '         label="IPU stack package build/run — full graph partition map", labelloc=t, labeljust=l, pad=0.25, nodesep=0.4, ranksep=1.0, overlap=false, splines=polyline, concentrate=true, outputorder=edgesfirst];',
        '  node [shape=box, fontname="DejaVu Sans", fontsize=9, style="rounded,filled", color="#52606d", penwidth=0.8];',
        '  edge [fontname="DejaVu Sans", fontsize=7, color="#8a94a0", arrowsize=0.55, penwidth=0.7];',
    ]
    for crate in sorted({partition.crate for partition in partitions.values()}):
        color = CRATE_COLORS.get(crate, "#eeeeee")
        lines.append(f'  subgraph cluster_{crate} {{ label="{crate}"; color="{color}"; style="rounded";')
        for key, member_keys in sorted(module_partitions.items()):
            module_crate, module = key
            if module_crate != crate:
                continue
            function_count = sum(len(partitions[member].functions) for member in member_keys)
            type_count = sum(len(partitions[member].types) for member in member_keys)
            page_count = len(member_keys)
            counts = f"{function_count} functions, {type_count} types, {page_count} detail page{'s' if page_count != 1 else ''}"
            anchor = f"module-{file_slug(module)}"
            lines.append(
                f'    {node_ids[key]} [label="{dot_escape(module)}\\n{counts}", fillcolor="{color}", '
                f'URL="ipu-stack-package-callgraph/index.html#{anchor}", target="_top", tooltip="open module detail list"];'
            )
        lines.append("  }")
    for (source, target), (call_count, type_count) in sorted(module_dependencies.items()):
        label_parts = []
        if call_count:
            label_parts.append(f"{call_count} calls")
        if type_count:
            label_parts.append(f"{type_count} type uses")
        label = ", ".join(label_parts)
        lines.append(f'  {node_ids[source]} -> {node_ids[target]} [tooltip="{label}"];')
    lines.extend(
        [
            '  help [shape=note, fillcolor="#ffffff", color="#9aa5b1", label="Each box opens that module in the lightweight index.\\nThe raw monolithic DOT remains available for tools."];',
            "}",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index_html(
    output: Path,
    partitions: dict[str, GraphPartition],
    dependencies: dict[tuple[str, str], tuple[int, int]],
    build_count: int,
    run_count: int,
    function_count: int,
    type_count: int,
    mir_status: dict[str, str],
) -> None:
    incoming: dict[str, int] = collections.Counter(target for _, target in dependencies)
    outgoing: dict[str, int] = collections.Counter(source for source, _ in dependencies)
    rows = []
    seen_modules: set[str] = set()
    for key, partition in sorted(partitions.items(), key=lambda item: (item[1].crate, item[1].title, item[0])):
        module = re.sub(r" — part \d+/\d+$", "", partition.title)
        row_id = ""
        if module not in seen_modules:
            row_id = f' id="module-{file_slug(module)}"'
            seen_modules.add(module)
        rows.append(
            f"<tr{row_id}>"
            f"<td><code>{html.escape(partition.crate)}</code></td>"
            f'<td><a href="{html.escape(partition.filename)}">{html.escape(partition.title)}</a></td>'
            f"<td>{len(partition.functions)}</td><td>{len(partition.types)}</td>"
            f"<td>{outgoing.get(key, 0)}</td><td>{incoming.get(key, 0)}</td>"
            f'<td><a href="{html.escape(key)}.dot">DOT</a></td>'
            "</tr>"
        )
    statuses = "".join(
        f"<li><code>{html.escape(crate)}</code>: {html.escape(status)}</li>" for crate, status in sorted(mir_status.items())
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IPU stack package build/run call graph</title>
<style>
  :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
  body {{ max-width: 1120px; margin: 2rem auto; padding: 0 1rem 3rem; line-height: 1.45; }}
  nav a {{ margin-right: 1.2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border-bottom: 1px solid #9996; padding: .5rem .65rem; text-align: left; }}
  th {{ position: sticky; top: 0; background: Canvas; }}
  td:nth-child(n+3):nth-child(-n+6), th:nth-child(n+3):nth-child(-n+6) {{ text-align: right; }}
  code {{ font-size: .9em; }}
  .summary {{ padding: .8rem 1rem; border: 1px solid #9996; }}
</style>
</head>
<body>
<h1>IPU stack package build/run call graph</h1>
<p class="summary"><strong>{function_count}</strong> reachable workspace functions and
<strong>{type_count}</strong> involved workspace types, split into <strong>{len(partitions)}</strong> bounded SVGs.
Build reaches {build_count} functions; run reaches {run_count}.</p>
<nav>
  <a href="../ipu-stack-package-callgraph-full.svg">Partition map</a>
  <a href="../ipu-stack-package-callgraph-overview.svg">Overview</a>
  <a href="../ipu-stack-package-callgraph-full.dot">Raw full DOT</a>
</nav>
<table>
<thead><tr><th>Crate</th><th>Module / partition</th><th>Functions</th><th>Types</th><th>Outgoing partitions</th><th>Incoming partitions</th><th>Source</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<details><summary>Extraction status</summary><ul>{statuses}</ul></details>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--no-mir", action="store_true", help="use source extraction for every crate")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output_directory = (arguments.output_directory or root / "docs").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    functions, types = discover_source(root)
    resolver = Resolver(functions)
    calls: dict[str, set[str]] = collections.defaultdict(set)
    edge_provenance: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    source_by_crate: dict[str, dict[str, set[str]]] = collections.defaultdict(dict)
    for key, function in functions.items():
        source_by_crate[function.crate][key] = source_calls(function, resolver)

    mir_status: dict[str, str] = {}
    for package in WORKSPACE_PACKAGES:
        crate = crate_name(package)
        extracted: dict[str, set[str]] | None = None
        if not arguments.no_mir:
            mir, error = run_mir(root, package)
            if mir is not None:
                extracted = mir_calls(mir, crate, functions, resolver)
                mir_status[crate] = "MIR"
            else:
                mir_status[crate] = f"source fallback: {error}"
        else:
            mir_status[crate] = "source fallback: --no-mir"
        crate_edges: dict[str, set[str]] = {}
        crate_provenance: dict[str, str] = {}
        for caller, source_callees in source_by_crate[crate].items():
            if extracted is not None and caller in extracted:
                crate_edges[caller] = extracted[caller]
                crate_provenance[caller] = "mir"
            else:
                crate_edges[caller] = source_callees
                crate_provenance[caller] = "source"
        for caller, callees in crate_edges.items():
            calls[caller].update(callees)
            provenance = crate_provenance[caller]
            for callee in callees:
                edge_provenance[(caller, callee)].add(provenance)

    build_roots = {
        find_function(functions, "ipu_codegen::package::build_package"),
        find_function(functions, "ipu_package::Application::write"),
    }
    run_roots = {
        find_function(functions, "ipu_package::Application::read"),
        find_function(functions, "ipu_runtime::Runtime::open"),
        find_function(functions, "ipu_runtime::Runtime::load"),
        find_function(functions, "ipu_runtime::Runtime::host_session"),
        find_function(functions, "ipu_driver::HostSession::start"),
        find_function(functions, "ipu_driver::HostSession::invoke"),
    }
    build_functions, build_depth = reachable(build_roots, calls)
    run_functions, run_depth = reachable(run_roots, calls)
    all_functions = build_functions | run_functions

    all_type_edges = {key: type_uses(functions[key], types) for key in all_functions}
    all_types = set().union(*all_type_edges.values()) if all_type_edges else set()

    overview_functions = {
        key
        for key in all_functions
        if min(build_depth.get(key, 999), run_depth.get(key, 999)) <= 4
    }
    overview_dot = output_directory / "ipu-stack-package-callgraph-overview.dot"
    overview_svg = output_directory / "ipu-stack-package-callgraph-overview.svg"
    full_dot = output_directory / "ipu-stack-package-callgraph-full.dot"
    full_svg = output_directory / "ipu-stack-package-callgraph-full.svg"
    map_dot = output_directory / "ipu-stack-package-callgraph-map.dot"
    map_svg = output_directory / "ipu-stack-package-callgraph-map.svg"
    partition_directory = output_directory / "ipu-stack-package-callgraph"
    partition_directory.mkdir(parents=True, exist_ok=True)

    write_dot(
        overview_dot,
        f"IPU stack package build/run call graph — overview ({len(overview_functions)} functions)",
        root,
        functions,
        types,
        overview_functions,
        set(),
        calls,
        edge_provenance,
        build_functions,
        run_functions,
        {},
        False,
    )
    write_dot(
        full_dot,
        f"IPU stack package build/run call graph — full ({len(all_functions)} functions, {len(all_types)} types)",
        root,
        functions,
        types,
        all_functions,
        all_types,
        calls,
        edge_provenance,
        build_functions,
        run_functions,
        all_type_edges,
        True,
    )
    render(overview_dot, overview_svg, "dot")

    partitions, function_partition, type_partition = make_partitions(
        functions, types, all_functions, all_types
    )
    expected_partition_files = {"index.html"}
    for partition in partitions.values():
        expected_partition_files.update({f"{partition.key}.dot", partition.filename})
    for existing in partition_directory.iterdir():
        if (
            existing.is_file()
            and existing.suffix in {".dot", ".svg"}
            and existing.name not in expected_partition_files
        ):
            existing.unlink()
    for partition in partitions.values():
        partition_dot = partition_directory / f"{partition.key}.dot"
        partition_svg = partition_directory / partition.filename
        write_partition_dot(
            partition_dot,
            root,
            partition,
            functions,
            types,
            all_functions,
            calls,
            edge_provenance,
            build_functions,
            run_functions,
            all_type_edges,
            function_partition,
            type_partition,
        )
        render(partition_dot, partition_svg, "dot")

    dependencies = partition_dependencies(
        partitions, function_partition, type_partition, calls, all_type_edges
    )
    write_partition_map_dot(map_dot, partitions, dependencies)
    render(map_dot, map_svg, "dot")
    # Keep the old full-SVG path useful and browser-safe: it is now the small
    # clickable partition map.  The complete monolithic graph remains as DOT.
    render(map_dot, full_svg, "dot")
    index_html = partition_directory / "index.html"
    write_index_html(
        index_html,
        partitions,
        dependencies,
        len(build_functions),
        len(run_functions),
        len(all_functions),
        len(all_types),
        mir_status,
    )

    print(f"overview: {overview_dot.relative_to(root)} -> {overview_svg.relative_to(root)}")
    print(f"full raw graph: {full_dot.relative_to(root)}")
    print(f"partition map: {map_dot.relative_to(root)} -> {map_svg.relative_to(root)}")
    print(f"browser index: {index_html.relative_to(root)}")
    print(f"detail partitions: {len(partitions)}")
    print(f"reachable functions: build={len(build_functions)} run={len(run_functions)} union={len(all_functions)}")
    print(f"reachable workspace types: {len(all_types)}")
    for crate, status in sorted(mir_status.items()):
        print(f"{crate}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
