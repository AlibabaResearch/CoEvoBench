"""Layering discipline (design Section 2.2): imports only flow downward.

cli > orchestrate > bench > core > runtime > config, with the service
consumers (research, status, mcp) at the top. Enforced as a test so
the CI gate is the same artifact developers run locally.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "coevobench"

LEVEL = {
    "cli": 0, "orchestrate": 1, "bench": 2, "core": 3, "runtime": 4, "config": 5,
    "research": 0, "status": 1, "mcp": 0,
}
ALLOWED = {
    "cli": {"orchestrate", "bench", "core", "runtime", "config", "research", "status", "mcp"},
    "orchestrate": {"bench", "core", "runtime", "config"},
    "bench": {"core", "runtime", "config"},
    "core": {"runtime", "config"},
    "runtime": {"config"},
    "research": {"bench", "core", "runtime", "config", "cli"},
    "status": {"core", "runtime", "config", "cli"},
    "mcp": {"core", "runtime", "config", "cli", "research", "status", "orchestrate"},
}


def test_no_upward_layer_imports() -> None:
    violations: list[str] = []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT)
        layer = rel.parts[0]
        if layer not in LEVEL:
            continue
        tree = ast.parse(py.read_text())
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module
                    and node.module.startswith("coevobench.")):
                continue
            target = node.module.split(".")[1]
            if target not in LEVEL or target == layer:
                continue
            if target in ALLOWED.get(layer, set()):
                continue
            violations.append(
                f"{rel}:{node.lineno} imports coevobench.{target} "
                f"(layer {layer} -> {target})"
            )
    assert not violations, "\n".join(violations)
