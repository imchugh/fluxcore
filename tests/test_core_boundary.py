"""Guard the generic-core / TERN boundary (see CLAUDE.md).

Tier-1 core modules must never import TERN-adapter or ops code: the core is
destined for extraction into its own package, where those imports would have
nothing to resolve to. Every TERN-side module lives in a `tern` subpackage,
or in services/network, tasks, tools or ui.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE_PATHS = [
    "domain",
    "infrastructure",
    "services/data",
    "services/metadata/core",
    "services/config_loader.py",
    "orchestration",
]

FORBIDDEN_PREFIXES = ("services.network", "tasks", "tools", "ui", "run")


def _core_files() -> list[Path]:
    files = []
    for rel in CORE_PATHS:
        path = ROOT / rel
        candidates = [path] if path.is_file() else path.rglob("*.py")
        files.extend(f for f in candidates if "tern" not in f.relative_to(ROOT).parts)
    return files


def _is_forbidden(module: str) -> bool:
    if "tern" in module.split("."):
        return True
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in FORBIDDEN_PREFIXES
    )


def _imported_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module
            for alias in node.names:
                yield node.lineno, f"{node.module}.{alias.name}"


class CoreBoundaryTestCase(unittest.TestCase):
    def test_core_files_found(self):
        self.assertGreater(len(_core_files()), 20)

    def test_core_never_imports_tern_or_ops(self):
        violations = []
        for file in _core_files():
            tree = ast.parse(file.read_text(), filename=str(file))
            for lineno, module in _imported_modules(tree):
                if _is_forbidden(module):
                    rel = file.relative_to(ROOT)
                    violations.append(f"{rel}:{lineno} imports {module}")
        self.assertEqual(violations, [], "\n".join(sorted(set(violations))))


if __name__ == "__main__":
    unittest.main()
