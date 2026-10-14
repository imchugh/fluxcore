"""Guard that fluxcore is self-contained.

Every first-party import in the package and its tests must resolve inside
`fluxcore`. Network adapters (such as the TERN-EP pipeline) depend on
fluxcore, never the other way round, so an import of a bare `services`,
`orchestration`, `infrastructure` or `domain` module, or of any network
adapter package, would only work on a machine that happens to have that
adapter on its path.
"""

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ["fluxcore", "tests"]


def _imported_top_levels(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module.split(".")[0]


def _is_allowed(top: str) -> bool:
    if top == "fluxcore" or top in sys.stdlib_module_names:
        return True
    # Anything else must be an installed third-party distribution, not a
    # first-party package living beside the repo.
    return not (ROOT / top).exists() and top not in {
        "domain",
        "infrastructure",
        "services",
        "orchestration",
        "tasks",
        "tools",
        "ui",
        "run",
    }


class SelfContainedTestCase(unittest.TestCase):
    def test_package_found(self):
        self.assertTrue((ROOT / "fluxcore" / "__init__.py").exists())

    def test_first_party_imports_stay_inside_fluxcore(self):
        violations = []
        for rel in SCANNED:
            for file in sorted((ROOT / rel).rglob("*.py")):
                tree = ast.parse(file.read_text(), filename=str(file))
                for lineno, top in _imported_top_levels(tree):
                    if not _is_allowed(top):
                        violations.append(f"{file.relative_to(ROOT)}:{lineno} {top}")
        self.assertEqual(violations, [], "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
