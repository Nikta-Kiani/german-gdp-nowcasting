"""Read-only preflight checks for package and script entry points."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from german_gdp_nowcasting.config import paths


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    comparison = node.test
    return (
        isinstance(comparison.left, ast.Name)
        and comparison.left.id == "__name__"
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.Eq)
        and len(comparison.comparators) == 1
        and isinstance(comparison.comparators[0], ast.Constant)
        and comparison.comparators[0].value == "__main__"
    )


class ScriptPreflightTests(unittest.TestCase):
    def test_all_scripts_parse_and_guard_model_loops(self) -> None:
        scripts = sorted(SCRIPTS_DIR.rglob("*.py"))
        self.assertGreater(len(scripts), 0)

        for script in scripts:
            with self.subTest(script=script.relative_to(REPO_ROOT)):
                source = script.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(script))
                self.assertTrue(
                    any(_is_main_guard(node) for node in tree.body),
                    "script must protect execution behind an __main__ guard",
                )

    def test_canonical_package_paths_resolve_from_repository(self) -> None:
        self.assertEqual(paths.REPO_ROOT, REPO_ROOT)
        self.assertTrue((REPO_ROOT / "src" / "german_gdp_nowcasting").is_dir())
        self.assertTrue(SCRIPTS_DIR.is_dir())
        self.assertEqual(paths.DATA_PANEL, paths.DATA / "panel")
        self.assertEqual(paths.DATA_METADATA, paths.DATA / "metadata")
        self.assertEqual(paths.OUT_NOWCASTING, paths.OUTPUTS / "nowcasting")
        self.assertEqual(
            paths.EN_ONLY_MATRIX_CSV.parent,
            paths.OUT_INDICATOR_SELECTION / "dfm_input_sets",
        )

    def test_notebook_path_constants_exist(self) -> None:
        for notebook in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
            names = set(re.findall(r"_tp\.([A-Z][A-Z0-9_]*)", notebook.read_text()))
            with self.subTest(notebook=notebook.name):
                missing = sorted(name for name in names if not hasattr(paths, name))
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
