"""Configuration and package-import smoke tests."""

from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
import unittest
from pathlib import Path
from unittest import mock

import german_gdp_nowcasting
from german_gdp_nowcasting.config import paths


REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_ENV_VARS = (
    "GERMAN_GDP_NOWCASTING_DATA_DIR",
    "GERMAN_GDP_NOWCASTING_OUTPUTS_DIR",
    "GERMAN_GDP_NOWCASTING_DATASET_XLSX",
    "GERMAN_GDP_NOWCASTING_SUPERVISOR_EXPORTS_DIR",
    "GERMAN_GDP_NOWCASTING_SUPERVISOR_KEPT_XLSX",
    "GERMAN_GDP_NOWCASTING_SUPERVISOR_DROPPED_XLSX",
)

# These modules deliberately depend on optional, computationally heavy extras.
OPTIONAL_IMPORTS = {
    "german_gdp_nowcasting.models.dfm.dfm_sv_bayes": ("jax", "numpyro"),
    "german_gdp_nowcasting.models.dfm.dfm_sv_integrated": ("jax", "numpyro"),
    "german_gdp_nowcasting.models.xgboost.xgb_utils": ("xgboost",),
    "german_gdp_nowcasting.visualization.mlp_plots": ("matplotlib",),
    "german_gdp_nowcasting.visualization.nowcast_plots": ("matplotlib",),
    "german_gdp_nowcasting.visualization.xgb_plots": ("matplotlib",),
}


class ConfigPathTests(unittest.TestCase):
    def tearDown(self) -> None:
        """Leave the shared paths module in its normal process environment."""
        importlib.reload(paths)

    def test_default_paths_use_clean_repository_layout(self) -> None:
        with mock.patch.dict(os.environ, {name: "" for name in PATH_ENV_VARS}):
            importlib.reload(paths)

            self.assertEqual(paths.REPO_ROOT, REPO_ROOT)
            self.assertEqual(paths.ROOT, REPO_ROOT)
            self.assertEqual(paths.PROJECT_FILES, REPO_ROOT)
            self.assertEqual(paths.DATA, REPO_ROOT / "data")
            self.assertEqual(paths.OUTPUTS, REPO_ROOT / "outputs")
            self.assertEqual(
                paths.DATASET_XLSX,
                REPO_ROOT / "Dataset" / "ifoCAST_DATA.xlsx",
            )
            self.assertEqual(
                paths.SUPERVISOR_EXPORTS,
                REPO_ROOT / "outputs" / "supervisor_exports",
            )
            self.assertEqual(
                paths.SUPERVISOR_KEPT_XLSX,
                REPO_ROOT / "outputs" / "supervisor_exports"
                / "indicators_kept_review.xlsx",
            )
            self.assertEqual(
                paths.SUPERVISOR_DROPPED_XLSX,
                REPO_ROOT / "outputs" / "supervisor_exports"
                / "indicators_dropped_review.xlsx",
            )

    def test_environment_overrides_are_resolved_and_propagated(self) -> None:
        overrides = {
            "GERMAN_GDP_NOWCASTING_DATA_DIR": str(REPO_ROOT / "tests" / "data_override"),
            "GERMAN_GDP_NOWCASTING_OUTPUTS_DIR": str(
                REPO_ROOT / "tests" / "outputs_override"
            ),
            "GERMAN_GDP_NOWCASTING_DATASET_XLSX": str(
                REPO_ROOT / "tests" / "synthetic_dataset.xlsx"
            ),
            "GERMAN_GDP_NOWCASTING_SUPERVISOR_EXPORTS_DIR": str(
                REPO_ROOT / "tests" / "supervisor_override"
            ),
            "GERMAN_GDP_NOWCASTING_SUPERVISOR_KEPT_XLSX": str(
                REPO_ROOT / "tests" / "kept_override.xlsx"
            ),
            "GERMAN_GDP_NOWCASTING_SUPERVISOR_DROPPED_XLSX": str(
                REPO_ROOT / "tests" / "dropped_override.xlsx"
            ),
        }
        with mock.patch.dict(os.environ, overrides):
            importlib.reload(paths)

            self.assertEqual(paths.DATA, Path(overrides[PATH_ENV_VARS[0]]).resolve())
            self.assertEqual(
                paths.DATA_PANEL,
                Path(overrides[PATH_ENV_VARS[0]]).resolve() / "panel",
            )
            self.assertEqual(paths.OUTPUTS, Path(overrides[PATH_ENV_VARS[1]]).resolve())
            self.assertEqual(
                paths.OUT_NOWCASTING,
                Path(overrides[PATH_ENV_VARS[1]]).resolve() / "nowcasting",
            )
            self.assertEqual(
                paths.DATASET_XLSX,
                Path(overrides[PATH_ENV_VARS[2]]).resolve(),
            )
            self.assertEqual(
                paths.SUPERVISOR_EXPORTS,
                Path(overrides[PATH_ENV_VARS[3]]).resolve(),
            )
            self.assertEqual(
                paths.SUPERVISOR_KEPT_XLSX,
                Path(overrides[PATH_ENV_VARS[4]]).resolve(),
            )
            self.assertEqual(
                paths.SUPERVISOR_DROPPED_XLSX,
                Path(overrides[PATH_ENV_VARS[5]]).resolve(),
            )


class PackageImportTests(unittest.TestCase):
    def test_all_package_modules_import(self) -> None:
        module_names = sorted(
            module.name
            for module in pkgutil.walk_packages(
                german_gdp_nowcasting.__path__,
                german_gdp_nowcasting.__name__ + ".",
            )
        )
        self.assertGreater(len(module_names), 0)

        for module_name in module_names:
            with self.subTest(module=module_name):
                missing = [
                    dependency
                    for dependency in OPTIONAL_IMPORTS.get(module_name, ())
                    if importlib.util.find_spec(dependency) is None
                ]
                if missing:
                    raise unittest.SkipTest(
                        f"optional dependency unavailable: {', '.join(missing)}"
                    )
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
