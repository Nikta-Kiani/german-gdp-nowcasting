"""Tests for selection smoothing and forecast-evaluation helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from german_gdp_nowcasting.models.dfm.nowcast_utils import (
    compute_rmsfe,
    compute_rmsfe_by_month_in_quarter,
    diebold_mariano_test,
)
from german_gdp_nowcasting.selection.selection_postprocessing import (
    apply_frequency_smoothing,
)


class SelectionSmoothingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = pd.DataFrame(
            {
                "persistent_early": [1, 1, 1, 0, 0, 0],
                "persistent_late": [0, 0, 1, 1, 1, 1],
            },
            index=["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"],
            dtype=int,
        )
        self.raw.index.name = "forecast_origin"

    def test_one_quarter_window_pools_monthly_vintages(self) -> None:
        smoothed = apply_frequency_smoothing(
            self.raw,
            window_quarters=1,
            min_freq=0.5,
        )

        expected = pd.DataFrame(
            {
                "persistent_early": [1, 1, 1, 0, 0, 0],
                "persistent_late": [0, 0, 0, 1, 1, 1],
            },
            index=self.raw.index,
            dtype=int,
        )
        pd.testing.assert_frame_equal(smoothed, expected)

    def test_two_quarter_window_applies_inclusive_frequency_threshold(self) -> None:
        smoothed = apply_frequency_smoothing(
            self.raw,
            window_quarters=2,
            min_freq=0.5,
        )

        self.assertEqual(smoothed.loc["2020-04", "persistent_early"], 1)
        self.assertEqual(smoothed.loc["2020-04", "persistent_late"], 1)
        self.assertEqual(smoothed.index.name, self.raw.index.name)
        self.assertTrue(all(dtype.kind in "iu" for dtype in smoothed.dtypes))

    def test_invalid_smoothing_parameters_raise(self) -> None:
        for bad_frequency in (0.0, -0.1, 1.1):
            with self.subTest(min_freq=bad_frequency):
                with self.assertRaises(ValueError):
                    apply_frequency_smoothing(self.raw, min_freq=bad_frequency)
        with self.assertRaises(ValueError):
            apply_frequency_smoothing(self.raw, window_quarters=0)


class ForecastMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.DataFrame(
            {
                "quarter": ["2020Q1", "2020Q1", "2020Q2", "2020Q2"],
                "month_in_quarter": [2, 3, 2, 3],
                "error": [4.0, 3.0, np.nan, 0.0],
            }
        )

    def test_rmsfe_filters_window_month_and_missing_values(self) -> None:
        self.assertAlmostEqual(
            compute_rmsfe(self.results),
            np.sqrt((4.0**2 + 3.0**2 + 0.0**2) / 3),
        )
        self.assertAlmostEqual(
            compute_rmsfe(
                self.results,
                eval_start="2020Q1",
                eval_end="2020Q1",
                month_in_quarter=3,
            ),
            3.0,
        )
        empty = self.results.assign(error=np.nan)
        self.assertTrue(np.isnan(compute_rmsfe(empty)))

    def test_rmsfe_by_month_reports_each_origin_sleeve(self) -> None:
        by_month = compute_rmsfe_by_month_in_quarter(self.results)

        self.assertAlmostEqual(by_month.loc[2], 4.0)
        self.assertAlmostEqual(by_month.loc[3], np.sqrt((3.0**2 + 0.0**2) / 2))
        self.assertTrue(np.isnan(by_month.loc[1]))

    def test_diebold_mariano_direction_nan_pairing_and_validation(self) -> None:
        errors_a = np.array([0.2, -0.1, 0.3, -0.2, 0.1, -0.4, 0.2, -0.2, 0.3, np.nan])
        errors_b = np.array([1.0, -0.8, 1.2, -0.7, 0.9, -1.1, 0.8, -1.3, 1.1, 0.9])

        result = diebold_mariano_test(errors_a, errors_b, h=1, loss="se")

        self.assertEqual(result["n"], 9)
        self.assertLess(result["DM"], 0.0)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)
        with self.assertRaises(ValueError):
            diebold_mariano_test(np.arange(8), np.arange(8) + 1, loss="unsupported")


if __name__ == "__main__":
    unittest.main()
