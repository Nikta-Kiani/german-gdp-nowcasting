"""Synthetic tests for real-time masking and quarterly aggregation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from german_gdp_nowcasting.models.dfm.ragged_edge import (
    apply_pub_lag_mask,
    last_observed_month,
)
from german_gdp_nowcasting.selection.aggregation import (
    monthly_to_quarterly_raw,
    raw_to_quarterly_levels,
    retransform_quarterly,
)


class PublicationLagMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2020-01-01", periods=6, freq="MS")
        self.panel = pd.DataFrame(
            {
                "fast": np.arange(1.0, 7.0),
                "hard": np.arange(11.0, 17.0),
                "unmapped": np.arange(21.0, 27.0),
            },
            index=self.index,
        )

    def test_last_observed_month_subtracts_publication_lag(self) -> None:
        origin = pd.Period("2020-04", freq="M")
        self.assertEqual(last_observed_month(origin, 0), origin)
        self.assertEqual(
            last_observed_month(origin, 2),
            pd.Period("2020-02", freq="M"),
        )

    def test_mask_respects_per_series_lags_and_preserves_input(self) -> None:
        original = self.panel.copy()
        masked = apply_pub_lag_mask(
            self.panel,
            origin="2020-04",
            pub_lag_map=pd.Series({"fast": 0, "hard": 2}),
        )

        pd.testing.assert_frame_equal(self.panel, original)
        pd.testing.assert_series_equal(
            masked.loc[:"2020-04-01", "fast"],
            original.loc[:"2020-04-01", "fast"],
        )
        self.assertTrue(masked.loc["2020-05-01":, "fast"].isna().all())
        pd.testing.assert_series_equal(
            masked.loc[:"2020-02-01", "hard"],
            original.loc[:"2020-02-01", "hard"],
        )
        self.assertTrue(masked.loc["2020-03-01":, "hard"].isna().all())
        self.assertTrue(masked.loc["2020-05-01":, "unmapped"].isna().all())


class QuarterlyAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2020-01-01", periods=6, freq="MS")

    def test_raw_levels_support_mean_and_last_rules(self) -> None:
        raw = pd.DataFrame(
            {"flow": range(1, 7), "stock": range(11, 17)},
            index=self.index,
            dtype=float,
        )
        methods = pd.Series({"flow": "mean", "stock": "last"})

        quarterly = raw_to_quarterly_levels(raw, methods)

        expected = pd.DataFrame(
            {"flow": [2.0, 5.0], "stock": [13.0, 16.0]},
            index=pd.period_range("2020Q1", periods=2, freq="Q"),
        )
        expected.index.name = "quarter"
        pd.testing.assert_frame_equal(quarterly, expected)

    def test_retransform_uses_identity_log_growth_and_difference_fallback(self) -> None:
        quarters = pd.period_range("2020Q1", periods=2, freq="Q")
        levels = pd.DataFrame(
            {
                "level": [10.0, 20.0],
                "positive_growth": [100.0, 110.0],
                "nonpositive_growth": [-1.0, 2.0],
            },
            index=quarters,
        )
        trafo = pd.Series(
            {"level": 0, "positive_growth": 1, "nonpositive_growth": 1}
        )

        transformed = retransform_quarterly(levels, trafo)

        self.assertEqual(transformed.index.name, "quarter")
        np.testing.assert_allclose(transformed["level"], [10.0, 20.0])
        self.assertTrue(np.isnan(transformed.loc[quarters[0], "positive_growth"]))
        self.assertAlmostEqual(
            transformed.loc[quarters[1], "positive_growth"],
            np.log(110.0) - np.log(100.0),
        )
        self.assertTrue(np.isnan(transformed.loc[quarters[0], "nonpositive_growth"]))
        self.assertEqual(
            transformed.loc[quarters[1], "nonpositive_growth"],
            3.0,
        )

    def test_full_raw_bridge_uses_supplied_synthetic_panel(self) -> None:
        raw = pd.DataFrame(
            {"survey": [1, 2, 3, 4, 5, 6], "index": [90, 100, 110, 99, 110, 121]},
            index=self.index,
            dtype=float,
        )
        transformed = monthly_to_quarterly_raw(
            columns=["survey", "index"],
            trafo_map=pd.Series({"survey": 0, "index": 1}),
            index=self.index,
            raw_panel=raw,
            methods=pd.Series({"survey": "mean", "index": "mean"}),
        )

        self.assertEqual(transformed.loc[pd.Period("2020Q1"), "survey"], 2.0)
        self.assertEqual(transformed.loc[pd.Period("2020Q2"), "survey"], 5.0)
        self.assertAlmostEqual(
            transformed.loc[pd.Period("2020Q2"), "index"],
            np.log(110.0) - np.log(100.0),
        )


if __name__ == "__main__":
    unittest.main()
