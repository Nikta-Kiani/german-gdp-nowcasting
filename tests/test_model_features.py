"""Fast feature-engineering tests for XGBoost and the factor MLP."""

from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from german_gdp_nowcasting.models.mlp.mlp_utils import (
    make_lag_features,
    train_pred_arrays,
)


XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
if XGBOOST_AVAILABLE:
    from german_gdp_nowcasting.models.xgboost.xgb_utils import build_xgb_design_matrix


@unittest.skipUnless(XGBOOST_AVAILABLE, "optional dependency xgboost is unavailable")
class XGBoostDesignMatrixTests(unittest.TestCase):
    def test_design_matrix_excludes_future_months_and_target_gdp(self) -> None:
        monthly_index = pd.date_range("2019-01-01", "2020-06-01", freq="MS")
        base = pd.DataFrame(
            {"indicator": np.arange(len(monthly_index), dtype=float)},
            index=monthly_index,
        )
        changed_future = base.copy()
        changed_future.loc["2020-03-01":, "indicator"] = 1_000_000.0

        quarters = pd.period_range("2019Q1", "2020Q1", freq="Q")
        y_a = pd.Series([1.0, 2.0, 3.0, 4.0, 999.0], index=quarters)
        y_b = y_a.copy()
        y_b.loc[pd.Period("2020Q1")] = -999.0
        captured_panels: list[pd.DataFrame] = []

        def synthetic_quarterly_bridge(
            filled: pd.DataFrame,
            origin: pd.Period,
            pub_lag_map: pd.Series,
            trafo_map: pd.Series,
        ) -> pd.DataFrame:
            del origin, pub_lag_map, trafo_map
            captured_panels.append(filled.copy())
            quarterly = filled.groupby(filled.index.to_period("Q")).mean()
            quarterly.index.name = "quarter"
            return quarterly

        kwargs = {
            "origin": pd.Period("2020-02", freq="M"),
            "selected_cols": ["indicator"],
            "pub_lag_map": pd.Series({"indicator": 0}),
            "lags": (0, 1),
            "train_start": "2019Q1",
            "trafo_map": pd.Series({"indicator": 0}),
            "fill_method": "none",
            "gdp_lags": (1,),
        }
        patch_target = (
            "german_gdp_nowcasting.selection.aggregation.quarterly_block_realtime"
        )
        with mock.patch(patch_target, side_effect=synthetic_quarterly_bridge):
            train_a, target_a, pred_a = build_xgb_design_matrix(
                X_monthly=base,
                y_quarterly=y_a,
                **kwargs,
            )
            train_b, target_b, pred_b = build_xgb_design_matrix(
                X_monthly=changed_future,
                y_quarterly=y_b,
                **kwargs,
            )

        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_series_equal(target_a, target_b)
        pd.testing.assert_frame_equal(pred_a, pred_b)
        self.assertEqual(
            list(train_a.index),
            list(pd.period_range("2019Q1", "2019Q4", freq="Q")),
        )
        self.assertEqual(pred_a.loc[pd.Period("2020Q1"), "gdp__L1"], 4.0)
        self.assertTrue(
            all(panel.loc["2020-03-01", "indicator"] != panel.loc["2020-03-01", "indicator"]
                for panel in captured_panels)
        )

    def test_design_matrix_requires_realtime_inputs(self) -> None:
        monthly = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="MS"),
        )
        y = pd.Series([1.0], index=pd.period_range("2020Q1", periods=1, freq="Q"))

        with self.assertRaisesRegex(ValueError, "selected_cols"):
            build_xgb_design_matrix(
                monthly,
                y,
                pd.Period("2020-03", freq="M"),
                selected_cols=[],
                pub_lag_map=pd.Series(dtype=int),
            )
        with self.assertRaisesRegex(ValueError, "pub_lag_map"):
            build_xgb_design_matrix(
                monthly,
                y,
                pd.Period("2020-03", freq="M"),
                selected_cols=["x"],
                pub_lag_map=None,
            )


class MLPFeatureTests(unittest.TestCase):
    def test_make_lag_features_has_expected_names_and_values(self) -> None:
        quarters = pd.period_range("2020Q1", periods=4, freq="Q")
        factors = pd.DataFrame(
            {"F1": [1.0, 2.0, 3.0, 4.0], "F2": [10.0, 20.0, 30.0, 40.0]},
            index=quarters,
        )

        features = make_lag_features(factors, n_lags=2)

        self.assertEqual(
            list(features.columns),
            ["F1_L0", "F2_L0", "F1_L1", "F2_L1", "F1_L2", "F2_L2"],
        )
        expected_last = [4.0, 40.0, 3.0, 30.0, 2.0, 20.0]
        np.testing.assert_allclose(features.iloc[-1], expected_last)
        self.assertTrue(features.iloc[0][["F1_L1", "F2_L1", "F1_L2", "F2_L2"]].isna().all())

    def test_train_prediction_arrays_exclude_target_from_training(self) -> None:
        quarters = pd.period_range("1991Q1", periods=7, freq="Q")
        factors = pd.DataFrame(
            {
                "F1": np.arange(1.0, 8.0),
                "F2": np.arange(11.0, 18.0),
            },
            index=quarters,
        )
        target_q = quarters[-1]
        y = pd.Series(np.arange(101.0, 108.0), index=quarters)
        y_changed = y.copy()
        y_changed.loc[target_q] = -10_000.0

        train, target, pred, n_train = train_pred_arrays(factors, y, target_q)
        train_changed, target_changed, pred_changed, n_changed = train_pred_arrays(
            factors,
            y_changed,
            target_q,
        )

        np.testing.assert_allclose(train, train_changed)
        np.testing.assert_allclose(target, target_changed)
        np.testing.assert_allclose(pred, pred_changed)
        self.assertEqual(n_train, n_changed)
        self.assertEqual(n_train, 4)
        np.testing.assert_allclose(target, y.iloc[2:-1])
        self.assertEqual(pred.shape, (1, 6))


if __name__ == "__main__":
    unittest.main()
