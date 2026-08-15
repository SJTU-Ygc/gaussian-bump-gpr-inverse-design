import unittest

import numpy as np
import pandas as pd

from bump_inverse_design.datasets import detect_case, standardize_dataframe
from bump_inverse_design.metrics import (
    performance_evaluation_criterion,
    regression_metrics,
    temperature_ratio,
)


class DatasetAndMetricTests(unittest.TestCase):
    def test_alias_standardisation_does_not_modify_input(self):
        source = pd.DataFrame(
            {
                "epsilon_w": [0.1, 0.2],
                "epsilon_h": [0.02, 0.04],
                "Nu_ratio": [1.02, 1.05],
                "pressure_ratio": [1.01, 1.03],
                "Tmax_over_T0": [1.04, 1.06],
                "T0_over_Tmin": [1.03, 1.05],
            }
        )
        original_columns = tuple(source.columns)
        self.assertEqual(detect_case(source), "single")
        standard, schema = standardize_dataframe(source)
        self.assertEqual(schema.case, "single")
        self.assertEqual(tuple(source.columns), original_columns)
        self.assertIn("T_ratio_hot", standard)
        self.assertIn("T_ratio_cold", standard)

    def test_temperature_and_pec_formula(self):
        self.assertAlmostEqual(temperature_ratio(1.2, 1.1), 1.32)
        expected = 1.15 / ((1.09 ** (1.0 / 3.0)) * (1.32**0.1))
        actual = performance_evaluation_criterion(1.15, 1.09, 1.2, 1.1)
        self.assertAlmostEqual(actual, expected)

    def test_regression_metrics(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertTrue(np.isfinite(metrics["mape_percent"]))


if __name__ == "__main__":
    unittest.main()
