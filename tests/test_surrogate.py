import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from bump_inverse_design.surrogate import ScaledGPR, default_gpr_config, fit_surrogate_bundle


class SurrogateTests(unittest.TestCase):
    def setUp(self):
        width = np.linspace(0.07, 0.30, 6)
        height = np.linspace(0.01, 0.11, 5)
        self.designs = np.array([(w, h) for w in width for h in height])
        w = self.designs[:, 0]
        h = self.designs[:, 1]
        self.targets = {
            "Nu_ratio_avg": 1.0 + 0.35 * h + 0.03 * w,
            "P_ratio": 1.0 + 0.10 * h + 0.02 * w,
            "T_ratio_hot": 1.0 + 0.08 * h,
            "T_ratio_cold": 1.0 + 0.05 * h,
        }
        self.config = default_gpr_config("single", n_restarts_optimizer=0)

    def test_scaled_gpr_fit_predict(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = ScaledGPR("single", config=self.config).fit(
                self.designs, self.targets["Nu_ratio_avg"]
            )
        mean, std = model.predict(self.designs[:3], return_std=True)
        self.assertEqual(mean.shape, (3,))
        self.assertTrue(np.all(std >= 0.0))
        np.testing.assert_allclose(mean, self.targets["Nu_ratio_avg"][:3], atol=5e-3)

    def test_bundle_derives_temperature_and_pec(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            bundle = fit_surrogate_bundle(
                self.designs,
                self.targets,
                case="single",
                config=self.config,
            )
        prediction = bundle.predict_performance([0.20, 0.05])
        self.assertGreater(prediction.pec, 0.0)
        self.assertAlmostEqual(
            prediction.temperature_ratio,
            prediction.hot_ratio * prediction.cold_ratio,
        )


if __name__ == "__main__":
    unittest.main()
