import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from bump_inverse_design.metrics import performance_evaluation_criterion
from bump_inverse_design.surrogate import validation_gpr_config
from bump_inverse_design.validation import build_group_splits, cross_validate_surrogates


class ValidationTests(unittest.TestCase):
    def setUp(self):
        designs = np.array(
            [(w, h) for w in np.linspace(0.08, 0.28, 5) for h in np.linspace(0.02, 0.10, 4)]
        )
        self.frame = pd.DataFrame(designs, columns=["eps_w", "eps_h"])
        self.frame["Nu_ratio_avg"] = 1.0 + 0.25 * self.frame["eps_h"] + 0.02 * self.frame["eps_w"]
        self.frame["P_ratio"] = 1.0 + 0.10 * self.frame["eps_h"]
        self.frame["T_ratio_hot"] = 1.0 + 0.08 * self.frame["eps_h"]
        self.frame["T_ratio_cold"] = 1.0 + 0.05 * self.frame["eps_h"]
        self.frame["PEC"] = performance_evaluation_criterion(
            self.frame["Nu_ratio_avg"],
            self.frame["P_ratio"],
            self.frame["T_ratio_hot"],
            self.frame["T_ratio_cold"],
        )

    def test_group_splits_have_no_test_overlap_with_training(self):
        splits = build_group_splits(self.frame, "single", mode="leave_width", n_splits=5)
        self.assertEqual(len(splits), 5)
        for train, test in splits:
            self.assertEqual(np.intersect1d(train, test).size, 0)

    def test_cross_validation_returns_in_memory_results(self):
        config = validation_gpr_config(n_restarts_optimizer=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = cross_validate_surrogates(
                self.frame,
                case="single",
                mode="leave_width",
                target_columns=("Nu_ratio_avg",),
                n_splits=5,
                include_direct_pec=False,
                config=config,
            )
        self.assertTrue(np.all(result.fold_ids > 0))
        self.assertTrue(np.all(np.isfinite(result.predictions["Nu_ratio_avg"])))
        self.assertEqual(len(result.metrics), 6)

    def test_pec_is_recomputed_and_direct_pec_is_diagnostic(self):
        config = validation_gpr_config(n_restarts_optimizer=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = cross_validate_surrogates(
                self.frame,
                case="single",
                mode="boundary_holdout",
                config=config,
            )
        evaluated = result.fold_ids > 0
        self.assertTrue(np.all(np.isfinite(result.predictions["PEC_direct"][evaluated])))
        self.assertTrue(
            np.all(np.isfinite(result.predictions["PEC_recomputed_from_components"][evaluated]))
        )


if __name__ == "__main__":
    unittest.main()
