import unittest

import numpy as np

from bump_inverse_design.inverse_query import direct_inverse_query, refine_inverse_query
from bump_inverse_design.metrics import performance_evaluation_criterion
from bump_inverse_design.optimization import (
    ConstraintTargets,
    optimize_design,
    optimize_unconstrained,
    structured_grid,
)
from bump_inverse_design.surrogate import PerformancePrediction


class AnalyticPerformanceModel:
    def predict_performance(self, design):
        x, y = np.asarray(design, dtype=float)
        nu = 1.0 + 0.20 * x
        pressure = 1.0 + 0.10 * y
        combined_temperature = 1.0 + 0.05 * (x + y)
        hot = np.sqrt(combined_temperature)
        cold = np.sqrt(combined_temperature)
        pec = performance_evaluation_criterion(nu, pressure, hot, cold)
        return PerformancePrediction(nu, pressure, hot, cold, combined_temperature, pec)


class OptimizationAndQueryTests(unittest.TestCase):
    def setUp(self):
        self.model = AnalyticPerformanceModel()
        self.targets = ConstraintTargets(nu_min=1.10, pressure_max=1.04, temperature_ratio_max=1.06)

    def test_constrained_optimization_is_feasible(self):
        grid = structured_grid(((0.0, 1.0), (0.0, 1.0)), (11, 11))
        result = optimize_design(
            self.model,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            targets=self.targets,
            screen_designs=grid,
            n_starts=5,
        )
        self.assertTrue(result.success)
        self.assertLessEqual(result.maximum_constraint_violation, 1e-7)
        self.assertGreaterEqual(result.prediction.nu_ratio, self.targets.nu_min - 1e-7)
        self.assertLessEqual(result.prediction.pressure_ratio, self.targets.pressure_max + 1e-7)

    def test_unconstrained_grid_screening_and_lbfgsb(self):
        grid = structured_grid(((0.0, 1.0), (0.0, 1.0)), (11, 11))
        result = optimize_unconstrained(
            self.model,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            screen_designs=grid,
            n_starts=5,
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.design[0], 1.0, places=5)
        self.assertAlmostEqual(result.design[1], 0.0, places=5)

    def test_direct_query_and_refinement(self):
        grid = np.array([(x, y) for x in np.linspace(0.0, 1.0, 6) for y in np.linspace(0.0, 1.0, 6)])
        predictions = [self.model.predict_performance(row) for row in grid]
        arrays = {
            "Nu_ratio_avg": [item.nu_ratio for item in predictions],
            "P_ratio": [item.pressure_ratio for item in predictions],
            "Tmax_over_Tmin": [item.temperature_ratio for item in predictions],
            "PEC": [item.pec for item in predictions],
        }
        direct = direct_inverse_query(grid, arrays, self.targets)
        self.assertTrue(direct.feasible)
        self.assertGreater(direct.feasible_indices.size, 0)
        refined = refine_inverse_query(
            self.model,
            direct,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            targets=self.targets,
            screen_designs=grid,
            n_starts=4,
        )
        self.assertTrue(refined.success)
        self.assertGreaterEqual(refined.prediction.pec, direct.prediction["PEC"] - 1e-8)


if __name__ == "__main__":
    unittest.main()
