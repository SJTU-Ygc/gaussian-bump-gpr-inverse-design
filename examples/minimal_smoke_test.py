"""End-to-end demonstration using generated in-memory data only."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from bump_inverse_design import (
    ConstraintTargets,
    default_gpr_config,
    direct_inverse_query,
    fit_surrogate_bundle,
    optimize_design,
)


def main() -> None:
    eps_w = np.linspace(0.0625, 0.3125, 7)
    eps_h = np.linspace(0.00625, 0.11875, 7)
    designs = np.array([(width, height) for width in eps_w for height in eps_h])
    width = designs[:, 0]
    height = designs[:, 1]
    targets = {
        "Nu_ratio_avg": 1.0 + 0.50 * height + 0.02 * width,
        "P_ratio": 1.0 + 0.20 * height + 0.01 * width,
        "T_ratio_hot": 1.0 + 0.16 * height,
        "T_ratio_cold": 1.0 + 0.10 * height,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        bundle = fit_surrogate_bundle(
            designs,
            targets,
            case="single",
            config=default_gpr_config("single", n_restarts_optimizer=0),
        )

    library_predictions = bundle.predict_outputs(designs)
    constraints = ConstraintTargets(
        nu_min=1.03,
        pressure_max=1.04,
        temperature_ratio_max=1.04,
    )
    direct = direct_inverse_query(designs, library_predictions, constraints)
    continuous = optimize_design(
        bundle,
        bounds=((eps_w.min(), eps_w.max()), (eps_h.min(), eps_h.max())),
        targets=constraints,
        screen_designs=designs,
        initial_design=direct.design,
        n_starts=6,
    )
    if not direct.feasible or not continuous.success:
        raise RuntimeError("Synthetic smoke-test constraints were not satisfied.")
    print("Smoke test passed: training, direct query, and constrained optimisation completed in memory.")


if __name__ == "__main__":
    main()
