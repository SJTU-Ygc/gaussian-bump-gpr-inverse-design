"""Direct design-library queries with optional continuous surrogate refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .optimization import ConstraintTargets, OptimizationResult, PerformanceModel, optimize_design


@dataclass(frozen=True)
class DirectQueryResult:
    design: NDArray[np.float64]
    index: int
    feasible_indices: NDArray[np.int64]
    prediction: dict[str, float]
    feasible: bool
    maximum_constraint_violation: float


def _prediction_arrays(predictions: Mapping[str, ArrayLike], n_rows: int) -> dict[str, NDArray[np.float64]]:
    required = ("Nu_ratio_avg", "P_ratio", "Tmax_over_Tmin", "PEC")
    missing = [name for name in required if name not in predictions]
    if missing:
        raise ValueError(f"Missing prediction arrays: {missing}.")
    arrays = {name: np.asarray(predictions[name], dtype=float).reshape(-1) for name in required}
    if any(values.size != n_rows for values in arrays.values()):
        raise ValueError("Every prediction array must match the number of designs.")
    if any(np.any(~np.isfinite(values)) for values in arrays.values()):
        raise ValueError("Prediction arrays contain non-finite values.")
    return arrays


def direct_inverse_query(
    designs: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    targets: ConstraintTargets | None = None,
) -> DirectQueryResult:
    """Return the highest-PEC feasible library design, or least-violating design."""

    design_array = np.asarray(designs, dtype=float)
    if design_array.ndim != 2 or design_array.shape[0] == 0 or np.any(~np.isfinite(design_array)):
        raise ValueError("designs must be a non-empty finite two-dimensional array.")
    arrays = _prediction_arrays(predictions, design_array.shape[0])
    requested = targets or ConstraintTargets()
    violations = np.zeros(design_array.shape[0], dtype=float)
    if requested.nu_min is not None:
        violations = np.maximum(violations, requested.nu_min - arrays["Nu_ratio_avg"])
    if requested.pressure_max is not None:
        violations = np.maximum(violations, arrays["P_ratio"] - requested.pressure_max)
    if requested.temperature_ratio_max is not None:
        violations = np.maximum(violations, arrays["Tmax_over_Tmin"] - requested.temperature_ratio_max)
    violations = np.maximum(violations, 0.0)
    feasible = violations <= 1e-12
    if np.any(feasible):
        eligible = np.flatnonzero(feasible)
        index = int(eligible[np.argmax(arrays["PEC"][eligible])])
        is_feasible = True
    else:
        minimum = float(np.min(violations))
        eligible = np.flatnonzero(np.isclose(violations, minimum))
        index = int(eligible[np.argmax(arrays["PEC"][eligible])])
        is_feasible = False
    return DirectQueryResult(
        design=design_array[index].copy(),
        index=index,
        feasible_indices=np.flatnonzero(feasible).astype(int),
        prediction={name: float(values[index]) for name, values in arrays.items()},
        feasible=is_feasible,
        maximum_constraint_violation=float(violations[index]),
    )


def refine_inverse_query(
    model: PerformanceModel,
    direct_result: DirectQueryResult,
    bounds: Sequence[tuple[float, float]],
    targets: ConstraintTargets | None = None,
    screen_designs: ArrayLike | None = None,
    n_starts: int = 30,
) -> OptimizationResult:
    """Use a direct library result as one start for continuous constrained optimisation."""

    return optimize_design(
        model,
        bounds=bounds,
        targets=targets,
        screen_designs=screen_designs,
        initial_design=direct_result.design,
        n_starts=n_starts,
    )
