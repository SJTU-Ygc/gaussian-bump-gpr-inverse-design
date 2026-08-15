"""Grid screening, L-BFGS-B refinement, and constrained SLSQP optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .surrogate import PerformancePrediction


class PerformanceModel(Protocol):
    def predict_performance(self, design: ArrayLike) -> PerformancePrediction: ...


@dataclass(frozen=True)
class ConstraintTargets:
    nu_min: float | None = None
    pressure_max: float | None = None
    temperature_ratio_max: float | None = None


@dataclass(frozen=True)
class OptimizationResult:
    design: NDArray[np.float64]
    prediction: PerformancePrediction
    success: bool
    message: str
    n_function_evaluations: int
    maximum_constraint_violation: float


def _bounds_array(bounds: Sequence[tuple[float, float]]) -> NDArray[np.float64]:
    array = np.asarray(bounds, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or np.any(array[:, 0] >= array[:, 1]):
        raise ValueError("bounds must contain valid (lower, upper) pairs.")
    return array


def structured_grid(
    bounds: Sequence[tuple[float, float]],
    points_per_dimension: int | Sequence[int],
) -> NDArray[np.float64]:
    """Build the dense Cartesian design grid used before local refinement."""

    bound_array = _bounds_array(bounds)
    if np.isscalar(points_per_dimension):
        counts = np.full(bound_array.shape[0], int(points_per_dimension), dtype=int)
    else:
        counts = np.asarray(points_per_dimension, dtype=int)
    if counts.shape != (bound_array.shape[0],) or np.any(counts < 2):
        raise ValueError("points_per_dimension must provide at least two points for each variable.")
    axes = [np.linspace(low, high, count) for (low, high), count in zip(bound_array, counts)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([values.ravel() for values in mesh])


def _violations(prediction: PerformancePrediction, targets: ConstraintTargets) -> NDArray[np.float64]:
    values = []
    if targets.nu_min is not None:
        values.append(max(0.0, targets.nu_min - prediction.nu_ratio))
    if targets.pressure_max is not None:
        values.append(max(0.0, prediction.pressure_ratio - targets.pressure_max))
    if targets.temperature_ratio_max is not None:
        values.append(max(0.0, prediction.temperature_ratio - targets.temperature_ratio_max))
    return np.asarray(values or [0.0], dtype=float)


def _constraints(model: PerformanceModel, targets: ConstraintTargets):
    constraints = []
    if targets.nu_min is not None:
        constraints.append({"type": "ineq", "fun": lambda x: model.predict_performance(x).nu_ratio - targets.nu_min})
    if targets.pressure_max is not None:
        constraints.append({"type": "ineq", "fun": lambda x: targets.pressure_max - model.predict_performance(x).pressure_ratio})
    if targets.temperature_ratio_max is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x: targets.temperature_ratio_max - model.predict_performance(x).temperature_ratio}
        )
    return constraints


def _predict_many(model: PerformanceModel, designs: NDArray[np.float64]):
    if hasattr(model, "predict_outputs"):
        output = model.predict_outputs(designs)
        return {
            "nu": np.asarray(output["Nu_ratio_avg"], dtype=float),
            "pressure": np.asarray(output["P_ratio"], dtype=float),
            "temperature": np.asarray(output["Tmax_over_Tmin"], dtype=float),
            "pec": np.asarray(output["PEC"], dtype=float),
        }
    predictions = [model.predict_performance(row) for row in designs]
    return {
        "nu": np.asarray([item.nu_ratio for item in predictions]),
        "pressure": np.asarray([item.pressure_ratio for item in predictions]),
        "temperature": np.asarray([item.temperature_ratio for item in predictions]),
        "pec": np.asarray([item.pec for item in predictions]),
    }


def screen_initial_designs(
    model: PerformanceModel,
    designs: ArrayLike,
    targets: ConstraintTargets | None = None,
    n_starts: int = 30,
    minimum_separation: float = 0.03,
) -> NDArray[np.float64]:
    """Select separated high-PEC starts from a precomputed design field."""

    candidates = np.asarray(designs, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("designs must be a non-empty two-dimensional array.")
    output = _predict_many(model, candidates)
    requested = targets or ConstraintTargets()
    feasible = np.isfinite(output["pec"])
    if requested.nu_min is not None:
        feasible &= output["nu"] >= requested.nu_min
    if requested.pressure_max is not None:
        feasible &= output["pressure"] <= requested.pressure_max
    if requested.temperature_ratio_max is not None:
        feasible &= output["temperature"] <= requested.temperature_ratio_max
    feasible_index = np.flatnonzero(feasible)
    if feasible_index.size == 0:
        return np.empty((0, candidates.shape[1]), dtype=float)

    order = feasible_index[np.argsort(output["pec"][feasible_index])[::-1]]
    low = candidates.min(axis=0)
    span = np.maximum(candidates.max(axis=0) - low, 1.0e-12)
    selected: list[NDArray[np.float64]] = []
    for index in order:
        point = candidates[index]
        normalized = (point - low) / span
        if not selected or all(
            np.linalg.norm(normalized - (old - low) / span) >= minimum_separation for old in selected
        ):
            selected.append(point.copy())
        if len(selected) >= int(n_starts):
            break
    return np.asarray(selected, dtype=float)


def _unique_starts(starts: Sequence[ArrayLike], dimension: int) -> list[NDArray[np.float64]]:
    unique = []
    for start in starts:
        point = np.asarray(start, dtype=float).reshape(-1)
        if point.shape != (dimension,) or np.any(~np.isfinite(point)):
            continue
        if not any(np.allclose(point, old, atol=1.0e-10) for old in unique):
            unique.append(point)
    return unique


def optimize_unconstrained(
    model: PerformanceModel,
    bounds: Sequence[tuple[float, float]],
    screen_designs: ArrayLike | None = None,
    initial_design: ArrayLike | None = None,
    n_starts: int = 30,
    max_iterations: int = 500,
) -> OptimizationResult:
    """Maximise PEC by full-domain screening followed by L-BFGS-B."""

    bound_array = _bounds_array(bounds)
    starts: list[ArrayLike] = []
    if initial_design is not None:
        starts.append(initial_design)
    if screen_designs is not None:
        starts.extend(screen_initial_designs(model, screen_designs, n_starts=n_starts))
    if not starts:
        starts.append(bound_array.mean(axis=1))
    starts = _unique_starts(starts, bound_array.shape[0])

    candidates = []
    evaluations = 0
    for start in starts:
        result = minimize(
            lambda x: -float(model.predict_performance(x).pec),
            np.clip(start, bound_array[:, 0], bound_array[:, 1]),
            method="L-BFGS-B",
            bounds=[tuple(pair) for pair in bound_array],
            options={"maxiter": int(max_iterations), "ftol": 1.0e-12},
        )
        evaluations += int(result.nfev)
        design = np.asarray(result.x, dtype=float)
        prediction = model.predict_performance(design)
        candidates.append((design, prediction, bool(result.success), str(result.message)))
    best = max(candidates, key=lambda item: item[1].pec)
    return OptimizationResult(best[0], best[1], best[2], best[3], evaluations, 0.0)


def optimize_design(
    model: PerformanceModel,
    bounds: Sequence[tuple[float, float]],
    targets: ConstraintTargets | None = None,
    screen_designs: ArrayLike | None = None,
    initial_design: ArrayLike | None = None,
    n_starts: int = 30,
    max_iterations: int = 500,
    feasibility_tolerance: float = 1e-6,
) -> OptimizationResult:
    """Maximise PEC under Nu, pressure, and temperature constraints using SLSQP."""

    bound_array = _bounds_array(bounds)
    requested = targets or ConstraintTargets()
    starts: list[ArrayLike] = []
    screened = np.empty((0, bound_array.shape[0]))
    if initial_design is not None:
        starts.append(initial_design)
    if screen_designs is not None:
        screened = screen_initial_designs(model, screen_designs, requested, n_starts=n_starts)
        starts.extend(screened)
    if not starts:
        starts.append(bound_array.mean(axis=1))
    starts = _unique_starts(starts, bound_array.shape[0])
    constraints = _constraints(model, requested)
    candidates = []
    evaluations = 0

    for point in screened:
        prediction = model.predict_performance(point)
        candidates.append((point.copy(), prediction, float(np.max(_violations(prediction, requested))), True, "Grid-screened feasible point"))
    for start in starts:
        result = minimize(
            lambda x: -float(model.predict_performance(x).pec),
            np.clip(start, bound_array[:, 0], bound_array[:, 1]),
            method="SLSQP",
            bounds=[tuple(pair) for pair in bound_array],
            constraints=constraints,
            options={"maxiter": int(max_iterations), "ftol": 1.0e-7, "disp": False},
        )
        evaluations += int(result.nfev)
        design = np.clip(np.asarray(result.x, dtype=float), bound_array[:, 0], bound_array[:, 1])
        prediction = model.predict_performance(design)
        violation = float(np.max(_violations(prediction, requested)))
        candidates.append((design, prediction, violation, bool(result.success), str(result.message)))

    feasible = [item for item in candidates if item[2] <= feasibility_tolerance]
    if feasible:
        best = max(feasible, key=lambda item: item[1].pec)
        return OptimizationResult(best[0], best[1], True, best[4], evaluations, best[2])
    best = min(candidates, key=lambda item: (item[2], -item[1].pec))
    return OptimizationResult(
        best[0],
        best[1],
        False,
        "No feasible design was found; returning the minimum-violation point. " + best[4],
        evaluations,
        best[2],
    )


def trace_constraint_path(
    model: PerformanceModel,
    bounds: Sequence[tuple[float, float]],
    target_sequence: Sequence[ConstraintTargets],
    screen_designs: ArrayLike | None = None,
    initial_design: ArrayLike | None = None,
    n_starts: int = 30,
) -> tuple[OptimizationResult, ...]:
    """Solve a sequence of relaxed constraints using the preceding optimum as a start."""

    current = initial_design
    results = []
    for targets in target_sequence:
        result = optimize_design(
            model,
            bounds,
            targets,
            screen_designs=screen_designs,
            initial_design=current,
            n_starts=n_starts,
        )
        results.append(result)
        if result.success:
            current = result.design
    return tuple(results)
