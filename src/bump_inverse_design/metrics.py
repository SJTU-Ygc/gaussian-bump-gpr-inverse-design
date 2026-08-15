"""Performance and regression metrics used in the study."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _return_scalar_if_scalar(value: NDArray[np.float64] | np.float64):
    array = np.asarray(value)
    return float(array) if array.ndim == 0 else array


def temperature_ratio(
    hot_ratio: ArrayLike,
    cold_ratio: ArrayLike,
):
    """Return ``T_max/T_min`` from ``T_max/T_0`` and ``T_0/T_min``."""

    hot, cold = np.broadcast_arrays(np.asarray(hot_ratio, dtype=float), np.asarray(cold_ratio, dtype=float))
    if np.any(~np.isfinite(hot)) or np.any(~np.isfinite(cold)):
        raise ValueError("Temperature ratios must be finite.")
    return _return_scalar_if_scalar(hot * cold)


def performance_evaluation_criterion(
    nu_ratio: ArrayLike,
    pressure_ratio: ArrayLike,
    hot_ratio: ArrayLike,
    cold_ratio: ArrayLike,
    temperature_exponent: float = 0.1,
):
    """Compute the paper's heat-transfer performance evaluation criterion.

    ``PEC = (Nu/Nu0) / ((P/P0)^(1/3) * (Tmax/Tmin)^temperature_exponent)``
    """

    nu, pressure, hot, cold = np.broadcast_arrays(
        np.asarray(nu_ratio, dtype=float),
        np.asarray(pressure_ratio, dtype=float),
        np.asarray(hot_ratio, dtype=float),
        np.asarray(cold_ratio, dtype=float),
    )
    if not np.isfinite(temperature_exponent) or temperature_exponent < 0.0:
        raise ValueError("temperature_exponent must be finite and non-negative.")
    if np.any(~np.isfinite(nu)) or np.any(~np.isfinite(pressure)):
        raise ValueError("Nu and pressure ratios must be finite.")
    combined_temperature = np.asarray(temperature_ratio(hot, cold), dtype=float)
    pressure_safe = np.maximum(pressure, 1.0e-12)
    temperature_safe = np.maximum(combined_temperature, 1.0e-12)
    pec = nu / (np.power(pressure_safe, 1.0 / 3.0) * np.power(temperature_safe, temperature_exponent))
    return _return_scalar_if_scalar(pec)


def pec_standard_deviation(
    nu: ArrayLike,
    nu_std: ArrayLike,
    pressure: ArrayLike,
    pressure_std: ArrayLike,
    hot: ArrayLike,
    hot_std: ArrayLike,
    cold: ArrayLike,
    cold_std: ArrayLike,
    temperature_exponent: float = 0.1,
):
    """First-order uncertainty propagation used in the Group-CV analysis."""

    nu, nu_std, pressure, pressure_std, hot, hot_std, cold, cold_std = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (nu, nu_std, pressure, pressure_std, hot, hot_std, cold, cold_std)]
    )
    pec = np.asarray(
        performance_evaluation_criterion(nu, pressure, hot, cold, temperature_exponent),
        dtype=float,
    )
    variance_log = (
        (nu_std / np.maximum(np.abs(nu), 1.0e-12)) ** 2
        + ((pressure_std / np.maximum(np.abs(pressure), 1.0e-12)) / 3.0) ** 2
        + (temperature_exponent * hot_std / np.maximum(np.abs(hot), 1.0e-12)) ** 2
        + (temperature_exponent * cold_std / np.maximum(np.abs(cold), 1.0e-12)) ** 2
    )
    result = np.abs(pec) * np.sqrt(np.maximum(variance_log, 0.0))
    return _return_scalar_if_scalar(result)


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return R2, MAE, RMSE and MAPE for finite one-dimensional arrays."""

    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays with equal shape.")
    if np.any(~np.isfinite(true)) or np.any(~np.isfinite(pred)):
        raise ValueError("Regression inputs must be finite.")
    mape = float(np.mean(np.abs((pred - true) / np.maximum(np.abs(true), 1.0e-12))) * 100.0)
    return {
        "r2": float(r2_score(true, pred)),
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mape_percent": mape,
    }
