"""Feature transforms used by the single-bump and H3 GPR models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_2d(values: ArrayLike, expected_columns: int, label: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != expected_columns:
        raise ValueError(f"{label} must have shape (n, {expected_columns}).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values.")
    return array


def single_bump_features(values: ArrayLike, mode: str = "raw") -> NDArray[np.float64]:
    """Transform ``[eps_w, eps_h]`` single-bump variables."""

    design = _as_2d(values, 2, "Single-bump designs")
    if mode == "raw":
        return design.copy()
    if mode not in {"engineered", "validation"}:
        raise ValueError("Single-bump feature mode must be 'raw' or 'validation'.")

    eps_w = design[:, 0]
    eps_h = design[:, 1]
    if np.any(eps_w < 0.0) or np.any(eps_h < 0.0):
        raise ValueError("eps_w and eps_h must be non-negative.")
    return np.column_stack((eps_w, eps_h, eps_h / np.maximum(eps_w, 1.0e-12), eps_w * eps_h))


def h3_features(values: ArrayLike, mode: str = "engineered") -> NDArray[np.float64]:
    """Transform ``[s_mm, alpha_w, alpha_h]`` H3 variables."""

    design = _as_2d(values, 3, "Homogeneous three-bump designs")
    if mode == "raw":
        return design.copy()

    s = design[:, 0]
    alpha_w = design[:, 1]
    alpha_h = design[:, 2]
    if np.any(alpha_w <= 0.0) or np.any(alpha_h <= 0.0):
        raise ValueError("alpha_w and alpha_h must be positive.")

    tiny = 1.0e-12
    if mode == "validation":
        return np.column_stack(
            (
                s,
                alpha_w,
                alpha_h,
                alpha_h / np.maximum(alpha_w, tiny),
                alpha_w * alpha_h,
                s * alpha_w,
                s * alpha_h,
            )
        )
    if mode != "engineered":
        raise ValueError("H3 feature mode must be 'raw', 'validation', or 'engineered'.")

    spacing_over_width = s / (alpha_w + tiny)
    spacing_over_height = s / (alpha_h + tiny)
    return np.column_stack(
        (
            s,
            alpha_w,
            alpha_h,
            alpha_h / (alpha_w + tiny),
            alpha_w / (alpha_h + tiny),
            np.log((alpha_h + tiny) / (alpha_w + tiny)),
            alpha_w * alpha_h,
            alpha_h - alpha_w,
            s * alpha_w,
            s * alpha_h,
            spacing_over_width,
            spacing_over_height,
            np.exp(-((spacing_over_width / 6.0) ** 2)),
            np.exp(-((spacing_over_height / 6.0) ** 2)),
        )
    )


def make_features(values: ArrayLike, case: str, mode: str | None = None) -> NDArray[np.float64]:
    """Apply the feature transformation for ``single`` or ``h3`` designs."""

    normalized = case.strip().lower()
    if normalized == "single":
        return single_bump_features(values, mode or "raw")
    if normalized == "h3":
        return h3_features(values, mode or "engineered")
    raise ValueError("case must be 'single' or 'h3'.")
