"""Dataset schema detection and in-memory column standardisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSchema:
    case: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...] = ("Nu_ratio_avg", "P_ratio", "T_ratio_hot", "T_ratio_cold")


SINGLE_SCHEMA = DatasetSchema("single", ("eps_w", "eps_h"))
H3_SCHEMA = DatasetSchema("h3", ("s_mm", "alpha_w", "alpha_h"))

_ALIASES: dict[str, tuple[str, ...]] = {
    "eps_w": ("eps_w", "epsilon_w", "epsilon_width", "width_factor", "w_factor"),
    "eps_h": ("eps_h", "epsilon_h", "epsilon_height", "height_factor", "h_factor"),
    "s_mm": ("s_mm", "spacing_mm", "spacing", "s"),
    "alpha_w": ("alpha_w", "aw", "width_scale", "alpha_width"),
    "alpha_h": ("alpha_h", "ah", "height_scale", "alpha_height"),
    "Nu_ratio_avg": ("Nu_ratio_avg", "Nu_Nu0", "Nu_ratio"),
    "P_ratio": ("P_ratio", "P_P0", "pressure_ratio"),
    "T_ratio_hot": (
        "T_ratio_hot",
        "T_ratio_max",
        "T_ratio_max_Tmax_over_T0",
        "Tmax_T0",
        "Tmax_over_T0",
        "Tmax_over_T",
    ),
    "T_ratio_cold": (
        "T_ratio_cold",
        "T_ratio_min",
        "T_ratio_min_T0_over_Tmin",
        "T0_over_Tmin",
        "T0_Tmin",
        "T0_over_Tmin_ratio",
    ),
}


def _resolve_alias(columns: pd.Index, canonical: str) -> str | None:
    return next((name for name in _ALIASES[canonical] if name in columns), None)


def detect_case(frame: pd.DataFrame) -> str:
    """Detect ``single`` or ``h3`` from recognised input column aliases."""

    has_single = all(_resolve_alias(frame.columns, name) is not None for name in SINGLE_SCHEMA.feature_columns)
    has_h3 = all(_resolve_alias(frame.columns, name) is not None for name in H3_SCHEMA.feature_columns)
    if has_single and not has_h3:
        return "single"
    if has_h3 and not has_single:
        return "h3"
    if has_single and has_h3:
        raise ValueError("Dataset contains both single-bump and H3 feature schemas; specify case explicitly.")
    raise ValueError("Could not identify single-bump or homogeneous three-bump input columns.")


def standardize_dataframe(
    frame: pd.DataFrame,
    case: str | None = None,
    require_targets: bool = True,
) -> tuple[pd.DataFrame, DatasetSchema]:
    """Return a cleaned copy with canonical numeric columns; never modify input."""

    normalized_case = (case or detect_case(frame)).strip().lower()
    if normalized_case not in {"single", "h3"}:
        raise ValueError("case must be 'single' or 'h3'.")
    schema = SINGLE_SCHEMA if normalized_case == "single" else H3_SCHEMA
    canonical = frame.copy(deep=True)

    required = list(schema.feature_columns)
    if require_targets:
        required.extend(schema.target_columns)
    for name in required:
        source = _resolve_alias(frame.columns, name)
        if source is None:
            raise ValueError(f"Required column '{name}' was not found. Accepted aliases: {_ALIASES[name]}.")
        canonical[name] = pd.to_numeric(frame[source], errors="coerce")

    canonical = canonical.dropna(subset=required).reset_index(drop=True)
    if canonical.empty:
        raise ValueError("No complete numeric rows remain after standardisation.")
    return canonical, schema


def load_csv(
    path: str | Path,
    case: str | None = None,
    require_targets: bool = True,
    **read_csv_kwargs,
) -> tuple[pd.DataFrame, DatasetSchema]:
    """Read and standardise a CSV dataset without writing any files."""

    frame = pd.read_csv(Path(path), **read_csv_kwargs)
    return standardize_dataframe(frame, case=case, require_targets=require_targets)
