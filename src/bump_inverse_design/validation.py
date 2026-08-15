"""Grouped validation used for the surrogate models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from .datasets import H3_SCHEMA, SINGLE_SCHEMA
from .metrics import pec_standard_deviation, performance_evaluation_criterion, regression_metrics
from .surrogate import GPRConfig, ScaledGPR, validation_gpr_config


PHYSICAL_TARGETS = ("Nu_ratio_avg", "P_ratio", "T_ratio_hot", "T_ratio_cold")


@dataclass(frozen=True)
class CrossValidationResult:
    fold_ids: NDArray[np.int64]
    predictions: dict[str, NDArray[np.float64]]
    standard_deviations: dict[str, NDArray[np.float64]]
    metrics: tuple[dict[str, float | int | str], ...]


def _level_rank(values: pd.Series) -> NDArray[np.int64]:
    levels = sorted(pd.unique(values.astype(float)))
    mapping = {value: index for index, value in enumerate(levels)}
    return values.astype(float).map(mapping).to_numpy(dtype=int)


def _boundary_mask(frame: pd.DataFrame, columns: Iterable[str]) -> NDArray[np.bool_]:
    mask = np.zeros(len(frame), dtype=bool)
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        mask |= np.isclose(values, values.min()) | np.isclose(values, values.max())
    return mask


def _group_splits(groups: NDArray, n_splits: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    effective = min(int(n_splits), np.unique(groups).size)
    if effective < 2:
        raise ValueError("Grouped validation requires at least two groups.")
    index = np.arange(groups.size)
    splitter = GroupKFold(n_splits=effective)
    return [(train.astype(int), test.astype(int)) for train, test in splitter.split(index, groups=groups)]


def build_group_splits(
    frame: pd.DataFrame,
    case: str,
    mode: str = "run_group",
    n_splits: int = 5,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Create the Group-CV partitions described in the Supplementary Information."""

    case = case.strip().lower()
    mode = mode.strip().lower()
    if case == "single":
        columns = SINGLE_SCHEMA.feature_columns
    elif case == "h3":
        columns = H3_SCHEMA.feature_columns
    else:
        raise ValueError("case must be 'single' or 'h3'.")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}.")

    if mode == "boundary_holdout":
        boundary = _boundary_mask(frame, columns)
        train = np.flatnonzero(~boundary)
        test = np.flatnonzero(boundary)
        if train.size > 5 and test.size:
            return [(train.astype(int), test.astype(int))]
        mode = "checkerboard"

    if mode == "checkerboard":
        ranks = [_level_rank(frame[column]) for column in columns]
        groups = (np.sum(np.column_stack(ranks), axis=1) % int(n_splits)).astype(str)
        return _group_splits(groups, n_splits)

    if case == "single":
        if mode in {"run_group", "leave_width"}:
            groups = ("w_" + frame["eps_w"].round(8).astype(str)).to_numpy()
        elif mode == "leave_height":
            groups = ("h_" + frame["eps_h"].round(8).astype(str)).to_numpy()
        elif mode == "spacing":
            raise ValueError("Spacing validation applies only to H3 data.")
        else:
            raise ValueError(f"Unknown validation mode: {mode}.")
    else:
        if mode == "run_group":
            groups = (
                "s_"
                + frame["s_mm"].round(8).astype(str)
                + "_w_"
                + frame["alpha_w"].round(8).astype(str)
            ).to_numpy()
        elif mode == "spacing":
            groups = ("s_" + frame["s_mm"].round(8).astype(str)).to_numpy()
        elif mode == "leave_width":
            groups = ("w_" + frame["alpha_w"].round(8).astype(str)).to_numpy()
        elif mode == "leave_height":
            groups = ("h_" + frame["alpha_h"].round(8).astype(str)).to_numpy()
        else:
            raise ValueError(f"Unknown validation mode: {mode}.")
    return _group_splits(groups, n_splits)


def _metric_row(target: str, fold: int | str, train_size: int, truth, prediction):
    row: dict[str, float | int | str] = {
        "target": target,
        "fold": fold,
        "n_train": int(train_size),
        "n_test": int(len(truth)),
    }
    row.update(regression_metrics(truth, prediction))
    return row


def cross_validate_surrogates(
    frame: pd.DataFrame,
    case: str,
    mode: str = "run_group",
    target_columns: Iterable[str] = PHYSICAL_TARGETS,
    n_splits: int = 5,
    include_direct_pec: bool = True,
    config: GPRConfig | None = None,
) -> CrossValidationResult:
    """Run Group-CV and reconstruct PEC from out-of-fold physical predictions."""

    case = case.strip().lower()
    if case not in {"single", "h3"}:
        raise ValueError("case must be 'single' or 'h3'.")
    feature_columns = SINGLE_SCHEMA.feature_columns if case == "single" else H3_SCHEMA.feature_columns
    physical_targets = tuple(target_columns)
    missing = [column for column in (*feature_columns, *physical_targets) if column not in frame]
    if missing:
        raise ValueError(f"Missing cross-validation columns: {missing}.")
    can_recompute_pec = set(PHYSICAL_TARGETS).issubset(physical_targets)
    if include_direct_pec and not can_recompute_pec:
        raise ValueError("Direct PEC validation requires all four physical targets.")

    designs = frame.loc[:, feature_columns].to_numpy(dtype=float)
    observed = {name: frame[name].to_numpy(dtype=float) for name in physical_targets}
    if can_recompute_pec:
        observed["PEC_recomputed_from_components"] = np.asarray(
            performance_evaluation_criterion(
                observed["Nu_ratio_avg"],
                observed["P_ratio"],
                observed["T_ratio_hot"],
                observed["T_ratio_cold"],
            ),
            dtype=float,
        )
    model_targets = dict(observed)
    model_targets.pop("PEC_recomputed_from_components", None)
    if include_direct_pec:
        if "PEC" in frame and np.all(np.isfinite(pd.to_numeric(frame["PEC"], errors="coerce"))):
            model_targets["PEC_direct"] = pd.to_numeric(frame["PEC"], errors="coerce").to_numpy(dtype=float)
        else:
            model_targets["PEC_direct"] = observed["PEC_recomputed_from_components"].copy()
        observed["PEC_direct"] = model_targets["PEC_direct"]

    result_names = list(model_targets)
    if can_recompute_pec:
        result_names.extend(["Tmax_over_Tmin", "PEC_recomputed_from_components"])
    predictions = {name: np.full(len(frame), np.nan) for name in result_names}
    deviations = {name: np.full(len(frame), np.nan) for name in result_names}
    fold_ids = np.full(len(frame), -1, dtype=int)
    rows: list[dict[str, float | int | str]] = []
    splits = build_group_splits(frame, case, mode, n_splits)
    model_config = config or validation_gpr_config()

    for fold, (train, test) in enumerate(splits, start=1):
        fold_ids[test] = fold
        for target, values in model_targets.items():
            model = ScaledGPR(case, feature_mode="validation", config=model_config).fit(
                designs[train], values[train]
            )
            mean, std = model.predict(designs[test], return_std=True)
            predictions[target][test] = mean
            deviations[target][test] = std
            rows.append(_metric_row(target, fold, train.size, values[test], mean))

        if can_recompute_pec:
            hot = predictions["T_ratio_hot"][test]
            cold = predictions["T_ratio_cold"][test]
            hot_std = deviations["T_ratio_hot"][test]
            cold_std = deviations["T_ratio_cold"][test]
            predictions["Tmax_over_Tmin"][test] = hot * cold
            deviations["Tmax_over_Tmin"][test] = np.abs(hot * cold) * np.sqrt(
                (hot_std / np.maximum(np.abs(hot), 1.0e-12)) ** 2
                + (cold_std / np.maximum(np.abs(cold), 1.0e-12)) ** 2
            )
            predictions["PEC_recomputed_from_components"][test] = performance_evaluation_criterion(
                predictions["Nu_ratio_avg"][test], predictions["P_ratio"][test], hot, cold
            )
            deviations["PEC_recomputed_from_components"][test] = pec_standard_deviation(
                predictions["Nu_ratio_avg"][test],
                deviations["Nu_ratio_avg"][test],
                predictions["P_ratio"][test],
                deviations["P_ratio"][test],
                hot,
                hot_std,
                cold,
                cold_std,
            )
            rows.append(
                _metric_row(
                    "PEC_recomputed_from_components",
                    fold,
                    train.size,
                    observed["PEC_recomputed_from_components"][test],
                    predictions["PEC_recomputed_from_components"][test],
                )
            )

    metric_targets = list(model_targets)
    if can_recompute_pec:
        metric_targets.append("PEC_recomputed_from_components")
    for target in metric_targets:
        evaluated = np.isfinite(predictions[target])
        rows.append(
            _metric_row(
                target,
                "overall",
                0,
                observed[target][evaluated],
                predictions[target][evaluated],
            )
        )
    return CrossValidationResult(fold_ids, predictions, deviations, tuple(rows))
