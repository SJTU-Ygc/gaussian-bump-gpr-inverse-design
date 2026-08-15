"""Scaled Gaussian-process regression models for the four response ratios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .features import make_features
from .metrics import performance_evaluation_criterion, temperature_ratio

TARGET_COLUMNS = ("Nu_ratio_avg", "P_ratio", "T_ratio_hot", "T_ratio_cold")


@dataclass(frozen=True)
class GPRConfig:
    constant_bounds: tuple[float, float]
    length_scale_bounds: tuple[float, float]
    noise_level: float
    noise_bounds: tuple[float, float]
    alpha: float
    n_restarts_optimizer: int
    random_state: int = 42
    matern_nu: float = 2.5


@dataclass(frozen=True)
class PerformancePrediction:
    nu_ratio: float
    pressure_ratio: float
    hot_ratio: float
    cold_ratio: float
    temperature_ratio: float
    pec: float


def default_gpr_config(case: str, n_restarts_optimizer: int | None = None) -> GPRConfig:
    """Return the kernel settings used in the final single-bump or H3 scripts."""

    normalized = case.strip().lower()
    if normalized == "single":
        config = GPRConfig(
            constant_bounds=(1e-4, 1e4),
            length_scale_bounds=(1e-3, 1e3),
            noise_level=1e-5,
            noise_bounds=(1e-10, 1e-1),
            alpha=0.0,
            n_restarts_optimizer=15,
        )
    elif normalized == "h3":
        config = GPRConfig(
            constant_bounds=(1e-3, 1e3),
            length_scale_bounds=(1e-2, 1e3),
            noise_level=1e-6,
            noise_bounds=(1e-10, 1e-2),
            alpha=1e-6,
            n_restarts_optimizer=10,
        )
    else:
        raise ValueError("case must be 'single' or 'h3'.")
    if n_restarts_optimizer is None:
        return config
    if n_restarts_optimizer < 0:
        raise ValueError("n_restarts_optimizer must be non-negative.")
    return GPRConfig(**{**config.__dict__, "n_restarts_optimizer": int(n_restarts_optimizer)})


def validation_gpr_config(n_restarts_optimizer: int = 3) -> GPRConfig:
    """Settings used for the grouped validation reported in the paper."""

    return GPRConfig(
        constant_bounds=(1e-3, 1e3),
        length_scale_bounds=(1e-3, 1e3),
        noise_level=1e-5,
        noise_bounds=(1e-8, 1e-2),
        alpha=1e-10,
        n_restarts_optimizer=int(n_restarts_optimizer),
        random_state=42,
        matern_nu=1.5,
    )


class ScaledGPR:
    """Gaussian-process regressor with input and target standardisation."""

    def __init__(
        self,
        case: str,
        feature_mode: str | None = None,
        config: GPRConfig | None = None,
    ) -> None:
        normalized = case.strip().lower()
        if normalized not in {"single", "h3"}:
            raise ValueError("case must be 'single' or 'h3'.")
        self.case = normalized
        self.feature_mode = feature_mode or ("raw" if normalized == "single" else "engineered")
        self.config = config or default_gpr_config(normalized)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model: GaussianProcessRegressor | None = None

    def fit(self, designs: ArrayLike, targets: ArrayLike) -> "ScaledGPR":
        features = make_features(designs, self.case, self.feature_mode)
        y = np.asarray(targets, dtype=float).reshape(-1, 1)
        if features.shape[0] != y.shape[0] or y.shape[0] < 2:
            raise ValueError("Designs and targets must contain the same number of at least two samples.")
        if np.any(~np.isfinite(y)):
            raise ValueError("Targets contain non-finite values.")

        x_scaled = self.x_scaler.fit_transform(features)
        y_scaled = self.y_scaler.fit_transform(y).ravel()
        kernel = (
            ConstantKernel(1.0, self.config.constant_bounds)
            * Matern(
                length_scale=np.ones(features.shape[1]),
                length_scale_bounds=self.config.length_scale_bounds,
                nu=self.config.matern_nu,
            )
            + WhiteKernel(self.config.noise_level, self.config.noise_bounds)
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.alpha,
            normalize_y=False,
            n_restarts_optimizer=self.config.n_restarts_optimizer,
            random_state=self.config.random_state,
        )
        self.model.fit(x_scaled, y_scaled)
        return self

    def predict(
        self,
        designs: ArrayLike,
        return_std: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.model is None:
            raise RuntimeError("The model must be fitted before prediction.")
        features = make_features(designs, self.case, self.feature_mode)
        x_scaled = self.x_scaler.transform(features)
        if return_std:
            mean_scaled, std_scaled = self.model.predict(x_scaled, return_std=True)
            mean = self.y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
            std = std_scaled * float(self.y_scaler.scale_[0])
            return mean, std
        mean_scaled = self.model.predict(x_scaled)
        return self.y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()


class SurrogateBundle:
    """Four response models with derived temperature ratio and PEC."""

    def __init__(self, models: Mapping[str, ScaledGPR]) -> None:
        missing = [name for name in TARGET_COLUMNS if name not in models]
        if missing:
            raise ValueError(f"Missing surrogate models: {missing}.")
        self.models = dict(models)

    def predict_outputs(self, designs: ArrayLike) -> dict[str, NDArray[np.float64]]:
        outputs = {name: np.asarray(self.models[name].predict(designs), dtype=float) for name in TARGET_COLUMNS}
        outputs["Tmax_over_Tmin"] = np.asarray(
            temperature_ratio(outputs["T_ratio_hot"], outputs["T_ratio_cold"]), dtype=float
        )
        outputs["PEC"] = np.asarray(
            performance_evaluation_criterion(
                outputs["Nu_ratio_avg"],
                outputs["P_ratio"],
                outputs["T_ratio_hot"],
                outputs["T_ratio_cold"],
            ),
            dtype=float,
        )
        return outputs

    def predict_performance(self, design: ArrayLike) -> PerformancePrediction:
        design_array = np.asarray(design, dtype=float).reshape(1, -1)
        output = self.predict_outputs(design_array)
        return PerformancePrediction(
            nu_ratio=float(output["Nu_ratio_avg"][0]),
            pressure_ratio=float(output["P_ratio"][0]),
            hot_ratio=float(output["T_ratio_hot"][0]),
            cold_ratio=float(output["T_ratio_cold"][0]),
            temperature_ratio=float(output["Tmax_over_Tmin"][0]),
            pec=float(output["PEC"][0]),
        )


def fit_surrogate_bundle(
    designs: ArrayLike,
    targets: Mapping[str, ArrayLike],
    case: str,
    feature_mode: str | None = None,
    config: GPRConfig | None = None,
) -> SurrogateBundle:
    """Fit the four response models and return an in-memory bundle."""

    missing = [name for name in TARGET_COLUMNS if name not in targets]
    if missing:
        raise ValueError(f"Missing target arrays: {missing}.")
    models = {
        name: ScaledGPR(case, feature_mode=feature_mode, config=config).fit(designs, targets[name])
        for name in TARGET_COLUMNS
    }
    return SurrogateBundle(models)
