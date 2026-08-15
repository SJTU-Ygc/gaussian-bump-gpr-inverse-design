"""Surrogate modelling and inverse design for Gaussian interfacial bumps."""

from .datasets import DatasetSchema, detect_case, load_csv, standardize_dataframe
from .features import h3_features, make_features, single_bump_features
from .inverse_query import DirectQueryResult, direct_inverse_query, refine_inverse_query
from .metrics import (
    pec_standard_deviation,
    performance_evaluation_criterion,
    regression_metrics,
    temperature_ratio,
)
from .optimization import (
    ConstraintTargets,
    OptimizationResult,
    optimize_design,
    optimize_unconstrained,
    screen_initial_designs,
    structured_grid,
    trace_constraint_path,
)
from .surrogate import (
    GPRConfig,
    PerformancePrediction,
    ScaledGPR,
    SurrogateBundle,
    default_gpr_config,
    fit_surrogate_bundle,
    validation_gpr_config,
)
from .validation import CrossValidationResult, build_group_splits, cross_validate_surrogates

__all__ = [
    "ConstraintTargets",
    "CrossValidationResult",
    "DatasetSchema",
    "DirectQueryResult",
    "GPRConfig",
    "OptimizationResult",
    "PerformancePrediction",
    "ScaledGPR",
    "SurrogateBundle",
    "build_group_splits",
    "cross_validate_surrogates",
    "default_gpr_config",
    "detect_case",
    "direct_inverse_query",
    "fit_surrogate_bundle",
    "h3_features",
    "load_csv",
    "make_features",
    "optimize_design",
    "optimize_unconstrained",
    "pec_standard_deviation",
    "performance_evaluation_criterion",
    "refine_inverse_query",
    "regression_metrics",
    "screen_initial_designs",
    "single_bump_features",
    "standardize_dataframe",
    "structured_grid",
    "temperature_ratio",
    "trace_constraint_path",
    "validation_gpr_config",
]

__version__ = "1.0.0"
