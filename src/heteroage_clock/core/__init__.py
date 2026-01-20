"""
heteroage_clock.core

Core functionality for metrics, optimization, and sampling.
"""

from .optimization import optimize_elasticnet_optuna
from .metrics import compute_regression_metrics
from .age_transform import AgeTransformer
from .splits import make_stratified_group_folds
from .sampling import adaptive_sampler

# Explicitly export these to make them available when importing from .core
__all__ = [
    "optimize_elasticnet_optuna",
    "compute_regression_metrics",
    "AgeTransformer",
    "make_stratified_group_folds",
    "adaptive_sampler"
]