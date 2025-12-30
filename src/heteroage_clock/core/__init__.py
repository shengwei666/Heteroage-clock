"""
heteroage_clock.core

Core mathematical and statistical components of the clock.
This module includes:
- Age transformation logic (Log-Linear)
- Performance metrics (MAE, R2, HeteroAge Score)
- Sampling strategies (Stratified Group K-Fold, Adaptive Sampling)
"""

from .age_transform import AgeTransformer
from .metrics import compute_regression_metrics, heteroage_score, RegressionMetrics
from .sampling import adaptive_sampler, make_stratified_group_folds

__all__ = [
    "AgeTransformer",
    "compute_regression_metrics",
    "heteroage_score",
    "RegressionMetrics",
    "adaptive_sampler",
    "make_stratified_group_folds",
]