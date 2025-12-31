"""
heteroage_clock.core

Core mathematical and statistical components of the clock.
"""

from .age_transform import AgeTransformer
from .metrics import compute_regression_metrics, heteroage_score, RegressionMetrics
from .splits import make_stratified_group_folds 
# [KEEP] Ensure optimization is available if users want to call it directly
from .optimization import tune_elasticnet_macro_micro

__all__ = [
    "AgeTransformer",
    "compute_regression_metrics",
    "heteroage_score",
    "RegressionMetrics",
    "make_stratified_group_folds",
    "tune_elasticnet_macro_micro",
]