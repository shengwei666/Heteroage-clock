"""
heteroage_clock.core

Core mathematical and statistical components of the clock.
"""

from .age_transform import AgeTransformer
from .metrics import compute_regression_metrics, heteroage_score, RegressionMetrics
# [FIX] Import from splits.py (the source of truth), not sampling.py
from .splits import make_stratified_group_folds 

__all__ = [
    "AgeTransformer",
    "compute_regression_metrics",
    "heteroage_score",
    "RegressionMetrics",
    "make_stratified_group_folds",
]