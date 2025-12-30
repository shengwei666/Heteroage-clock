"""
heteroage_clock.core.metrics

Standard metrics for regression and biological age evaluation.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from scipy.stats import pearsonr
from typing import Dict

RegressionMetrics = Dict[str, float]

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """
    Compute standard regression metrics: MAE, R2, Pearson R, MedianAE.
    """
    # Ensure inputs are 1D numpy arrays to prevent shape mismatch warnings
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    
    # Pearson R handling for constant output edge case (prevent NaN)
    if len(y_pred) > 1 and np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
        pearson_r, _ = pearsonr(y_true, y_pred)
    else:
        pearson_r = 0.0
        
    return {
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "PearsonR": round(pearson_r, 4),
        "MedianAE": round(median_ae, 4)
    }

def heteroage_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Placeholder for custom metric calculation.
    """
    return mean_absolute_error(y_true, y_pred)