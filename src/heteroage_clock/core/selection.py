"""
heteroage_clock.core.selection

Feature selection algorithms.
Implements Correlation-based Ranking & Orthogonalization (Winner-Takes-All by Rank).
This ensures stable, deterministic feature assignment across Hallmarks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from heteroage_clock.utils.logging import log

def fast_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate Pearson correlation coefficient between each column of X and y.
    Vectorized implementation for high performance.
    
    Args:
        X: (n_samples, n_features)
        y: (n_samples,)
    
    Returns:
        Array of correlations (n_features,)
    """
    n = X.shape[0]
    if n < 2:
        return np.zeros(X.shape[1])

    # Center inputs
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - X_mean
    y_centered = y - y_mean
    
    # Standard deviation
    X_std = X.std(axis=0)
    y_std = y.std()
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Covariance * n / (n * std_x * std_y) -> Covariance / (std_x * std_y)
        # Note: np.dot(u, v) is sum(u_i * v_i), which is numerator of Pearson
        numerator = np.dot(X_centered.T, y_centered)
        denominator = n * X_std * y_std + 1e-12
        corrs = numerator / denominator
    
    return np.nan_to_num(corrs, nan=0.0)

def orthogonalize_by_correlation(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: List[str], 
    hallmark_mapping: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Performs orthogonalization based on Correlation Ranking.
    
    Logic:
    1. Calculate correlation of ALL features with y (Age/Residual).
    2. For each Hallmark, rank its features by abs(correlation).
    3. If a feature appears in multiple Hallmarks, assign it to the one 
       where it has the highest relative rank (smallest rank index).
       
    This provides a STABLE definition of Hallmarks.
    """
    # Sanity check
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Feature count mismatch: X has {X.shape[1]}, names has {len(feature_names)}")

    log("  > Calculating global correlations for ranking...")
    
    # 1. Global Correlation Calculation
    corrs = fast_correlation(X, y)
    abs_corrs = np.abs(corrs)
    
    # Map feature name -> correlation score
    feat_score_map = dict(zip(feature_names, abs_corrs))
    
    # 2. Calculate Rank within each Hallmark
    # feature -> list of (hallmark, rank)
    feature_ranks = defaultdict(list)
    
    for h, feats in hallmark_mapping.items():
        # Only consider features present in our dataset
        valid_feats = [f for f in feats if f in feat_score_map]
        
        if not valid_feats:
            continue

        # Sort features by correlation (descending) -> Best feature is Rank 0
        sorted_feats = sorted(valid_feats, key=lambda f: feat_score_map[f], reverse=True)
        
        # Record rank (0-based index)
        for rank, f in enumerate(sorted_feats):
            feature_ranks[f].append((h, rank))
            
    # 3. Resolve Conflicts (Winner-Takes-All by Rank)
    final_ortho_dict = defaultdict(list)
    
    for f, possibilities in feature_ranks.items():
        if len(possibilities) == 1:
            # No conflict
            best_h = possibilities[0][0]
        else:
            # Conflict: Choose the hallmark where this feature ranks highest (smallest rank index)
            # Tie-breaker: sort by hallmark name (x[0]) to be deterministic if ranks are equal
            best_h = min(possibilities, key=lambda x: (x[1], x[0]))[0]
            
        final_ortho_dict[best_h].append(f)
        
    return dict(final_ortho_dict)

# Compatibility stub (Optional, keeps old scripts from breaking)
def select_features_internal(*args, **kwargs):
    raise NotImplementedError("Use orthogonalize_by_correlation instead.")