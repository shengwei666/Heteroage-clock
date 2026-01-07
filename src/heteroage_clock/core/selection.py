"""
heteroage_clock.core.selection

Feature selection algorithms.
Implements Correlation-based Ranking & Orthogonalization (Winner-Takes-All by Rank).
Updates:
- Optimized 'fast_correlation' to use batched processing.
- Prevents OOM when calculating correlations on large Memmap arrays.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from heteroage_clock.utils.logging import log

def fast_correlation(X: np.ndarray, y: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """
    Calculate Pearson correlation coefficient between each column of X and y.
    
    Memory-Efficient Implementation:
    - Processes features in batches to avoid allocating a huge 'X_centered' matrix.
    - Safe to use with large Memmap arrays.
    
    Args:
        X: (n_samples, n_features)
        y: (n_samples,)
        batch_size: Number of features to process at once.
    
    Returns:
        Array of correlations (n_features,)
    """
    n_samples, n_features = X.shape
    if n_samples < 2:
        return np.zeros(n_features, dtype=np.float32)

    # 1. Pre-compute y statistics (small vector, safe in RAM)
    y = y.astype(np.float32)
    y_mean = y.mean()
    y_centered = y - y_mean
    # Denominator part for y: sum((y-y_mean)^2)
    y_ss = np.sum(y_centered ** 2)
    
    corrs = np.zeros(n_features, dtype=np.float32)
    
    # 2. Process X in batches of FEATURES (columns)
    # This avoids creating a (n_samples, n_features) matrix of centered X in RAM.
    # For a 30k sample x 200k feature matrix, this is the difference between 
    # crashing (24GB+ needed) vs using <100MB RAM.
    
    for start in range(0, n_features, batch_size):
        end = min(start + batch_size, n_features)
        
        # Load only a slice of columns into memory
        # If X is a memmap, this reads from disk.
        # X_batch shape: (n_samples, current_batch_size)
        X_batch = X[:, start:end].astype(np.float32)
        
        # Center X batch
        X_mean_batch = X_batch.mean(axis=0)
        X_centered_batch = X_batch - X_mean_batch
        
        # Compute Numerator: dot(y_centered, X_centered)
        # y_centered is (n_samples,), X_centered_batch is (n_samples, batch)
        # Result is (batch,)
        numerators = np.dot(y_centered, X_centered_batch)
        
        # Compute Denominator: sqrt(sum(x_c^2) * sum(y_c^2))
        X_ss_batch = np.sum(X_centered_batch ** 2, axis=0)
        denominators = np.sqrt(X_ss_batch * y_ss)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            batch_corrs = numerators / (denominators + 1e-12)
        
        # Store results
        corrs[start:end] = np.nan_to_num(batch_corrs, nan=0.0)
        
    return corrs

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

    log("  > Calculating global correlations for ranking (Batch mode)...")
    
    # 1. Global Correlation Calculation (Memory Optimized)
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

# Compatibility stub
def select_features_internal(*args, **kwargs):
    raise NotImplementedError("Use orthogonalize_by_correlation instead.")