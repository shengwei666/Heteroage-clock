"""
heteroage_clock.core.selection

Feature selection algorithms, specifically for Hallmark Orthogonalization.
"""

import numpy as np
from typing import List, Dict
from sklearn.linear_model import ElasticNetCV
from heteroage_clock.utils.logging import log

def select_features_internal(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: List[str], 
    hallmark_mapping: Dict[str, List[str]]
) -> List[str]:
    """
    Selects orthogonal features for each hallmark using ElasticNet.
    
    Args:
        X: Feature matrix (N_samples, N_features).
        y: Target vector.
        feature_names: List of feature names corresponding to X columns.
        hallmark_mapping: Dict mapping Hallmark_Name -> List[Feature_Names_with_Suffix].
        
    Returns:
        List[str]: The list of selected (orthogonal) feature names.
    """
    selected_features_set = set()
    
    # Map feature name to column index for fast access
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    
    for hallmark, feats in hallmark_mapping.items():
        # Find indices for this hallmark's features
        indices = [name_to_idx[f] for f in feats if f in name_to_idx]
        
        if not indices:
            continue
            
        X_sub = X[:, indices]
        
        # Use a lighter setup for rapid selection
        selector = ElasticNetCV(cv=3, random_state=42, n_jobs=-1, max_iter=1000)
        selector.fit(X_sub, y)
        
        # Extract non-zero coefs
        coefs = selector.coef_
        selected_mask = coefs != 0
        
        # Map back to names
        subset_names = [feats[i] for i, is_selected in enumerate(selected_mask) if is_selected]
        
        if len(subset_names) > 0:
            log(f"  > Hallmark '{hallmark}': Selected {len(subset_names)} / {len(feats)} features.")
            selected_features_set.update(subset_names)
        
    return list(selected_features_set)