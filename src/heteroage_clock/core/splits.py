"""
heteroage_clock.core.splits

Splitting strategies for leakage-free validation.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple, Any

def make_stratified_group_folds(
    groups: Any, 
    tissues: Any, 
    n_splits: int = 5, 
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates folds that are stratified by Tissue and grouped by Project/Subject.
    
    Args:
        groups: Group identifiers (e.g. project_id) to keep distinct.
        tissues: Class labels for stratification (e.g. Tissue type).
        n_splits: Number of folds.
        seed: Random seed.
        
    Returns:
        List of (train_idx, val_idx) tuples.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Create dummy X for the split method (it only needs length)
    X_dummy = np.zeros((len(groups), 1))
    
    folds = list(sgkf.split(X_dummy, y=tissues, groups=groups))
    return folds