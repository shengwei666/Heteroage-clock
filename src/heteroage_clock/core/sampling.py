"""
heteroage_clock.core.sampling

Industrial-grade utilities for sampling and stratified splitting.
Updates: Added robustness for missing 'project_id' and optimized sampling logic.
"""

from typing import List, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict

def make_stratified_group_folds(groups: pd.Series, tissues: pd.Series, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified group folds ensuring no leakage. 
    Samples are grouped by `groups` (e.g., project) and stratified by `tissues`.

    Args:
        groups (pd.Series): Group labels (e.g., project_id).
        tissues (pd.Series): Tissue types.
        n_splits (int): Number of folds.
        seed (int): Random seed.

    Returns:
        List of (train_idx, val_idx) tuples.
    """
    # Ensure inputs are pandas Series for consistent handling
    if not isinstance(groups, pd.Series):
        groups = pd.Series(groups)
    if not isinstance(tissues, pd.Series):
        tissues = pd.Series(tissues)

    df_meta = pd.DataFrame({'group': groups.values, 'tissue': tissues.values})
    
    if df_meta.empty:
        return []
    
    # Get the most common tissue per project for stratified sampling
    # Fallback: if a group has multiple tissues, take the mode
    project_map = df_meta.groupby('group')['tissue'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]).reset_index()
    
    tissue_to_projs = defaultdict(list)
    for _, row in project_map.iterrows():
        tissue_to_projs[row['tissue']].append(row['group'])
    
    folds = [[] for _ in range(n_splits)]
    rng = np.random.RandomState(seed)
    
    # Sort keys for deterministic behavior before shuffling
    for t_type in sorted(tissue_to_projs.keys()):
        projs = tissue_to_projs[t_type]
        rng.shuffle(projs)
        for i, p_id in enumerate(projs):
            folds[i % n_splits].append(p_id)
    
    all_idx = np.arange(len(groups))
    cv_indices = []
    
    for i in range(n_splits):
        val_projects = set(folds[i]) # Use set for faster lookup
        # Boolean mask for validation
        is_val = df_meta['group'].isin(val_projects).values
        
        train_idx = all_idx[~is_val]
        val_idx = all_idx[is_val]
        cv_indices.append((train_idx, val_idx))
    
    return cv_indices


def adaptive_sampler(df: pd.DataFrame, min_coh: int, min_c: int, max_c: int, mult: float, seed: int) -> Tuple[pd.DataFrame, int]:
    """
    Adaptive sampler to balance samples across different tissue groups.
    Down-samples majority classes (like Blood) to `max_c` while keeping minority classes.

    Args:
        df (pd.DataFrame): Data containing 'Tissue' and optionally 'project_id'.
        min_coh (int): Min cohorts to keep a tissue.
        min_c (int): Min samples per tissue (soft floor).
        max_c (int): Max samples per tissue (hard cap).
        mult (float): Multiplier for median calculation.
        seed (int): Random seed.

    Returns:
        Tuple[pd.DataFrame, int]: Sampled DataFrame and the calculated target_n.
    """
    if df.empty:
        return df, 0
    
    # 1. Filter by Minimum Cohorts (if project_id exists)
    if 'project_id' in df.columns:
        cohort_stats = df.groupby('Tissue')['project_id'].nunique()
        valid_tissues = cohort_stats[cohort_stats >= min_coh].index
    else:
        # If no project_id, we cannot filter by cohort count, so keep all tissues
        valid_tissues = df['Tissue'].unique()
    
    if len(valid_tissues) == 0:
        return df.iloc[:0], 0
    
    df_f = df[df['Tissue'].isin(valid_tissues)].copy()
    
    # 2. Calculate Target N
    # Logic: Median * Multiplier, but clamped between [min_c, max_c]
    counts = df_f['Tissue'].value_counts()
    median_n = counts.median()
    if pd.isna(median_n): 
        median_n = 0
    
    # Calculate target based on median
    raw_target = median_n * mult
    target_n = int(np.clip(raw_target, min_c, max_c))
    
    # Safety: Strictly enforce max_c as the ceiling
    if target_n > max_c:
        target_n = max_c
        
    rng = np.random.RandomState(seed)
    selected_indices = []
    
    # 3. Perform Sampling
    for t in valid_tissues:
        # Get indices for this tissue in the filtered dataframe
        t_indices = df_f[df_f['Tissue'] == t].index.tolist()
        n_available = len(t_indices)
        
        # Determine how many to keep for this specific tissue
        # If n_available (15000) > target_n (500) -> Keep 500
        # If n_available (120) < target_n (500) -> Keep 120
        n_keep = min(n_available, target_n)
        
        # If we have more than we need, sample randomly
        if n_available > n_keep:
            selected = rng.choice(t_indices, size=n_keep, replace=False)
            selected_indices.extend(selected)
        else:
            # Keep all if we don't have enough to hit the cap
            selected_indices.extend(t_indices)
    
    # Return sampled dataframe and the target used
    return df_f.loc[selected_indices].reset_index(drop=True), target_n