"""
heteroage_clock.core.sampling

Industrial-grade utilities for sampling and stratified splitting, ensuring leakage-free and heterogeneity-aware splits.

Design goals:
- Stratified sampling based on project and tissue type to prevent leakage.
- Robust handling of different group sizes and sample imbalance.
- Efficient data handling with compatibility across stages.

This file contains functions for:
- Stratified K-Fold splitting by project and tissue group (avoid leakage).
- Adaptive sampling for balancing data based on sample size constraints (min/max).
"""

from typing import List, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


def make_stratified_group_folds(groups: pd.Series, tissues: pd.Series, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified group folds ensuring no leakage. Samples are grouped by `groups` (e.g., project) and stratified by `tissues`.

    Args:
        groups (pd.Series): Group labels (e.g., project_id) to ensure leakage-free splits.
        tissues (pd.Series): Tissue types for stratified splitting.
        n_splits (int): Number of folds for the stratified splitting.
        seed (int): Random seed for reproducibility.

    Returns:
        List of tuples: Each tuple contains two arrays (train indices, validation indices).
    """
    df_meta = pd.DataFrame({'group': groups, 'tissue': tissues})
    
    if df_meta.empty:
        return []
    
    # Get the most common tissue per project for stratified sampling
    project_map = df_meta.groupby('group')['tissue'].agg(lambda x: x.mode()[0]).reset_index()
    tissue_to_projs = defaultdict(list)
    
    for _, row in project_map.iterrows():
        tissue_to_projs[row['tissue']].append(row['group'])
    
    folds = [[] for _ in range(n_splits)]
    rng = np.random.RandomState(seed)
    
    for t_type in sorted(tissue_to_projs.keys()):
        projs = tissue_to_projs[t_type]
        rng.shuffle(projs)
        for i, p_id in enumerate(projs):
            folds[i % n_splits].append(p_id)
    
    all_idx = np.arange(len(groups))
    cv_indices = []
    
    for i in range(n_splits):
        val_projects = folds[i]
        is_val = np.isin(groups, val_projects)
        train_idx = all_idx[~is_val]
        val_idx = all_idx[is_val]
        cv_indices.append((train_idx, val_idx))
    
    return cv_indices


def adaptive_sampler(df: pd.DataFrame, min_coh: int, min_c: int, max_c: int, mult: float, seed: int) -> Tuple[pd.DataFrame, int]:
    """
    Adaptive sampler to balance samples across different tissue groups with size constraints. 
    Ensures each tissue group in the dataframe has a minimum sample size (min_coh), 
    and adjusts the sample size per tissue based on the specified multiplier.

    Args:
        df (pd.DataFrame): The data to sample from.
        min_coh (int): Minimum number of cohorts per tissue.
        min_c (int): Minimum number of samples per tissue.
        max_c (int): Maximum number of samples per tissue.
        mult (float): Multiplier to scale the number of samples per tissue.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple: A DataFrame containing the sampled data and the number of samples selected.
    """
    if df.empty:
        return df, 0
    
    cohort_stats = df.groupby('Tissue')['project_id'].nunique()
    valid_tissues = cohort_stats[cohort_stats >= min_coh].index
    
    if len(valid_tissues) == 0:
        return df.iloc[:0], 0
    
    df_f = df[df['Tissue'].isin(valid_tissues)].copy()
    median_n = df_f['Tissue'].value_counts().median()
    if pd.isna(median_n):
        median_n = 0
    
    target_n = int(np.clip(median_n * mult, min_c, max_c))
    
    rng = np.random.RandomState(seed)
    selected_indices = []
    
    for t in valid_tissues:
        t_idx = df_f[df_f['Tissue'] == t].index
        selected = rng.choice(t_idx, size=min(len(t_idx), target_n), replace=False)
        selected_indices.extend(selected.tolist())
    
    return df_f.loc[selected_indices].reset_index(drop=True), target_n
