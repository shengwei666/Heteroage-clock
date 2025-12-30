"""
heteroage_clock.core.splits

Utilities for splitting the dataset into training and validation folds.
These functions implement cross-validation strategies and data partitioning
while ensuring no data leakage between folds. Specifically, we use:

- **Stratified Group K-Folds**: Ensure that the splits are stratified based on group labels (e.g., `project_id`), and
  tissue types are balanced in each fold to avoid leakage.
- **Data partitioning based on project IDs**: This prevents overlap of data from the same project between training and
  validation sets.

This module contains functions for generating train/validation splits that are consistent with the original research
scripts and can be reused across all stages of the model pipeline.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from collections import defaultdict


def make_stratified_group_folds(groups: pd.Series, tissues: pd.Series, n_splits: int, seed: int) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate stratified group folds for cross-validation. The group split ensures no leakage from the same group (e.g., project_id)
    while keeping the tissue distribution balanced across folds.

    Args:
        groups (pd.Series): Group labels for the samples (e.g., project_id).
        tissues (pd.Series): Tissue labels for the samples (e.g., Brain, Blood, etc.).
        n_splits (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.

    Returns:
        Optional[List[Tuple[np.ndarray, np.ndarray]]]: List of tuples containing train and validation indices for each fold.
    """
    df_meta = pd.DataFrame({'project': groups, 'tissue': tissues})
    
    # Ensure there is data to work with
    if df_meta.empty:
        return None
    
    # Group the samples by project_id and ensure each project is assigned to a tissue group
    project_map = df_meta.groupby('project')['tissue'].agg(lambda x: x.mode()[0]).reset_index()
    tissue_to_projs = defaultdict(list)
    
    for _, row in project_map.iterrows():
        tissue_to_projs[row['tissue']].append(row['project'])
    
    folds = [[] for _ in range(n_splits)]
    rng = np.random.RandomState(seed)
    
    for t_type in sorted(tissue_to_projs.keys()):
        projs = tissue_to_projs[t_type]
        rng.shuffle(projs)
        for i, p_id in enumerate(projs): 
            folds[i % n_splits].append(p_id)
    
    # Create train/validation indices
    all_idx = np.arange(len(groups))
    cv_indices = []
    
    for i in range(n_splits):
        val_projects = folds[i]
        is_val = np.isin(groups, val_projects)
        train_idx = all_idx[~is_val]
        val_idx = all_idx[is_val]
        cv_indices.append((train_idx, val_idx))
        
    return cv_indices


def make_grouped_split(groups: pd.Series, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into n_splits using GroupKFold. Ensures that the same group does not appear in both training and validation
    sets within the same fold.

    Args:
        groups (pd.Series): Group labels (e.g., project_id).
        n_splits (int): Number of folds for the cross-validation.
        seed (int): Random seed for reproducibility.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of tuples containing training and validation indices.
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    # Convert group labels to numpy array
    groups_array = groups.values if isinstance(groups, pd.Series) else np.array(groups)
    
    # Generate indices for each fold
    cv_indices = [(train_idx, val_idx) for train_idx, val_idx in gkf.split(np.zeros(len(groups_array)), groups=groups_array)]
    
    return cv_indices


def stratified_train_test_split(df: pd.DataFrame, group_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and testing sets, ensuring that no leakage happens across the same group (e.g., project).
    This uses stratified splitting based on the group column.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        group_col (str): The column to use for stratified splitting (e.g., 'project_id').
        test_size (float): Fraction of data to reserve for the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    # Get the unique groups
    groups = df[group_col].values
    
    # Create the GroupKFold object
    gkf = GroupKFold(n_splits=int(1 / test_size))
    
    # Get the split
    for train_idx, test_idx in gkf.split(df, groups=groups):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        break
    
    return df_train, df_test
