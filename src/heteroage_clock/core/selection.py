"""
heteroage_clock.core.selection

Utility functions for feature selection and importance ranking.

Design goals:
- Ensure consistent feature selection across stages.
- Leverage statistical and machine learning methods to select the most predictive features.
- Facilitate model interpretability by ranking features based on importance.

This file contains functions for:
- Feature selection using correlation and importance ranking.
- Selecting orthogonal features across different stages.
- Ranking features based on their predictive ability.
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import ElasticNetCV


def fast_vectorized_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    High-performance Pearson correlation using NumPy matrix operations.

    Args:
        X (np.ndarray): Feature matrix (samples x features).
        y (np.ndarray): Target vector (samples).

    Returns:
        np.ndarray: Correlation coefficients for each feature.
    """
    n = X.shape[0]
    if n < 2:
        return np.zeros(X.shape[1])
        
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - X_mean
    y_centered = y - y_mean
    
    X_std = X.std(axis=0)
    y_std = y.std()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = np.dot(X_centered.T, y_centered) / (n * X_std * y_std + 1e-12)
        corrs = np.abs(np.nan_to_num(corrs, nan=0.0))
        
    return corrs


def select_features_internal(X_train: np.ndarray, y_train: np.ndarray, candidate_names: List[str], hallmark_dict_suffixed: Dict[str, List[str]]) -> List[str]:
    """
    Select features for the model based on their correlation with the target.

    Args:
        X_train (np.ndarray): Training feature matrix (samples x features).
        y_train (np.ndarray): Training target vector (samples).
        candidate_names (List[str]): List of feature names corresponding to X_train.
        hallmark_dict_suffixed (Dict[str, List[str]]): Dictionary containing hallmark features for selection.

    Returns:
        List[str]: Selected feature names.
    """
    corrs = fast_vectorized_correlation(X_train, y_train)
    scores = dict(zip(candidate_names, corrs))
    
    hall_ranks = {}
    for h, feats in hallmark_dict_suffixed.items():
        if h == "Global_Union":
            continue
        valid_feats = [f for f in feats if f in scores]
        sorted_feats = sorted(valid_feats, key=lambda x: scores[x], reverse=True)
        if sorted_feats:
            hall_ranks[h] = {f: idx for idx, f in enumerate(sorted_feats)}
    
    ortho_map = defaultdict(list)
    conflict_map = defaultdict(list)
    
    for h, ranks in hall_ranks.items():
        for f in ranks: 
            conflict_map[f].append(h)
            
    final_selected_features = set()
    for f, h_list in conflict_map.items():
        best_h = min(h_list, key=lambda h: hall_ranks[h][f])
        ortho_map[best_h].append(f)
        final_selected_features.add(f)
        
    return list(final_selected_features)


def select_features_with_elastic_net(X_train: np.ndarray, y_train: np.ndarray, candidate_names: List[str], l1_ratio: float = 0.1) -> List[str]:
    """
    Select features based on ElasticNet regularization. This method will select the features that have non-zero coefficients
    after training a regularized regression model.

    Args:
        X_train (np.ndarray): Training feature matrix (samples x features).
        y_train (np.ndarray): Training target vector (samples).
        candidate_names (List[str]): List of feature names corresponding to X_train.
        l1_ratio (float): ElasticNet mixing parameter for L1 (Lasso) vs. L2 (Ridge) regularization. Default is 0.1.

    Returns:
        List[str]: List of selected feature names based on non-zero coefficients.
    """
    model = ElasticNetCV(l1_ratio=l1_ratio, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    
    selected_features = [candidate_names[i] for i, coef in enumerate(model.coef_) if coef != 0]
    
    return selected_features


def select_orthogonal_features(X_train: np.ndarray, y_train: np.ndarray, candidate_names: List[str], hallmark_dict_suffixed: Dict[str, List[str]], l1_ratio: float = 0.1) -> List[str]:
    """
    Select orthogonal features for training by removing correlated features and ensuring minimal multicollinearity.

    Args:
        X_train (np.ndarray): Training feature matrix (samples x features).
        y_train (np.ndarray): Training target vector (samples).
        candidate_names (List[str]): List of feature names corresponding to X_train.
        hallmark_dict_suffixed (Dict[str, List[str]]): Dictionary containing hallmark features for selection.
        l1_ratio (float): ElasticNet mixing parameter for L1 (Lasso) vs. L2 (Ridge) regularization. Default is 0.1.

    Returns:
        List[str]: List of selected orthogonal features.
    """
    selected_features = select_features_internal(X_train, y_train, candidate_names, hallmark_dict_suffixed)
    
    if not selected_features:
        return selected_features
    
    X_train_selected = X_train[:, [candidate_names.index(f) for f in selected_features]]
    
    # Use ElasticNet to select orthogonal features
    orthogonal_features = select_features_with_elastic_net(X_train_selected, y_train, selected_features, l1_ratio)
    
    return orthogonal_features
