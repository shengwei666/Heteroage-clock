"""
heteroage_clock.core.optimization

Hyperparameter optimization routines.
Specifically implements the 'Macro + Micro' strategy.
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from joblib import Parallel, delayed
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.utils.logging import log
from typing import Optional, Any, List, Union

def _evaluate_config(alpha: float, l1: float, X, y, folds, trans_func, seed, search_max_iter):
    """
    Internal helper: Evaluates a single alpha/l1 configuration 
    using the Micro + Macro correlation score.
    """
    oof_preds = np.zeros(len(y))
    fold_corrs = []
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=seed, max_iter=search_max_iter)
    
    for train_idx, val_idx in folds:
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        
        # Apply inverse transform for evaluation if needed
        pred_eval = trans_func.inverse_transform(pred) if trans_func else pred
        y_val_eval = trans_func.inverse_transform(y[val_idx]) if trans_func else y[val_idx]
        
        oof_preds[val_idx] = pred_eval
        
        if np.std(pred_eval) > 1e-9 and np.std(y_val_eval) > 1e-9:
            fold_corrs.append(np.corrcoef(y_val_eval, pred_eval)[0, 1])
        else:
            fold_corrs.append(0.0)
            
    macro_r = np.mean(fold_corrs)
    y_eval_all = trans_func.inverse_transform(y) if trans_func else y
    micro_r = np.corrcoef(y_eval_all, oof_preds)[0, 1] if np.std(oof_preds) > 1e-9 else 0.0
    
    return (micro_r + macro_r), alpha, l1

def tune_elasticnet_macro_micro(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: Any, 
    tissues: Any, 
    trans_func: Optional[Any] = None, 
    alphas: Optional[List[float]] = None,
    l1_ratios: Optional[List[float]] = None,
    n_jobs: int = -1,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 2000
) -> ElasticNet:
    """
    Grid search for best ElasticNet alpha that maximizes (Micro_R + Macro_R).
    Supports list inputs and parallel execution.
    """
    if alphas is None:
        alphas = np.logspace(-4, -0.5, 30).tolist()
    if l1_ratios is None:
        l1_ratios = [0.5]
    
    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
    
    if not folds:
        log("Warning: Split failed in optimization. Returning default model.")
        return ElasticNet(alpha=0.01, l1_ratio=l1_ratios[0], random_state=seed)

    param_grid = [(a, l) for a in alphas for l in l1_ratios]
    log(f"Starting Grid Search: {len(param_grid)} configurations using {n_jobs} jobs...")
    
    search_max_iter = max(1000, max_iter // 2)

    # Parallel grid search
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_config)(a, l, X, y, folds, trans_func, seed, search_max_iter)
        for a, l in param_grid
    )
    
    # Select the best performing configuration
    best_score, best_alpha, best_l1 = max(results, key=lambda x: x[0])
            
    log(f"  > Best Alpha: {best_alpha:.5f}, Best L1: {best_l1:.2f} (Score: {best_score:.4f})")
    return ElasticNet(alpha=best_alpha, l1_ratio=best_l1, random_state=seed, max_iter=max_iter)