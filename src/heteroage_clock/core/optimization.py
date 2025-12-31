"""
heteroage_clock.core.optimization

Hyperparameter optimization routines.
Specifically implements the 'Macro + Micro' strategy.
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.utils.logging import log
from typing import Optional, Any

def tune_elasticnet_macro_micro(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: Any, 
    tissues: Any, 
    trans_func: Optional[Any] = None, 
    seed: int = 42
) -> ElasticNet:
    """
    Performs Grid Search to find the ElasticNet alpha that maximizes (Micro_R + Macro_R).
    """
    alphas = np.logspace(-4, -0.5, 30) 
    l1_ratio = 0.5
    
    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=5, seed=seed)
    
    if not folds:
        log("Warning: Split failed in optimization. Returning default model.")
        return ElasticNet(alpha=0.01, l1_ratio=l1_ratio, random_state=seed)

    best_score = -np.inf
    best_alpha = alphas[len(alphas)//2]
    
    for alpha in alphas:
        oof_preds = np.zeros(len(y))
        fold_corrs = []
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=1000)
        
        for train_idx, val_idx in folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            if trans_func:
                pred_eval = trans_func.inverse_transform(pred)
                y_val_eval = trans_func.inverse_transform(y_val)
            else:
                pred_eval = pred
                y_val_eval = y_val
            
            oof_preds[val_idx] = pred_eval
            
            if np.std(pred_eval) > 1e-9 and np.std(y_val_eval) > 1e-9:
                r_fold = np.corrcoef(y_val_eval, pred_eval)[0, 1]
            else:
                r_fold = 0.0
            fold_corrs.append(r_fold)
            
        macro_r = np.mean(fold_corrs)
        
        if trans_func:
            y_eval_all = trans_func.inverse_transform(y)
        else:
            y_eval_all = y
            
        if np.std(oof_preds) > 1e-9:
            micro_r = np.corrcoef(y_eval_all, oof_preds)[0, 1]
        else:
            micro_r = 0.0
        
        combined_score = micro_r + macro_r
        
        if combined_score > best_score:
            best_score = combined_score
            best_alpha = alpha
            
    log(f"  > Best Alpha: {best_alpha:.5f} (Score: {best_score:.4f})")
    return ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=2000)