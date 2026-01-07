"""
heteroage_clock.core.optimization

Hyperparameter optimization routines.
Updates:
- Added real-time logging (MAE, MedAE).
- [Critical] Added batched prediction to prevent OOM when using Memmap data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, median_absolute_error
from joblib import Parallel, delayed
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.sampling import adaptive_sampler
from heteroage_clock.utils.logging import log
from typing import Optional, Any, List, Union, Tuple

def _batched_predict(model, X, indices, batch_size=2048):
    """
    Helper to predict in chunks.
    This is crucial when X is a memmap, to avoid loading the entire validation set (X[indices]) into RAM at once.
    
    Args:
        model: Trained sklearn estimator.
        X: The full feature matrix (can be Memmap).
        indices: Array of indices to predict on.
        batch_size: Size of chunks to read from disk.
        
    Returns:
        np.ndarray: Predictions for the indices.
    """
    n_samples = len(indices)
    preds = np.zeros(n_samples, dtype=np.float32)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        
        # X[batch_idx] reads only 'batch_size' rows from disk to RAM
        # This keeps memory footprint tiny (e.g. <100MB instead of 10GB)
        batch_preds = model.predict(X[batch_idx])
        
        preds[start:end] = batch_preds
        
    return preds

def _evaluate_config(
    alpha: float, 
    l1: float, 
    X, y, 
    groups, 
    tissues, 
    folds, 
    trans_func, 
    seed, 
    search_max_iter,
    sampling_params: dict = None
) -> dict:
    """
    Internal helper: Evaluates a single alpha/l1 configuration.
    """
    oof_preds = np.zeros(len(y), dtype=np.float32)
    fold_corrs = []
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=seed, max_iter=search_max_iter)
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # --- Intelligent Down-sampling (Training Set Only) ---
        if sampling_params:
            train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
            train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
            
            df_train_meta = pd.DataFrame({
                'Tissue': train_tissues,
                'project_id': train_groups,
                'original_idx': train_idx 
            })
            
            df_bal, _ = adaptive_sampler(
                df_train_meta, 
                min_coh=sampling_params.get('min_cohorts', 1),
                min_c=sampling_params.get('min_cap', 30),
                max_c=sampling_params.get('max_cap', 500),
                mult=sampling_params.get('median_mult', 1.0),
                seed=seed + fold_idx 
            )
            final_train_idx = df_bal['original_idx'].values
        else:
            final_train_idx = train_idx

        # Train: X[final_train_idx] is small (downsampled), so memory usage is low.
        # Even if X is memmap, we load this subset into RAM for training efficiency.
        model.fit(X[final_train_idx], y[final_train_idx])
        
        # Predict: Val set is LARGE. Use batching to keep memory low.
        # Using model.predict(X[val_idx]) directly would load HUGE data into RAM and crash.
        pred = _batched_predict(model, X, val_idx, batch_size=2048)
        
        # Inverse transform for evaluation (if AgeTransformer is used)
        pred_eval = trans_func.inverse_transform(pred) if trans_func else pred
        y_val_eval = trans_func.inverse_transform(y[val_idx]) if trans_func else y[val_idx]
        
        oof_preds[val_idx] = pred_eval
        
        # Fold Correlation
        if np.std(pred_eval) > 1e-9 and np.std(y_val_eval) > 1e-9:
            fold_corrs.append(np.corrcoef(y_val_eval, pred_eval)[0, 1])
        else:
            fold_corrs.append(0.0)
            
    # --- Compute Comprehensive Metrics ---
    macro_r = np.mean(fold_corrs)
    
    y_eval_all = trans_func.inverse_transform(y) if trans_func else y
    
    if np.std(oof_preds) > 1e-9:
        micro_r = np.corrcoef(y_eval_all, oof_preds)[0, 1]
    else:
        micro_r = 0.0
        
    mae = mean_absolute_error(y_eval_all, oof_preds)
    med_ae = median_absolute_error(y_eval_all, oof_preds)
    score = micro_r + macro_r
    
    # Real-time Logging
    print(f"  [Evaluate] Alpha={alpha:.5f} L1={l1:.2f} | Score={score:.4f} (MicroR={micro_r:.3f}, MacroR={macro_r:.3f}) MAE={mae:.3f} MedAE={med_ae:.3f}")

    return {
        "alpha": alpha,
        "l1_ratio": l1,
        "score": score,
        "micro_r": micro_r,
        "macro_r": macro_r,
        "mae": mae,
        "median_ae": med_ae
    }

def tune_elasticnet_macro_micro(
    X, # Accepts Memmap or Array
    y: np.ndarray, 
    groups: Any, 
    tissues: Any, 
    trans_func: Optional[Any] = None, 
    alphas: Optional[List[float]] = None,
    alpha_start: float = -4.0,
    alpha_end: float = -0.5,
    n_alphas: int = 30,
    l1_ratios: Optional[List[float]] = None,
    l1_ratio: float = 0.5,
    n_jobs: int = -1,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
    min_cohorts: int = 1,
    min_cap: int = 30,
    max_cap: int = 500,
    median_mult: float = 1.0
) -> Tuple[ElasticNet, pd.DataFrame]:
    """
    Grid search for best ElasticNet using Macro+Micro correlation score.
    Returns: (best_model, results_dataframe)
    """
    if alphas is None:
        alphas = np.logspace(alpha_start, alpha_end, n_alphas).tolist()
    if l1_ratios is None:
        l1_ratios = [l1_ratio]
    
    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
    
    if not folds:
        log("Warning: Split failed in optimization. Returning default model.")
        return ElasticNet(alpha=alphas[0] if alphas else 0.01, l1_ratio=l1_ratios[0], random_state=seed), pd.DataFrame()

    sampling_params = {
        'min_cohorts': min_cohorts,
        'min_cap': min_cap,
        'max_cap': max_cap,
        'median_mult': median_mult
    }
    
    if max_cap < 100000:
        log(f"Optimization Strategy: Intelligent Down-sampling Active {sampling_params}")
    else:
        log("Optimization Strategy: Full Sampling")

    param_grid = [(a, l) for a in alphas for l in l1_ratios]
    log(f"Starting Grid Search: {len(param_grid)} configurations using {n_jobs} jobs...")
    
    search_max_iter = max(1000, max_iter // 2)

    # Run Parallel Search
    # Joblib efficiently handles Memmap objects passed as 'X' here.
    # Note: X is passed to subprocesses. Since it's memmap, no copying happens.
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_evaluate_config)(
            a, l, X, y, groups, tissues, folds, trans_func, seed, search_max_iter, sampling_params
        )
        for a, l in param_grid
    )
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Sort by Score (descending)
    df_results = df_results.sort_values(by="score", ascending=False).reset_index(drop=True)
    
    # Select Best
    best_row = df_results.iloc[0]
    best_alpha = best_row['alpha']
    best_l1 = best_row['l1_ratio']
    best_score = best_row['score']
    
    log(f"  > Best Result: Alpha={best_alpha:.5f}, L1={best_l1:.2f}, Score={best_score:.4f}, MAE={best_row['mae']:.4f}, MedAE={best_row['median_ae']:.4f}")
    
    best_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, random_state=seed, max_iter=max_iter)
    
    return best_model, df_results