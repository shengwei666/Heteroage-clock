import optuna
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from typing import Optional, Tuple, Dict, Any
# [核心修改 1] 引入 JournalStorage 模块
from optuna.storages import JournalStorage, JournalFileStorage

from heteroage_clock.utils.logging import log
from heteroage_clock.core.sampling import adaptive_sampler

def optimize_elasticnet_optuna(
    X: np.memmap,
    y: np.ndarray,
    groups: pd.Series,
    tissues: pd.Series,
    output_dir: str,
    trans_func: Any,
    n_trials: int = 100,
    n_jobs: int = 1,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 1000,
    min_cohorts: int = 2,
    search_config: Optional[Dict] = None,
    storage: Optional[str] = None,
    study_name: Optional[str] = None
) -> Tuple[Any, pd.DataFrame]:
    """
    Optimizes ElasticNet using Optuna with distributed support (NFS-Safe Journal Mode).
    """
    if search_config is None:
        search_config = {}

    # Define the objective function
    def objective(trial):
        # 1. Hyperparameters
        alpha = trial.suggest_float("alpha", search_config.get('alpha_low', 1e-4), search_config.get('alpha_high', 1.0), log=True)
        l1_ratio = trial.suggest_float("l1_ratio", search_config.get('l1_low', 0.1), search_config.get('l1_high', 1.0))
        
        # 2. Sampling Parameters
        min_cap = trial.suggest_int("min_cap", search_config.get('min_cap_low', 10), search_config.get('min_cap_high', 60))
        max_cap = trial.suggest_int("max_cap", search_config.get('max_cap_low', 200), search_config.get('max_cap_high', 1000))
        median_mult = trial.suggest_float("median_mult", search_config.get('median_mult_low', 0.5), search_config.get('median_mult_high', 2.5))
        
        trial.set_user_attr("min_cap", min_cap)
        trial.set_user_attr("max_cap", max_cap)
        trial.set_user_attr("median_mult", median_mult)

        # 3. Cross-Validation Loop
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        mae_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, tissues, groups)):
            # Adaptive Sampling
            train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
            train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
            
            df_meta_train = pd.DataFrame({'Tissue': train_tissues, 'project_id': train_groups, 'original_idx': train_idx})
            df_bal, _ = adaptive_sampler(
                df_meta_train, 
                min_coh=min_cohorts, 
                min_c=min_cap, 
                max_c=max_cap, 
                mult=median_mult, 
                seed=seed + fold_idx
            )
            final_train_idx = df_bal['original_idx'].values
            
            # Model Training
            model = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=max_iter, selection='random'
            )
            model.fit(X[final_train_idx], y[final_train_idx])
            
            # Predict & Evaluate
            preds_trans = model.predict(X[val_idx])
            preds_age = trans_func.inverse_transform(preds_trans)
            true_age = trans_func.inverse_transform(y[val_idx])
            
            mae = mean_absolute_error(true_age, preds_age)
            mae_scores.append(mae)
            
            trial.report(mae, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(mae_scores)

    # --- [核心修改 2] 智能判断存储后端 ---
    if storage and study_name:
        if storage.startswith("sqlite") or storage.startswith("mysql") or storage.startswith("postgresql"):
            # 传统数据库模式 (SQLAlchemy)
            log(f"Using RDB Storage: {storage}")
            storage_backend = storage
        else:
            # 文件日志模式 (JournalStorage) - 专治 NFS 报错
            log(f"Using NFS-Safe Journal Storage: {storage}")
            storage_backend = JournalStorage(JournalFileStorage(storage))

        study = optuna.create_study(
            direction="minimize",
            storage=storage_backend, # 传入对象而不是字符串
            study_name=study_name,
            load_if_exists=True
        )
    else:
        log("Creating in-memory study (Single process mode)")
        study = optuna.create_study(direction="minimize")

    # Run Optimization
    log(f"Starting optimization loop for {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    log("Optimization finished.")
    log(f"Best Trial: {study.best_trial.params}")
    log(f"Best MAE: {study.best_value}")

    # Retrain Best Model
    best_params = study.best_trial.params
    best_model = ElasticNet(
        alpha=best_params['alpha'],
        l1_ratio=best_params['l1_ratio'],
        random_state=seed,
        max_iter=max_iter
    )
    best_model.best_sampling_params_ = {
        'min_cap': best_params['min_cap'],
        'max_cap': best_params['max_cap'],
        'median_mult': best_params['median_mult']
    }

    trials_df = study.trials_dataframe()
    return best_model, trials_df