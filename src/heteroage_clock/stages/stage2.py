"""
heteroage_clock.stages.stage2

Stage 2: Hallmark Experts (Distributed Optuna Version)
Function:
- Trains separate ElasticNet models for each Hallmark using orthogonalized features.
- Supports distributed hyperparameter optimization via Optuna + SQLite/Journal.
- Generates Expert OOF predictions for Stage 3 fusion.
- [Update] Fix: Correctly loads JSON dictionary from Stage 1.
- [Update] Output now accumulates Stage 1 metadata and adds 'pred_resid_' prefix.
"""

import os
import gc
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.base import clone
from typing import Optional, Dict, Any

from heteroage_clock.core.optimization import optimize_elasticnet_optuna
from heteroage_clock.core.age_transform import AgeTransformer
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.sampling import adaptive_sampler
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage2 import Stage2Artifact

def train_stage2(
    output_dir: str,
    stage1_dir: str,
    mmap_path: str,
    # --- Optuna & Search Configuration ---
    n_trials: int = 50,
    n_jobs: int = 1,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
    min_cohorts: int = 2,
    # Search Ranges
    min_cap_low: int = 10, min_cap_high: int = 60,
    max_cap_low: int = 200, max_cap_high: int = 1000,
    median_mult_low: float = 0.5, median_mult_high: float = 2.5,
    alpha_low: float = 1e-4, alpha_high: float = 1.0,
    l1_low: float = 0.0, l1_high: float = 1.0,
    # Distributed Config
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Train Hallmark Expert models using distributed Optuna optimization.
    """
    artifact_handler = Stage2Artifact(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Load Resources from Stage 1 ---
    log("Loading Stage 1 artifacts...")
    
    # Load Feature List
    stage1_features_path = os.path.join(stage1_dir, "stage1_features.pkl")
    if not os.path.exists(stage1_features_path):
        stage1_features_path = os.path.join(stage1_dir, "stage1_features_list.pkl")
    stage1_features = joblib.load(stage1_features_path)
    
    # [FIX] Load Orthogonalized Dictionary (Handle JSON format)
    # Check for JSON first (Current Standard)
    json_path = os.path.join(stage1_dir, "Stage1_Orthogonalized_Hallmark_Dict.json")
    pkl_path = os.path.join(stage1_dir, "stage1_orthogonalized_dict.pkl")
    
    if os.path.exists(json_path):
        log(f"Loading dictionary from JSON: {json_path}")
        with open(json_path, 'r') as f:
            ortho_dict = json.load(f)
    elif os.path.exists(pkl_path):
        log(f"Loading dictionary from Pickle: {pkl_path}")
        ortho_dict = joblib.load(pkl_path)
    else:
        raise FileNotFoundError(
            f"Could not find Hallmark Dict in {stage1_dir}. \n"
            f"Expected {json_path} OR {pkl_path}"
        )
    
    # Load Stage 1 OOF to get Targets and Metadata alignment
    oof_path = os.path.join(stage1_dir, "Stage1_Global_Anchor_OOF.csv")
    stage1_oof = pd.read_csv(oof_path)
    
    # Ensure strict alignment
    target_samples = stage1_oof['sample_id'].astype(str).values
    y = stage1_oof['age'].values.astype(np.float32)
    groups = stage1_oof['project_id']
    tissues = stage1_oof['Tissue']
    
    # --- 2. Setup Memmap Access ---
    n_samples = len(target_samples)
    n_features = len(stage1_features)
    
    if not os.path.exists(mmap_path):
        raise FileNotFoundError(f"Memmap not found at {mmap_path}. Stage 1 must finish first.")
        
    log(f"Mapping data matrix: {n_samples} samples x {n_features} features")
    X_full = np.memmap(mmap_path, dtype='float32', mode='r', shape=(n_samples, n_features))
    
    feat_to_idx = {f: i for i, f in enumerate(stage1_features)}
    
    # --- 3. Iterate Hallmarks ---
    hallmark_models = {}
    oof_preds_dict = {}
    
    trans = AgeTransformer(adult_age=20)
    y_trans = trans.transform(y)
    
    hallmarks = sorted(ortho_dict.keys())
    log(f"Starting training for {len(hallmarks)} Hallmarks...")
    
    search_config = {
        'min_cap_low': min_cap_low, 'min_cap_high': min_cap_high,
        'max_cap_low': max_cap_low, 'max_cap_high': max_cap_high,
        'median_mult_low': median_mult_low, 'median_mult_high': median_mult_high,
        'alpha_low': alpha_low, 'alpha_high': alpha_high,
        'l1_low': l1_low, 'l1_high': l1_high
    }

    for h_name in hallmarks:
        feats = ortho_dict[h_name]
        
        valid_feats = [f for f in feats if f in feat_to_idx]
        if not valid_feats:
            log(f"Skipping {h_name} (No matching features found)")
            continue
            
        indices = [feat_to_idx[f] for f in valid_feats]
        log(f"=== Training Expert: {h_name} ({len(indices)} features) ===")
        
        # Slicing Memmap
        X_sub = X_full[:, indices]
        
        # Unique Study Name for Distributed DB
        h_study_name = f"{study_name}_{h_name}" if study_name else None
        
        # A. Optimize with Optuna
        best_model, _ = optimize_elasticnet_optuna(
            X=X_sub,
            y=y_trans,
            groups=groups,
            tissues=tissues,
            output_dir=os.path.join(output_dir, "optuna_logs", h_name),
            trans_func=trans,
            n_trials=n_trials,
            n_jobs=n_jobs,
            n_splits=n_splits,
            seed=seed,
            max_iter=max_iter,
            min_cohorts=min_cohorts,
            search_config=search_config,
            storage=storage,
            study_name=h_study_name
        )
        
        # B. Generate OOF Predictions
        log(f"   > Generating OOF for {h_name}...")
        folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
        oof_preds = np.zeros(len(y), dtype=np.float32)
        
        best_sampling = getattr(best_model, 'best_sampling_params_', {})

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Sampling Setup
            train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
            train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
            df_train_meta = pd.DataFrame({'Tissue': train_tissues, 'project_id': train_groups, 'original_idx': train_idx})
            
            df_bal, _ = adaptive_sampler(
                df_train_meta, 
                min_coh=min_cohorts, 
                min_c=best_sampling.get('min_cap', 30), 
                max_c=best_sampling.get('max_cap', 500), 
                mult=best_sampling.get('median_mult', 1.0), 
                seed=seed + fold_idx
            )
            final_train_idx = df_bal['original_idx'].values
            
            # Train & Predict Fold
            m_fold = clone(best_model)
            m_fold.fit(X_sub[final_train_idx], y_trans[final_train_idx])
            val_p_trans = m_fold.predict(X_sub[val_idx])
            oof_preds[val_idx] = trans.inverse_transform(val_p_trans)
            
        # Use prefix for easier Stage 3 identification
        oof_preds_dict[f"pred_resid_{h_name}"] = oof_preds
        
        # C. Retrain Final Model on Full Data
        log(f"   > Retraining final model for {h_name}...")
        df_full_meta = pd.DataFrame({'Tissue': tissues, 'project_id': groups, 'original_idx': np.arange(len(y))})
        df_full_bal, _ = adaptive_sampler(
            df_full_meta, 
            min_coh=min_cohorts, 
            min_c=best_sampling.get('min_cap', 30), 
            max_c=best_sampling.get('max_cap', 500), 
            mult=best_sampling.get('median_mult', 1.0), 
            seed=seed
        )
        final_idx = df_full_bal['original_idx'].values
        
        final_model = clone(best_model)
        final_model.fit(X_sub[final_idx], y_trans[final_idx])
        final_model.feature_names_ = valid_feats
        hallmark_models[h_name] = final_model
        
        del X_sub
        import gc
        gc.collect()

    # --- 4. Save Artifacts (Cumulative) ---
    log("Saving Stage 2 Artifacts...")
    
    # Save Models
    artifact_handler.save_expert_models(hallmark_models)
    
    # Save Cumulative OOF (S1 + S2)
    final_oof_df = stage1_oof.copy()
    
    for col_name, preds in oof_preds_dict.items():
        final_oof_df[col_name] = preds
    
    artifact_handler.save_oof_predictions(final_oof_df)
    
    del X_full
    gc.collect()
    
    log(f"Stage 2 Complete. Models saved to {output_dir}")


def predict_stage2(
    artifact_dir: str, 
    stage1_dir: str, 
    mmap_path: str, 
    meta_path: str, 
    output_path: str
) -> None:
    """
    Inference for Stage 2. Loads Hallmark models and predicts on new data.
    """
    log(">>> [Stage 2] Running Hallmark Experts Prediction...")
    
    # 1. Load Artifacts
    artifact_handler = Stage2Artifact(artifact_dir)
    hallmark_models = artifact_handler.load_expert_models()
    
    stage1_features_path = os.path.join(stage1_dir, "stage1_features.pkl")
    if not os.path.exists(stage1_features_path):
         stage1_features_path = os.path.join(stage1_dir, "stage1_features_list.pkl")
    stage1_features = joblib.load(stage1_features_path)
    
    # 2. Setup Data Access
    meta_df = pd.read_pickle(meta_path)
    n_samples = len(meta_df)
    n_features = len(stage1_features)
    
    feat_to_idx = {f: i for i, f in enumerate(stage1_features)}
    
    log(f"Mapping data matrix for inference: {n_samples}x{n_features}")
    X_full = np.memmap(mmap_path, dtype='float32', mode='r', shape=(n_samples, n_features))
    
    trans = AgeTransformer(adult_age=20)
    
    # 3. Predict per Hallmark
    results = {}
    results["sample_id"] = meta_df["sample_id"].values
    
    for h_name, model in hallmark_models.items():
        log(f"   > Predicting {h_name}...")
        
        if hasattr(model, "feature_names_"):
            required_feats = model.feature_names_
        else:
            log(f"Warning: Model for {h_name} lacks feature names. Skipping.")
            continue
            
        indices = [feat_to_idx[f] for f in required_feats if f in feat_to_idx]
        
        # Slice Data
        X_sub = X_full[:, indices]
        
        # Batched Prediction
        batch_size = 4096
        preds_trans = np.zeros(n_samples, dtype=np.float32)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_sub[start:end]
            preds_trans[start:end] = model.predict(X_batch)
            
        # Inverse Transform
        results[f"pred_resid_{h_name}"] = trans.inverse_transform(preds_trans)
        
        del X_sub
        gc.collect()
        
    # 4. Save Results
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    res_df.to_csv(output_path, index=False)
    
    del X_full
    gc.collect()
    log(f"Stage 2 Prediction saved to {output_path}")