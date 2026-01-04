"""
heteroage_clock.stages.stage2

Stage 2: Hallmark Experts
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from sklearn.base import clone
from typing import List, Optional

from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.data.assemble import assemble_features
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.optimization import tune_elasticnet_macro_micro
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage2 import Stage2Artifact
from heteroage_clock.core.sampling import adaptive_sampler # <--- [Added]

def train_stage2(
    output_dir: str, 
    stage1_oof_path: str, 
    stage1_dict_path: str, 
    pc_path: str,
    beta_path: str,
    chalm_path: str,
    camda_path: str,
    # Hyperparameters updated to support lists and parallel
    alpha_start: float = -4.0,
    alpha_end: float = -0.5,
    n_alphas: int = 30,
    l1_ratio: float = 0.5,
    alphas: Optional[List[float]] = None,
    l1_ratios: Optional[List[float]] = None,
    n_jobs: int = -1,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
    # New Intelligent Sampling Params
    min_cohorts: int = 2,
    min_cap: int = 30,
    max_cap: int = 600,
    median_mult: float = 1.0,
    **kwargs
) -> None:
    """
    Train Stage 2 Expert Models.
    Accepts explicit paths and hyperparameters.
    Now supports parallel processing, hyperparameter lists, and intelligent sampling.
    """
    artifact_handler = Stage2Artifact(output_dir)
    
    # --- 1. Load Data ---
    log(f"Loading Stage 1 OOF from {stage1_oof_path}...")
    stage1_oof = pd.read_csv(stage1_oof_path)
    if "residual" not in stage1_oof.columns: raise ValueError("Missing 'residual' in Stage 1 OOF")

    log(f"Loading Dictionary from {stage1_dict_path}...")
    if not os.path.exists(stage1_dict_path): raise FileNotFoundError(stage1_dict_path)
    
    # Fixed: Use json.load
    with open(stage1_dict_path, 'r') as f:
        hallmark_dict = json.load(f)

    log("Loading raw omics features...")
    cpg_beta = pd.read_pickle(beta_path)
    chalm_data = pd.read_pickle(chalm_path)
    camda_data = pd.read_pickle(camda_path)
    pc_data = pd.read_csv(pc_path)

    # --- 2. Assemble ---
    assembled_data = assemble_features(cpg_beta, chalm_data, camda_data, pc_data, cpg_beta)
    
    train_df = pd.merge(assembled_data, stage1_oof, on="sample_id", how="inner", suffixes=("", "_oof"))
    
    # Logic to handle potentially different column names after merge
    group_col = "project_id" if "project_id" in train_df.columns else "project_id_oof"
    tissue_col = "Tissue" if "Tissue" in train_df.columns else "Tissue_oof"
    
    groups = train_df[group_col]
    tissues = train_df[tissue_col]
    y_global = train_df["residual"].values
    
    # --- 3. Split ---
    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
    stage2_oof_df = train_df[["sample_id"]].copy()
    
    # --- 4. Train Hallmark Experts ---
    for hallmark, feat_list in hallmark_dict.items():
        hallmark_clean = hallmark.replace("/", "_").replace(" ", "_")
        log(f"Processing Hallmark: {hallmark_clean}...")
        
        relevant_cols = [c for c in feat_list if c in train_df.columns]
        
        if not relevant_cols:
            log(f"  > Warning: No valid features found for {hallmark_clean}. Skipping.")
            continue
            
        X = train_df[relevant_cols].values
        
        # Optimization (now with sampling params)
        best_model = tune_elasticnet_macro_micro(
            X=X, 
            y=y_global, 
            groups=groups, 
            tissues=tissues, 
            trans_func=None, # Predicting residuals directly, no transform needed
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            n_alphas=n_alphas,
            l1_ratio=l1_ratio,
            alphas=alphas,
            l1_ratios=l1_ratios,
            n_jobs=n_jobs,
            n_splits=n_splits,
            seed=seed,
            max_iter=max_iter,
            # Pass Sampling Params
            min_cohorts=min_cohorts,
            min_cap=min_cap,
            max_cap=max_cap,
            median_mult=median_mult
        )
        
        # OOF Generation with Asymmetric Sampling
        hallmark_oof = np.zeros(len(y_global))
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # --- A. Intelligent Down-sampling on Train Set ---
            train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
            train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
            
            df_train_meta = pd.DataFrame({
                'Tissue': train_tissues,
                'project_id': train_groups,
                'idx': train_idx 
            })
            
            df_bal, _ = adaptive_sampler(
                df_train_meta, 
                min_coh=min_cohorts,
                min_c=min_cap,
                max_c=max_cap,
                mult=median_mult,
                seed=seed + fold_idx
            )
            final_train_idx = df_bal['idx'].values
            
            # --- B. Train & Predict ---
            m = clone(best_model)
            m.fit(X[final_train_idx], y_global[final_train_idx]) # Fit on Balanced
            hallmark_oof[val_idx] = m.predict(X[val_idx]) # Predict on Full
            
        stage2_oof_df[f"pred_residual_{hallmark_clean}"] = hallmark_oof
        
        metrics = compute_regression_metrics(y_global, hallmark_oof)
        log(f"  > Metrics: {metrics}")
        
        # Final Model Training (Balanced Full Set)
        df_full_meta = pd.DataFrame({'Tissue': tissues, 'project_id': groups, 'idx': np.arange(len(y_global))})
        df_full_bal, _ = adaptive_sampler(df_full_meta, min_cohorts, min_cap, max_cap, median_mult, seed)
        final_idx = df_full_bal['idx'].values
        
        final_model = clone(best_model)
        final_model.fit(X[final_idx], y_global[final_idx])
        
        artifact_handler.save_expert_model(hallmark_clean, final_model)
        artifact_handler.save(f"stage2_{hallmark_clean}_features", relevant_cols)

    artifact_handler.save_oof_corrections(stage2_oof_df)
    log(f"Stage 2 Completed.")

# predict_stage2 logic remains same
def predict_stage2(artifact_dir: str, input_path: str, output_path: str) -> None:
    artifact_handler = Stage2Artifact(artifact_dir)
    log(f"Loading input data from {input_path}...")
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_pickle(input_path)
        
    output_df = pd.DataFrame()
    if "sample_id" in data.columns:
        output_df["sample_id"] = data["sample_id"]
        
    model_files = glob.glob(os.path.join(artifact_dir, "stage2_*_expert_model.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No Stage 2 models found in {artifact_dir}")
        
    for m_file in model_files:
        basename = os.path.basename(m_file)
        hallmark_name = basename.replace("stage2_", "").replace("_expert_model.joblib", "")
        
        model = artifact_handler.load_expert_model(hallmark_name)
        feat_name = f"stage2_{hallmark_name}_features"
        try:
            feature_cols = artifact_handler.load(feat_name)
        except FileNotFoundError:
            continue
        
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
             for c in missing: data[c] = 0.0
        
        X = data[feature_cols].values
        preds = model.predict(X)
        
        output_df[f"pred_residual_{hallmark_name}"] = preds
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log(f"Stage 2 predictions saved to {output_path}")