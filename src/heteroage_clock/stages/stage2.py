"""
heteroage_clock.stages.stage2

Stage 2: Hallmark Experts.
Trains specific ElasticNet models for each aging hallmark using the residuals from Stage 1.
Updates: 
- Implements 'Intelligent Down-sampling' (Asymmetric OOF) to match Stage 1 fairness.
- Fixes dictionary loading bug (uses json.load).
- Accepts **kwargs to handle pipeline arguments.
- Adapts to new optimization return signature.
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
from heteroage_clock.core.sampling import adaptive_sampler

def train_stage2(
    output_dir: str, 
    stage1_oof_path: str, 
    stage1_dict_path: str, 
    pc_path: str,
    beta_path: str,
    chalm_path: str,
    camda_path: str,
    # Hyperparameters
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
    # Intelligent Sampling Params
    min_cohorts: int = 1,
    min_cap: int = 30,
    max_cap: int = 500,
    median_mult: float = 1.0,
    **kwargs # Absorbs extra args passed by pipeline
) -> None:
    """
    Train Stage 2 Expert Models.
    """
    artifact_handler = Stage2Artifact(output_dir)
    
    # --- 1. Load Data ---
    log(f"Loading Stage 1 OOF from {stage1_oof_path}...")
    stage1_oof = pd.read_csv(stage1_oof_path)
    if "residual" not in stage1_oof.columns: 
        raise ValueError("Missing 'residual' in Stage 1 OOF. Cannot train Stage 2.")

    log(f"Loading Orthogonalized Dictionary from {stage1_dict_path}...")
    if not os.path.exists(stage1_dict_path): 
        raise FileNotFoundError(stage1_dict_path)
    
    # [Fix]: Use json.load instead of pd.read_json
    with open(stage1_dict_path, 'r') as f:
        hallmark_dict = json.load(f)

    log("Loading raw omics features...")
    cpg_beta = pd.read_pickle(beta_path)
    chalm_data = pd.read_pickle(chalm_path)
    camda_data = pd.read_pickle(camda_path)
    pc_data = pd.read_csv(pc_path)

    # --- 2. Assemble Features ---
    # Merge omics data
    assembled_data = assemble_features(cpg_beta, chalm_data, camda_data, pc_data, cpg_beta)
    
    # Merge with Stage 1 Residuals (Target for Stage 2)
    # Inner merge ensures we only use samples that have both Omics and Stage 1 OOF
    train_df = pd.merge(assembled_data, stage1_oof, on="sample_id", how="inner", suffixes=("", "_oof"))
    
    # Handle potential column renaming during merge
    group_col = "project_id" if "project_id" in train_df.columns else "project_id_oof"
    tissue_col = "Tissue" if "Tissue" in train_df.columns else "Tissue_oof"
    
    if group_col not in train_df.columns:
        # Fallback if project_id was lost (unlikely with fixed assemble.py)
        log("Warning: project_id column ambiguous. Using existing metadata.")
        groups = assembled_data.loc[train_df.index, "project_id"]
    else:
        groups = train_df[group_col]

    if tissue_col not in train_df.columns:
        tissues = assembled_data.loc[train_df.index, "Tissue"]
    else:
        tissues = train_df[tissue_col]

    # The target for Stage 2 is the RESIDUAL from Stage 1
    y_global = train_df["residual"].values
    
    log(f"Stage 2 Training Data Assembled: {len(train_df)} samples.")

    # --- 3. Split Folds ---
    # We use the same split logic to maintain consistency
    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
    
    # Prepare container for Stage 2 OOF corrections
    stage2_oof_df = train_df[["sample_id"]].copy()
    
    # --- 4. Train Hallmark Experts Loop ---
    for hallmark, feat_list in hallmark_dict.items():
        # Sanitize name
        hallmark_clean = hallmark.replace("/", "_").replace(" ", "_")
        log(f"Processing Hallmark Expert: {hallmark_clean}...")
        
        # Select features relevant to this hallmark
        relevant_cols = [c for c in feat_list if c in train_df.columns]
        
        if not relevant_cols:
            log(f"  > Warning: No valid features found for {hallmark_clean}. Skipping.")
            continue
            
        X = train_df[relevant_cols].values
        
        # A. Optimization (with Intelligent Sampling)
        # We find the best hyperparameters for this specific hallmark expert
        # [UPDATED]: Unpack model, ignore sweep report (_)
        best_model, _ = tune_elasticnet_macro_micro(
            X=X, 
            y=y_global, 
            groups=groups, 
            tissues=tissues, 
            trans_func=None, # Predicting residuals (linear), no age transform needed
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
        
        # B. OOF Generation (Asymmetric Strategy)
        # Train on Balanced Subset, Predict on Full Fold
        hallmark_oof = np.zeros(len(y_global))
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # 1. Down-sample Training Set
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
            
            # 2. Train Expert on Balanced Data
            m = clone(best_model)
            m.fit(X[final_train_idx], y_global[final_train_idx])
            
            # 3. Predict on Full Validation Data
            hallmark_oof[val_idx] = m.predict(X[val_idx])
            
        # Store predictions
        stage2_oof_df[f"pred_residual_{hallmark_clean}"] = hallmark_oof
        
        metrics = compute_regression_metrics(y_global, hallmark_oof)
        log(f"  > {hallmark_clean} Expert Metrics (R2 on Residuals): {metrics.get('R2', 0.0):.4f}")
        
        # C. Final Model Training (Balanced Full Set)
        # Ensure the exported model is also fair
        df_full_meta = pd.DataFrame({
            'Tissue': tissues, 
            'project_id': groups, 
            'idx': np.arange(len(y_global))
        })
        df_full_bal, _ = adaptive_sampler(
            df_full_meta, min_cohorts, min_cap, max_cap, median_mult, seed
        )
        final_idx = df_full_bal['idx'].values
        
        final_model = clone(best_model)
        final_model.fit(X[final_idx], y_global[final_idx])
        
        # Save Artifacts
        artifact_handler.save_expert_model(hallmark_clean, final_model)
        artifact_handler.save(f"stage2_{hallmark_clean}_features", relevant_cols)

    # Save all OOF corrections (Input for Stage 3)
    artifact_handler.save_oof_corrections(stage2_oof_df)
    log(f"Stage 2 Completed. Outputs in {output_dir}")

def predict_stage2(artifact_dir: str, input_path: str, output_path: str) -> None:
    """
    Inference for Stage 2.
    Loads all expert models and predicts residuals.
    """
    artifact_handler = Stage2Artifact(artifact_dir)
    log(f"Loading input data for Stage 2 from {input_path}...")
    
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_pickle(input_path)
        
    output_df = pd.DataFrame()
    if "sample_id" in data.columns:
        output_df["sample_id"] = data["sample_id"]
        
    # Find all trained expert models
    model_files = glob.glob(os.path.join(artifact_dir, "stage2_*_expert_model.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No Stage 2 models found in {artifact_dir}")
        
    log(f"Found {len(model_files)} expert models.")
        
    for m_file in model_files:
        basename = os.path.basename(m_file)
        # Extract hallmark name: stage2_Inflammation_expert_model.joblib -> Inflammation
        hallmark_name = basename.replace("stage2_", "").replace("_expert_model.joblib", "")
        
        model = artifact_handler.load_expert_model(hallmark_name)
        feat_name = f"stage2_{hallmark_name}_features"
        
        try:
            feature_cols = artifact_handler.load(feat_name)
        except FileNotFoundError:
            log(f"Warning: Features for {hallmark_name} not found. Skipping.")
            continue
        
        # Ensure features exist
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
             # Fill missing with 0.0
             for c in missing: data[c] = 0.0
        
        X = data[feature_cols].values
        preds = model.predict(X)
        
        output_df[f"pred_residual_{hallmark_name}"] = preds
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log(f"Stage 2 predictions saved to {output_path}")