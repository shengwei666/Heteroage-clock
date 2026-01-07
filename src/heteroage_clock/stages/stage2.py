"""
heteroage_clock.stages.stage2

Stage 2: Hallmark Experts.
Trains specific ElasticNet models for each aging hallmark using the residuals from Stage 1.
Updates: 
- [Critical] Implemented SEQUENTIAL DATA LOADING to minimize peak memory usage.
- Retains Disk-based Memory Mapping (Memmap) and Float32 optimizations.
"""

import os
import glob
import json
import gc
import shutil
import joblib
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
    **kwargs 
) -> None:
    """
    Train Stage 2 Expert Models.
    Optimized for LOW PEAK MEMORY via Sequential Loading & Memmap.
    """
    artifact_handler = Stage2Artifact(output_dir)
    
    # --- 1. Load Metadata & Target (Stage 1 OOF) ---
    log(f"Loading Stage 1 OOF from {stage1_oof_path}...")
    stage1_oof = pd.read_csv(stage1_oof_path)
    if "residual" not in stage1_oof.columns: 
        raise ValueError("Missing 'residual' in Stage 1 OOF. Cannot train Stage 2.")

    log(f"Loading Orthogonalized Dictionary from {stage1_dict_path}...")
    if not os.path.exists(stage1_dict_path): 
        raise FileNotFoundError(stage1_dict_path)
    
    with open(stage1_dict_path, 'r') as f:
        hallmark_dict = json.load(f)

    log(f"Loading Context (PCs) from {pc_path}...")
    pc_data = pd.read_csv(pc_path)

    # --- 2. Sequential Feature Assembly (Low Peak Memory) ---
    log("Starting Sequential Feature Assembly for Stage 2...")
    
    # Use stage1_oof as the base metadata dataframe
    # This ensures we only load features for samples that have Stage 1 predictions
    train_df = stage1_oof.copy()
    
    # Ensure sample_id is string for merging
    train_df['sample_id'] = train_df['sample_id'].astype(str)
    pc_data['sample_id'] = pc_data['sample_id'].astype(str)

    # Step A: Merge PCs
    log("  > Merging PCs...")
    train_df = pd.merge(train_df, pc_data, on='sample_id', how='inner', suffixes=('', '_dup'))
    # Clean up duplicate columns if any
    dup_cols = [c for c in train_df.columns if c.endswith('_dup')]
    if dup_cols: train_df.drop(columns=dup_cols, inplace=True)
    
    # Cleanup PCs
    del pc_data
    gc.collect()

    # Step B: Load Beta -> Merge -> Delete
    log(f"  > Loading Beta Matrix: {beta_path}")
    cpg_beta = pd.read_pickle(beta_path)
    cpg_beta['sample_id'] = cpg_beta['sample_id'].astype(str)
    
    log("  > Merging Beta...")
    # Using assemble_features logic but iteratively
    train_df = assemble_features(
        cpg_beta=cpg_beta, 
        chalm_data=None, 
        camda_data=None, 
        pc_data=None, 
        metadata=train_df
    )
    
    del cpg_beta
    gc.collect()
    log(f"  > Beta merged. Shape: {train_df.shape}")

    # Step C: Load Chalm -> Merge -> Delete
    log(f"  > Loading Chalm Matrix: {chalm_path}")
    chalm_data = pd.read_pickle(chalm_path)
    chalm_data['sample_id'] = chalm_data['sample_id'].astype(str)
    
    log("  > Merging Chalm...")
    train_df = assemble_features(
        cpg_beta=None, 
        chalm_data=chalm_data, 
        camda_data=None, 
        pc_data=None, 
        metadata=train_df
    )
    
    del chalm_data
    gc.collect()
    log(f"  > Chalm merged. Shape: {train_df.shape}")

    # Step D: Load Camda -> Merge -> Delete
    log(f"  > Loading Camda Matrix: {camda_path}")
    camda_data = pd.read_pickle(camda_path)
    camda_data['sample_id'] = camda_data['sample_id'].astype(str)
    
    log("  > Merging Camda...")
    train_df = assemble_features(
        cpg_beta=None, 
        chalm_data=None, 
        camda_data=camda_data, 
        pc_data=None, 
        metadata=train_df
    )
    
    del camda_data
    gc.collect()
    log(f"  > Camda merged. Final RAM Shape: {train_df.shape}")
    
    # Handle column renaming fallback
    group_col = "project_id" if "project_id" in train_df.columns else "project_id_oof"
    tissue_col = "Tissue" if "Tissue" in train_df.columns else "Tissue_oof"
    
    # --- [MEMORY OPTIMIZATION START: Memmap] ---
    log(f"Preparing Memory Mapping for Stage 2 (Supporting {n_jobs} jobs)...")
    
    temp_memmap_dir = os.path.join(output_dir, "stage2_memmap_cache")
    if os.path.exists(temp_memmap_dir):
        shutil.rmtree(temp_memmap_dir)
    os.makedirs(temp_memmap_dir)

    try:
        # A. Identify Metadata and Features
        exclude_cols = {
            "sample_id", "age", "project_id", "Tissue", "Sex", "Is_Healthy", "Sex_encoded", 
            "residual", "pred_age", "project_id_oof", "Tissue_oof", "residual_oof",
            "age_oof", "Tissue_x", "Tissue_y", "project_id_x", "project_id_y"
        }
        feature_candidates = [c for c in train_df.columns if c not in exclude_cols]
        
        # B. Extract Metadata (Target & Groups) into memory
        y_global = train_df["residual"].values.astype(np.float32)
        sample_ids = train_df["sample_id"].values
        groups = train_df[group_col].copy()
        tissues = train_df[tissue_col].copy()
        
        # C. Create Feature Map
        feat_to_idx = {name: i for i, name in enumerate(feature_candidates)}
        
        # D. Dump Full Feature Matrix to Disk
        X_full_ram = train_df[feature_candidates].values.astype(np.float32)
        X_memmap_path = os.path.join(temp_memmap_dir, "X_stage2_full.joblib")
        
        log(f"Dumping Stage 2 feature matrix to {X_memmap_path}...")
        joblib.dump(X_full_ram, X_memmap_path)
        
        # E. Aggressive Cleanup
        del X_full_ram
        del train_df
        gc.collect()
        
        # F. Load as Memmap
        X_full_mmap = joblib.load(X_memmap_path, mmap_mode='r')
        log(f"Stage 2 Data mapped. Shape: {X_full_mmap.shape}. RAM cleaned.")

        # --- 3. Split Folds ---
        folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
        stage2_oof_df = pd.DataFrame({"sample_id": sample_ids})
        
        # --- 4. Train Hallmark Experts Loop ---
        for hallmark, feat_list in hallmark_dict.items():
            hallmark_clean = hallmark.replace("/", "_").replace(" ", "_")
            log(f"Processing Hallmark Expert: {hallmark_clean}...")
            
            # Find indices for this hallmark's features
            indices = [feat_to_idx[f] for f in feat_list if f in feat_to_idx]
            
            if not indices:
                log(f"  > Warning: No valid features found for {hallmark_clean}. Skipping.")
                continue
                
            # Extract Feature Subset (Reads from Disk -> RAM)
            X_hallmark = X_full_mmap[:, indices]
            
            # A. Optimization
            best_model, _ = tune_elasticnet_macro_micro(
                X=X_hallmark, 
                y=y_global, 
                groups=groups, 
                tissues=tissues, 
                trans_func=None,
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
                min_cohorts=min_cohorts,
                min_cap=min_cap,
                max_cap=max_cap,
                median_mult=median_mult
            )
            
            # B. OOF Generation
            hallmark_oof = np.zeros(len(y_global), dtype=np.float32)
            
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                # Reconstruct Metadata for Sampler
                train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
                train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
                
                df_train_meta = pd.DataFrame({
                    'Tissue': train_tissues,
                    'project_id': train_groups,
                    'idx': train_idx 
                })
                
                df_bal, _ = adaptive_sampler(df_train_meta, min_coh=min_cohorts, min_c=min_cap, max_c=max_cap, mult=median_mult, seed=seed + fold_idx)
                final_train_idx = df_bal['idx'].values
                
                m = clone(best_model)
                m.fit(X_hallmark[final_train_idx], y_global[final_train_idx])
                hallmark_oof[val_idx] = m.predict(X_hallmark[val_idx])
                
            stage2_oof_df[f"pred_residual_{hallmark_clean}"] = hallmark_oof
            
            metrics = compute_regression_metrics(y_global, hallmark_oof)
            log(f"  > {hallmark_clean} Expert Metrics (R2 on Residuals): {metrics.get('R2', 0.0):.4f}")
            
            # C. Final Model Training
            df_full_meta = pd.DataFrame({'Tissue': tissues, 'project_id': groups, 'idx': np.arange(len(y_global))})
            df_full_bal, _ = adaptive_sampler(df_full_meta, min_cohorts, min_cap, max_cap, median_mult, seed)
            final_idx = df_full_bal['idx'].values
            
            final_model = clone(best_model)
            final_model.fit(X_hallmark[final_idx], y_global[final_idx])
            
            used_features = [feature_candidates[i] for i in indices]
            artifact_handler.save_expert_model(hallmark_clean, final_model)
            artifact_handler.save(f"stage2_{hallmark_clean}_features", used_features)

            del X_hallmark, m, final_model
            gc.collect()

        artifact_handler.save_oof_corrections(stage2_oof_df)
        log(f"Stage 2 Completed. Outputs in {output_dir}")

    finally:
        if os.path.exists(temp_memmap_dir):
            try:
                if 'X_full_mmap' in locals():
                    del X_full_mmap
                    gc.collect()
                shutil.rmtree(temp_memmap_dir)
                log(f"Cleaned up Stage 2 temp memmap: {temp_memmap_dir}")
            except Exception as e:
                log(f"Warning: Failed to clean temp dir {temp_memmap_dir}: {e}")

# predict_stage2 保持不变
def predict_stage2(artifact_dir: str, input_path: str, output_path: str) -> None:
    artifact_handler = Stage2Artifact(artifact_dir)
    log(f"Loading input data for Stage 2 from {input_path}...")
    
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
        
    log(f"Found {len(model_files)} expert models.")
        
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
        
        X = data[feature_cols].values.astype(np.float32)
        preds = model.predict(X)
        output_df[f"pred_residual_{hallmark_name}"] = preds
        del X, model
        gc.collect()
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log(f"Stage 2 predictions saved to {output_path}")