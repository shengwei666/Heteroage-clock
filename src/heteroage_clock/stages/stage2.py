"""
heteroage_clock.stages.stage2

Stage 2: Hallmark Experts (Parallel Architecture).
REVERTED TO RESIDUAL PREDICTION STRATEGY.

Strategy:
- Experts predict the RESIDUAL (Error) of Stage 1, not the absolute age.
- Target = True Age - Stage 1 Prediction.
- Goal: Experts specifically focus on what Stage 1 missed (the biological deviation).

[Changes from Route A]:
1. Target: Residuals (y - y_hat).
2. Output Columns: 'pred_residual_{hallmark}'.
3. Streaming: Keeps the memory-efficient disk-based assembly.
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
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.optimization import tune_elasticnet_macro_micro
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage2 import Stage2Artifact
from heteroage_clock.core.sampling import adaptive_sampler

# ==============================================================================
# Helper: Low-Memory Modality Processor (Unchanged)
# ==============================================================================
def process_and_dump_modality(name, path, target_samples, output_dir, suffix):
    """
    Loads one modality, aligns to target samples, imputes, and dumps to a temp joblib file.
    Returns: (temp_file_path, feature_names_list)
    """
    log(f"  > [Stream] Processing {name} from {path}...")
    
    # 1. Load Data
    df = pd.read_pickle(path)
    
    # 2. Standardize Metadata
    if 'sample_id' not in df.columns:
        if df.index.name == 'sample_id': df = df.reset_index()
    df['sample_id'] = df['sample_id'].astype(str)
    
    # 3. Align Samples
    df = df.set_index('sample_id')
    common_count = len(df.index.intersection(target_samples))
    log(f"     - Found {common_count}/{len(target_samples)} target samples.")
    df = df.reindex(target_samples)
    
    # 4. Identify Features (Exclude Metadata)
    meta_cols = {'project_id', 'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy', 'Sex_encoded', 'sample_id', 'sample_key'}
    
    # Select ONLY numeric columns first
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Filter out any numeric metadata that might have slipped through
    features = [c for c in df_numeric.columns if c not in meta_cols]
    
    if len(features) < len(df.columns) - len(meta_cols):
        log(f"     - Note: Filtered out non-numeric columns. Keeping {len(features)} numeric features.")
    
    # 5. Rename Features with Suffix
    feat_map = {}
    if suffix:
        for f in features:
            if not f.endswith(suffix):
                feat_map[f] = f"{f}{suffix}"
    
    if feat_map:
        df.rename(columns=feat_map, inplace=True)
        features = [feat_map.get(f, f) for f in features]
        
    # 6. Extract & Impute (In-Memory for single modality)
    log(f"     - Extracting {len(features)} features...")
    
    # Use df[features] which strictly contains only numeric columns
    X_mod = df[features].values.astype(np.float32)
    
    # Free memory immediately
    del df, df_numeric
    gc.collect()
    
    # Fast Imputation (Median)
    if np.isnan(X_mod).any():
        log(f"     - Imputing missing values (Median)...")
        col_medians = np.nanmedian(X_mod, axis=0)
        inds = np.where(np.isnan(X_mod))
        X_mod[inds] = np.take(col_medians, inds[1])
        X_mod = np.nan_to_num(X_mod, nan=0.0)
    
    # 7. Dump to Disk
    temp_path = os.path.join(output_dir, f"temp_{name}.joblib")
    joblib.dump(X_mod, temp_path)
    
    del X_mod
    gc.collect()
    
    return temp_path, features

# ==============================================================================
# Main Training Function (Reverted to Residuals)
# ==============================================================================
def train_stage2(
    output_dir: str, 
    stage1_oof_path: str, 
    stage1_dict_path: str, 
    pc_path: str,
    beta_path: str,
    chalm_path: str,
    camda_path: str,
    # Hyperparameters (Optimized for Residuals: Smaller Alphas)
    alpha_start: float = -5.0,     # Finer search for smaller residuals
    alpha_end: float = 0.0,        # Upper bound usually lower for residuals
    n_alphas: int = 50,
    l1_ratio: float = 0.5,
    alphas: Optional[List[float]] = None,
    l1_ratios: Optional[List[float]] = None,
    n_jobs: int = -1,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 5000,
    # Intelligent Sampling Params
    min_cohorts: int = 1,
    min_cap: int = 30,
    max_cap: int = 500,
    median_mult: float = 1.0,
    **kwargs 
) -> None:
    """
    Train Stage 2 Expert Models using Disk-Based Streaming Assembly.
    Strategy: Residual Prediction (Target = Age - Stage1_Pred).
    """
    artifact_handler = Stage2Artifact(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    log("=========================================================")
    log(" STAGE 2: Hallmark Experts (Parallel Architecture)")
    log(" Strategy: Residual Prediction (Target = True Age - Stage 1)")
    log("=========================================================")
    
    # --- 1. Load Metadata & Target (Stage 1 OOF) ---
    log(f"Loading Stage 1 OOF from {stage1_oof_path}...")
    stage1_oof = pd.read_csv(stage1_oof_path)
    
    # Standardize columns
    col_map = {}
    if 'Age' in stage1_oof.columns: col_map['Age'] = 'age'
    if 'Stage1_Pred' in stage1_oof.columns: col_map['Stage1_Pred'] = 'pred_age'
    if col_map:
        stage1_oof.rename(columns=col_map, inplace=True)
        
    if 'age' not in stage1_oof.columns or 'pred_age' not in stage1_oof.columns:
        raise ValueError(f"Stage 1 OOF missing 'age' or 'pred_age' columns.")
        
    # [CRITICAL RESTORATION] Target is Residual
    # We want experts to predict the ERROR of Stage 1
    stage1_oof['target_residual'] = stage1_oof['age'] - stage1_oof['pred_age']
    
    stage1_oof['sample_id'] = stage1_oof['sample_id'].astype(str)
    
    # Determine valid target samples (Sorted for alignment)
    target_samples = sorted(stage1_oof['sample_id'].unique())
    log(f"Stage 2 Target Samples: {len(target_samples)}")
    
    # Align OOF to target_samples strictly
    stage1_oof = stage1_oof.set_index('sample_id').reindex(target_samples).reset_index()

    # Define Target Vector (Residuals)
    y_global = stage1_oof['target_residual'].values.astype(np.float32)
    log(f"Target defined: Residuals (Range: {y_global.min():.2f} to {y_global.max():.2f})")

    log(f"Loading Hallmark Dictionary from {stage1_dict_path}...")
    if not os.path.exists(stage1_dict_path): 
        raise FileNotFoundError(stage1_dict_path)
    
    with open(stage1_dict_path, 'r') as f:
        hallmark_dict = json.load(f)

    # --- 2. Stream Process Modalities (Build X) ---
    # (Same memory-efficient loading logic)
    temp_dir = os.path.join(output_dir, "stage2_stream_cache")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    try:
        # A. Process PCs (Context)
        log(f"Processing Context (PCs) from {pc_path}...")
        pc_data = pd.read_csv(pc_path)
        pc_data['sample_id'] = pc_data['sample_id'].astype(str)
        pc_data = pc_data.set_index('sample_id').reindex(target_samples).reset_index()
        pc_cols = [c for c in pc_data.columns if c.startswith('RF_PC')]
        
        # B. Stream Omics Modalities
        tasks = [
            ('beta', beta_path, '_beta'),
            ('chalm', chalm_path, '_chalm'),
            ('camda', camda_path, '_camda')
        ]
        
        temp_files = []
        all_features = []
        
        for name, path, suffix in tasks:
            if path and os.path.exists(path):
                t_path, feats = process_and_dump_modality(name, path, target_samples, temp_dir, suffix)
                temp_files.append(t_path)
                all_features.extend(feats)
            else:
                log(f"Warning: Path for {name} not found or empty: {path}")
            
        all_features.extend(pc_cols)
        
        # C. Assemble Final Memmap
        log("Assembling Stage 2 X Matrix on Disk...")
        n_samples = len(target_samples)
        n_features = len(all_features)
        
        mm_file = os.path.join(temp_dir, "X_stage2.dat")
        X_final = np.memmap(mm_file, dtype='float32', mode='w+', shape=(n_samples, n_features))
        
        current_col = 0
        
        # Write Modalities
        for t_path in temp_files:
            log(f"  > Merging {os.path.basename(t_path)}...")
            X_part = joblib.load(t_path)
            n_cols = X_part.shape[1]
            X_final[:, current_col : current_col + n_cols] = X_part
            current_col += n_cols
            del X_part
            gc.collect()
            
        # Write PCs
        log(f"  > Merging PCs...")
        if len(pc_cols) > 0:
            X_pc = pc_data[pc_cols].values.astype(np.float32)
            X_final[:, current_col:] = X_pc
        
        X_final.flush()
        log(f"Stage 2 Data Assembly Complete. Shape: {X_final.shape}")
        
        # Switch to Read-Only
        X_full_mmap = np.memmap(mm_file, dtype='float32', mode='r', shape=(n_samples, n_features))
        feat_to_idx = {name: i for i, name in enumerate(all_features)}

        # --- 3. Prepare Metadata & Output Container ---
        group_col = "project_id" if "project_id" in stage1_oof.columns else "project_id_oof"
        tissue_col = "Tissue" if "Tissue" in stage1_oof.columns else "Tissue_oof"
        
        groups = stage1_oof[group_col].copy()
        tissues = stage1_oof[tissue_col].copy()

        # Output DataFrame
        stage2_oof_df = stage1_oof[['sample_id', 'age', 'pred_age', tissue_col, group_col, 'target_residual']].copy()
        stage2_oof_df.rename(columns={
            'age': 'Age', 
            'pred_age': 'Stage1_Pred',
            tissue_col: 'Tissue',
            group_col: 'project_id',
            'target_residual': 'True_Residual' # Save the target for checking later
        }, inplace=True)
        
        # --- 4. Train Hallmark Experts Loop ---
        folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
        
        for hallmark, feat_list in hallmark_dict.items():
            hallmark_clean = hallmark.replace("/", "_").replace(" ", "_")
            log(f"\n>> Processing Hallmark Expert: [{hallmark_clean}] (Target: Residual)")
            
            indices = [feat_to_idx[f] for f in feat_list if f in feat_to_idx]
            
            if not indices:
                log(f"   Warning: No valid features found for {hallmark_clean}. Skipping.")
                continue
            
            X_hallmark = X_full_mmap[:, indices]
            log(f"   Input shape: {X_hallmark.shape}")
            
            # A. Hyperparameter Optimization (Target = Residual)
            best_model, _ = tune_elasticnet_macro_micro(
                X=X_hallmark, y=y_global, groups=groups, tissues=tissues, 
                trans_func=None,
                alpha_start=alpha_start, alpha_end=alpha_end, n_alphas=n_alphas,
                l1_ratio=l1_ratio, alphas=alphas, l1_ratios=l1_ratios,
                n_jobs=n_jobs, n_splits=n_splits, seed=seed, max_iter=max_iter,
                min_cohorts=min_cohorts, min_cap=min_cap, max_cap=max_cap, median_mult=median_mult
            )
            
            # B. OOF Predictions
            hallmark_oof = np.zeros(len(y_global), dtype=np.float32)
            
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
                train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
                
                df_train_meta = pd.DataFrame({'Tissue': train_tissues, 'project_id': train_groups, 'idx': train_idx})
                df_bal, _ = adaptive_sampler(df_train_meta, min_coh=min_cohorts, min_c=min_cap, max_c=max_cap, mult=median_mult, seed=seed + fold_idx)
                final_train_idx = df_bal['idx'].values
                
                m = clone(best_model)
                m.fit(X_hallmark[final_train_idx], y_global[final_train_idx])
                hallmark_oof[val_idx] = m.predict(X_hallmark[val_idx])
                
            # [CRITICAL RESTORATION] Output Name: pred_residual_{hallmark}
            col_name = f"pred_residual_{hallmark_clean}"
            stage2_oof_df[col_name] = hallmark_oof
            
            # Metrics (MAE on Residuals - closer to 0 is better fit to the ERROR)
            # R2 here means "How much of the Stage 1 Error can we explain?"
            metrics = compute_regression_metrics(y_global, hallmark_oof)
            log(f"   > {hallmark_clean} Residual R2: {metrics.get('R2', 0.0):.4f} | MAE: {metrics.get('MAE', 0.0):.4f}")
            
            # C. Final Model Training
            df_full_meta = pd.DataFrame({'Tissue': tissues, 'project_id': groups, 'idx': np.arange(len(y_global))})
            df_full_bal, _ = adaptive_sampler(df_full_meta, min_cohorts, min_cap, max_cap, median_mult, seed)
            final_idx = df_full_bal['idx'].values
            
            final_model = clone(best_model)
            final_model.fit(X_hallmark[final_idx], y_global[final_idx])
            
            used_features = [all_features[i] for i in indices]
            artifact_handler.save_expert_model(hallmark_clean, final_model)
            artifact_handler.save(f"stage2_{hallmark_clean}_features", used_features)

            del X_hallmark, m, final_model
            gc.collect()

        # Save Final CSV
        artifact_handler.save_oof_corrections(stage2_oof_df)
        log(f"✅ Stage 2 Completed. Predictions saved to: {os.path.join(output_dir, 'Stage2_Hallmark_OOF.csv')}") # Updated filename

    finally:
        if os.path.exists(temp_dir):
            try:
                if 'X_full_mmap' in locals(): del X_full_mmap
                if 'X_final' in locals(): del X_final
                gc.collect()
                shutil.rmtree(temp_dir)
                log(f"Cleaned up Stage 2 temp memmap: {temp_dir}")
            except Exception as e:
                log(f"Warning: Failed to clean temp dir {temp_dir}: {e}")

def predict_stage2(artifact_dir: str, input_path: str, output_path: str) -> None:
    """
    Inference for Stage 2 (Target = Residuals).
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
        
    model_files = glob.glob(os.path.join(artifact_dir, "stage2_*_expert_model.joblib"))
    
    for m_file in model_files:
        basename = os.path.basename(m_file)
        hallmark_name = basename.replace("stage2_", "").replace("_expert_model.joblib", "")
        
        model = artifact_handler.load_expert_model(hallmark_name)
        feat_name = f"stage2_{hallmark_name}_features"
        try:
            feature_cols = artifact_handler.load(feat_name)
        except FileNotFoundError:
            continue
        
        # Handle missing columns
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
             for c in missing: data[c] = 0.0
        
        X = data[feature_cols].values.astype(np.float32)
        preds = model.predict(X)
        
        # [CRITICAL RESTORATION] Output Name
        output_df[f"pred_residual_{hallmark_name}"] = preds
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log(f"Stage 2 predictions saved to {output_path}")