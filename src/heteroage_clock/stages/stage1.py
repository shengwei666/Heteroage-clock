"""
heteroage_clock.stages.stage1

Stage 1: Global Anchor
Updates:
- [Critical Fix] SKIPPED Orthogonalization to prevent signal loss in Stage 2.
- [Fix] Added robust non-numeric column filtering.
- [Feature] Disk-Based Streaming Assembly (Plan C).
"""

import os
import json
import gc
import shutil
import joblib
import pandas as pd
import numpy as np
from sklearn.base import clone
from collections import defaultdict
from typing import List, Optional

from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.core.age_transform import AgeTransformer
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.optimization import tune_elasticnet_macro_micro
# Note: orthogonalize_by_correlation is imported but temporarily unused in this fix
from heteroage_clock.core.selection import orthogonalize_by_correlation
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage1 import Stage1Artifact
from heteroage_clock.core.sampling import adaptive_sampler

# --- Helper: Low-Memory Modality Processor ---
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
    
    # 3. Align Samples (Inner Join logic relative to target_samples)
    df = df.set_index('sample_id')
    common_count = len(df.index.intersection(target_samples))
    log(f"    - Found {common_count}/{len(target_samples)} target samples.")
    df = df.reindex(target_samples)
    
    # 4. Identify Features (ROBUST FIX)
    # A. Define metadata columns to exclude
    meta_cols = {'project_id', 'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy', 'Sex_encoded', 'sample_id', 'sample_key'}
    
    # B. Select ONLY numeric columns first
    df_numeric = df.select_dtypes(include=[np.number])
    
    # C. Filter out any numeric metadata that might have slipped through
    features = [c for c in df_numeric.columns if c not in meta_cols]
    
    if len(features) < len(df.columns) - len(meta_cols):
        log(f"    - Note: Filtered out non-numeric columns. Keeping {len(features)} numeric features.")

    # 5. Rename Features
    feat_map = {}
    if suffix:
        for f in features:
            if not f.endswith(suffix):
                feat_map[f] = f"{f}{suffix}"
    
    if feat_map:
        df.rename(columns=feat_map, inplace=True)
        features = [feat_map.get(f, f) for f in features]
        
    # 6. Extract & Impute (In-Memory for single modality)
    log(f"    - Extracting {len(features)} features...")
    
    # Safety: Use df[features] which now strictly contains only numeric columns
    X_mod = df[features].values.astype(np.float32)
    
    # Free the DataFrame immediately
    del df, df_numeric
    gc.collect()
    
    # Fast Imputation (Median)
    if np.isnan(X_mod).any():
        log(f"    - Imputing missing values (Median)...")
        col_medians = np.nanmedian(X_mod, axis=0)
        inds = np.where(np.isnan(X_mod))
        X_mod[inds] = np.take(col_medians, inds[1])
        X_mod = np.nan_to_num(X_mod, nan=0.0)
    
    # 7. Dump to Temp File
    temp_path = os.path.join(output_dir, f"temp_{name}.joblib")
    log(f"    - Dumping to temp file: {temp_path}")
    joblib.dump(X_mod, temp_path)
    
    del X_mod
    gc.collect()
    
    return temp_path, features

def train_stage1(
    output_dir: str, 
    pc_path: str, 
    dict_path: str,
    beta_path: str,
    chalm_path: str,
    camda_path: str,
    sweep_file: str = None,
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
    min_cohorts: int = 2,
    min_cap: int = 30,
    max_cap: int = 600,
    median_mult: float = 1.0,
    **kwargs
) -> None:
    """
    Train the Stage 1 model using Disk-Based Streaming Assembly.
    """
    artifact_handler = Stage1Artifact(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Load Metadata & Dictionaries ---
    log(f"Loading Hallmark Dictionary from {dict_path}...")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Hallmark dictionary not found at {dict_path}")
    
    with open(dict_path, 'r') as f:
        hallmark_dict = json.load(f)

    # --- 2. Establish Anchor Samples (from PC file) ---
    log(f"Loading Context (PCs) from {pc_path}...")
    pc_data = pd.read_csv(pc_path)
    pc_data['sample_id'] = pc_data['sample_id'].astype(str)
    
    # Sort samples to ensure consistent alignment across modalities
    target_samples = sorted(pc_data['sample_id'].unique())
    log(f"Established Anchor: {len(target_samples)} samples from PC file.")
    
    # Filter PC data to this order
    pc_data = pc_data.set_index('sample_id').reindex(target_samples).reset_index()
    
    # --- 3. Stream Process Modalities ---
    temp_dir = os.path.join(output_dir, "temp_stream_cache")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # Define modalities
        tasks = [
            ('beta', beta_path, '_beta'),
            ('chalm', chalm_path, '_chalm'),
            ('camda', camda_path, '_camda')
        ]
        
        temp_files = []
        all_features = []
        
        # A. Process each modality independently (Low Peak RAM)
        for name, path, suffix in tasks:
            if path and os.path.exists(path):
                t_path, feats = process_and_dump_modality(name, path, target_samples, temp_dir, suffix)
                temp_files.append(t_path)
                all_features.extend(feats)
            else:
                log(f"Warning: Path for {name} not found or empty: {path}")
            
        # Add PCs to features
        pc_cols = [c for c in pc_data.columns if c.startswith('RF_PC')]
        all_features.extend(pc_cols)
        
        # B. Assemble Final Memmap
        log("Assembling Final X Matrix on Disk...")
        n_samples = len(target_samples)
        n_features = len(all_features)
        
        final_memmap_path = os.path.join(temp_dir, "X_final.dat")
        X_final = np.memmap(final_memmap_path, dtype='float32', mode='w+', shape=(n_samples, n_features))
        
        current_col = 0
        
        # Write Modalities
        for t_path in temp_files:
            log(f"  > Merging {os.path.basename(t_path)} into final matrix...")
            X_part = joblib.load(t_path) 
            n_cols = X_part.shape[1]
            X_final[:, current_col : current_col + n_cols] = X_part
            current_col += n_cols
            del X_part
            gc.collect()
            
        # Write PCs
        log(f"  > Merging PCs into final matrix...")
        X_pc = pc_data[pc_cols].values.astype(np.float32)
        X_final[:, current_col:] = X_pc
        
        X_final.flush()
        log(f"Final Data Assembly Complete. Shape: {X_final.shape}")
        
        # Switch to Read-Only
        X_selected = np.memmap(final_memmap_path, dtype='float32', mode='r', shape=(n_samples, n_features))
        
        # --- Prepare Targets & Groups ---
        log("Loading metadata from Beta...")
        df_meta = pd.read_pickle(beta_path)
        
        # Robust metadata extraction
        # Ensure we have all needed columns
        needed = ['sample_id', 'project_id', 'Tissue', 'Age', 'Sex', 'Is_Healthy']
        for c in needed:
            if c not in df_meta.columns and c.lower() in df_meta.columns:
                df_meta.rename(columns={c.lower(): c}, inplace=True)
                
        df_meta = df_meta[needed].copy()
        df_meta['sample_id'] = df_meta['sample_id'].astype(str)
        df_meta = df_meta.set_index('sample_id').reindex(target_samples).reset_index()
        
        y = df_meta['Age'].values.astype(np.float32)
        groups = df_meta['project_id']
        tissues = df_meta['Tissue']
        sample_ids = df_meta['sample_id'].values
        meta_sex = df_meta['Sex'].values
        meta_healthy = df_meta['Is_Healthy'].values

        # --- 4. Training Loop (Standard) ---
        trans = AgeTransformer(adult_age=20)
        y_trans = trans.transform(y)

        log(f"Stage 1 Strategy: Training on FULL feature set (Memmap active).")
        log("Optimizing Hyperparameters (Macro + Micro)...")
        
        best_model, sweep_df = tune_elasticnet_macro_micro(
            X=X_selected, 
            y=y_trans, 
            groups=groups, 
            tissues=tissues, 
            trans_func=trans,
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

        log("Saving Sweep Report...")
        artifact_handler.save("Stage1_Sweep_Report", sweep_df)

        # --- 5. OOF & Final Model ---
        log("Generating Final OOF...")
        folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
        oof_preds_linear = np.zeros(len(y), dtype=np.float32)
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Sampler logic ...
            train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
            train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
            df_train_meta = pd.DataFrame({'Tissue': train_tissues, 'project_id': train_groups, 'original_idx': train_idx})
            
            df_bal, _ = adaptive_sampler(df_train_meta, min_coh=min_cohorts, min_c=min_cap, max_c=max_cap, mult=median_mult, seed=seed + fold_idx)
            final_train_idx = df_bal['original_idx'].values
            
            model_fold = clone(best_model)
            model_fold.fit(X_selected[final_train_idx], y_trans[final_train_idx])
            
            val_preds_trans = model_fold.predict(X_selected[val_idx])
            val_preds_linear = trans.inverse_transform(val_preds_trans)
            oof_preds_linear[val_idx] = val_preds_linear

        oof_metrics = compute_regression_metrics(y, oof_preds_linear)
        log(f"Stage 1 Final OOF Metrics: {oof_metrics}")

        log("Retraining final model...")
        df_full_meta = pd.DataFrame({'Tissue': tissues, 'project_id': groups, 'original_idx': np.arange(len(y))})
        df_full_bal, _ = adaptive_sampler(df_full_meta, min_cohorts, min_cap, max_cap, mult=median_mult, seed=seed)
        final_idx = df_full_bal['original_idx'].values
        
        final_model = clone(best_model)
        final_model.fit(X_selected[final_idx], y_trans[final_idx])

        # --- 6. Dictionary Mapping (MODIFIED: Skip Orthogonalization) ---
        log("Generating Feature Dictionary...")
        log("⚠️ Note: Orthogonalization SKIPPED to retain maximum signal for Stage 2.")
        
        hallmark_dict_suffixed = defaultdict(list)
        available_set = set(all_features) 
        
        # Simply map the Hallmark CpGs to the available features (beta, chalm, camda)
        # without filtering them against the global age signal.
        for h, cpgs in hallmark_dict.items():
            iter_cpgs = cpgs if isinstance(cpgs, list) else cpgs.values() if isinstance(cpgs, dict) else cpgs
            for cpg in iter_cpgs:
                if not isinstance(cpg, str): continue 
                for suffix in ["_beta", "_chalm", "_camda"]:
                    feat = f"{cpg}{suffix}"
                    if feat in available_set:
                        hallmark_dict_suffixed[h].append(feat)
        
        # [CRITICAL CHANGE] No orthogonalize_by_correlation call
        # Pass the full set of mapped features to Stage 2
        final_ortho_dict = dict(hallmark_dict_suffixed)
        
        count_summary = {k: len(v) for k, v in final_ortho_dict.items()}
        log(f"Feature Dictionary Ready. Feature counts per hallmark: {count_summary}")

        # --- Save Artifacts ---
        artifact_handler.save_global_model(final_model)
        artifact_handler.save("stage1_features", all_features) 
        artifact_handler.save_orthogonalized_dict(final_ortho_dict)
        
        oof_df = pd.DataFrame({
            "sample_id": sample_ids,
            "age": y,
            "pred_age": oof_preds_linear,
            "residual": y - oof_preds_linear,
            "project_id": groups.values,
            "Tissue": tissues.values,
            "Sex": meta_sex,
            "Is_Healthy": meta_healthy
        })
        artifact_handler.save_oof_predictions(oof_df)
        log(f"Stage 1 completed. Outputs in {output_dir}")

    finally:
        # --- CLEANUP ---
        if os.path.exists(temp_dir):
            try:
                if 'X_selected' in locals(): del X_selected 
                if 'X_final' in locals(): del X_final
                gc.collect()
                shutil.rmtree(temp_dir)
                log(f"Cleaned up temp storage: {temp_dir}")
            except Exception as e:
                log(f"Warning: Failed to clean temp dir: {e}")

def predict_stage1(artifact_dir, input_path, output_path):
    artifact_handler = Stage1Artifact(artifact_dir)
    log(f"Loading model from {artifact_dir}...")
    model = artifact_handler.load_global_model()
    feature_cols = artifact_handler.load("stage1_features")
    
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_pickle(input_path)
    
    if "Sex_encoded" in feature_cols and "Sex_encoded" not in data.columns:
         if "Sex" in data.columns:
             data["Sex_encoded"] = data["Sex"].map({"F": 0, "M": 1, "Female": 0, "Male": 1}).fillna(0)
         else:
             data["Sex_encoded"] = 0

    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        for c in missing: data[c] = 0.0
    
    X = data[feature_cols].values
    trans = AgeTransformer(adult_age=20)
    preds_trans = model.predict(X)
    preds_linear = trans.inverse_transform(preds_trans)
    
    out_df = pd.DataFrame()
    if "sample_id" in data.columns:
        out_df["sample_id"] = data["sample_id"]
    out_df["pred_age_stage1"] = preds_linear
    
    if "age" in data.columns:
        out_df["age"] = data["age"]
        out_df["residual_stage1"] = out_df["age"] - out_df["pred_age_stage1"]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    log(f"Predictions saved to {output_path}")