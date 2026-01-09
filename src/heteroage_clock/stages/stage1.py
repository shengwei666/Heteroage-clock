"""
heteroage_clock.stages.stage1

Stage 1: Global Anchor
Updates:
- [Critical] Implemented INCREMENTAL MEMMAP WRITE to eliminate memory peaks during X_full creation.
- [Critical] Retains SEQUENTIAL DATA LOADING for input matrices.
- Forces float32 everywhere.
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
from heteroage_clock.data.assemble import assemble_features, filter_and_impute
from heteroage_clock.core.age_transform import AgeTransformer
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.optimization import tune_elasticnet_macro_micro
from heteroage_clock.core.selection import orthogonalize_by_correlation
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage1 import Stage1Artifact
from heteroage_clock.core.sampling import adaptive_sampler

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
    Train the Stage 1 model.
    Optimized for LOW PEAK MEMORY via Sequential Loading & Incremental Memmap.
    """
    artifact_handler = Stage1Artifact(output_dir)
    
    # --- 1. Load Metadata & Dictionaries ---
    log(f"Loading Hallmark Dictionary from {dict_path}...")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Hallmark dictionary not found at {dict_path}")
    
    with open(dict_path, 'r') as f:
        hallmark_dict = json.load(f)

    log(f"Loading Context (PCs) from {pc_path}...")
    pc_data = pd.read_csv(pc_path)
    
    # --- 2. Sequential Assembly (Low Memory Peak) ---
    log("Starting Sequential Feature Assembly...")

    # Step A: Load Beta (Base)
    log(f"  > Loading Beta Matrix: {beta_path}")
    cpg_beta = pd.read_pickle(beta_path)
    
    # Initial Assembly: Beta + PCs
    # We use cpg_beta as both the 'beta' modality and the 'metadata' anchor
    log("  > Merging Beta + PCs...")
    assembled_data = assemble_features(
        cpg_beta=cpg_beta, 
        chalm_data=None, 
        camda_data=None, 
        pc_data=pc_data, 
        metadata=cpg_beta
    )
    
    # Cleanup Beta immediately
    del cpg_beta
    gc.collect()
    log(f"  > Beta merged. Current shape: {assembled_data.shape}")

    # Step B: Load Chalm & Merge
    log(f"  > Loading Chalm Matrix: {chalm_path}")
    chalm_data = pd.read_pickle(chalm_path)
    
    log("  > Merging Chalm...")
    # Pass assembled_data as metadata to keep growing it
    assembled_data = assemble_features(
        cpg_beta=None, 
        chalm_data=chalm_data, 
        camda_data=None, 
        pc_data=None, 
        metadata=assembled_data
    )
    
    # Cleanup Chalm
    del chalm_data
    gc.collect()
    log(f"  > Chalm merged. Current shape: {assembled_data.shape}")

    # Step C: Load Camda & Merge
    log(f"  > Loading Camda Matrix: {camda_path}")
    camda_data = pd.read_pickle(camda_path)
    
    log("  > Merging Camda...")
    assembled_data = assemble_features(
        cpg_beta=None, 
        chalm_data=None, 
        camda_data=camda_data, 
        pc_data=None, 
        metadata=assembled_data
    )
    
    # Cleanup Camda
    del camda_data
    del pc_data
    gc.collect()
    log(f"  > Camda merged. Final RAM Shape: {assembled_data.shape}")

    # Step D: Filter & Impute
    assembled_data = filter_and_impute(assembled_data)

    metadata_cols = ["sample_id", "age", "project_id", "Tissue", "Sex", "Is_Healthy", "Sex_encoded"]
    all_cols = assembled_data.columns.tolist()
    feature_candidates = [c for c in all_cols if c not in metadata_cols]
    
    # --- [MEMORY OPTIMIZATION: Incremental Memmap Write] ---
    # Instead of creating X_full_ram (which doubles memory), we write to disk in chunks.
    log(f"Preparing Memory Mapping (Incremental Write) to support {n_jobs} jobs...")
    
    temp_memmap_dir = os.path.join(output_dir, "temp_memmap_cache")
    if os.path.exists(temp_memmap_dir):
        shutil.rmtree(temp_memmap_dir)
    os.makedirs(temp_memmap_dir)
    
    # 1. Extract Metadata into RAM (Small enough)
    y = assembled_data["age"].values.astype(np.float32)
    groups = assembled_data["project_id"]
    tissues = assembled_data["Tissue"]
    sample_ids = assembled_data["sample_id"].values
    
    meta_sex = assembled_data["Sex"].values if "Sex" in assembled_data.columns else None
    meta_healthy = assembled_data["Is_Healthy"].values if "Is_Healthy" in assembled_data.columns else None

    # 2. Initialize Memmap File on Disk
    shape = (len(assembled_data), len(feature_candidates))
    mm_file = os.path.join(temp_memmap_dir, "raw_data.dat")
    
    log(f"Initializing memmap on disk: {shape} ...")
    fp = np.memmap(mm_file, dtype='float32', mode='w+', shape=shape)
    
    # 3. Write Features in Chunks
    # This ensures we never hold the full X matrix in RAM.
    chunk_size = 5000 # Number of columns to process at a time
    total_cols = len(feature_candidates)
    
    for i in range(0, total_cols, chunk_size):
        end = min(i + chunk_size, total_cols)
        cols_chunk = feature_candidates[i:end]
        
        # Write chunk to disk
        fp[:, i:end] = assembled_data[cols_chunk].values.astype(np.float32)
        
        # Flush to ensure data hits the disk
        fp.flush()
        
        if i > 0 and i % 20000 == 0:
            log(f"  > Written {end}/{total_cols} columns to disk...")
            gc.collect()
            
    log(f"  > Finished writing all {total_cols} columns.")
    
    # 4. Clean up the massive DataFrame immediately
    del assembled_data
    del fp # Close the write-mode memmap
    gc.collect()
    
    # 5. Re-open Memmap in Read-Only Mode
    # This object acts like a numpy array but reads from disk on demand
    X_selected = np.memmap(mm_file, dtype='float32', mode='r', shape=shape)
    log(f"Data mapped from disk (Read-Only). Shape: {X_selected.shape}. RAM is clean.")
    
    # --- [MEMORY OPTIMIZATION END] ---

    try:
        # --- 3. Transform Target ---
        trans = AgeTransformer(adult_age=20)
        y_trans = trans.transform(y)

        # --- 4. Strategy ---
        log(f"Stage 1 Strategy: Training on FULL feature set (Memmap active).")

        # --- 5. Optimization ---
        log("Optimizing Hyperparameters (Macro + Micro)...")
        
        # X_selected is the Memmap. Joblib handles it efficiently.
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

        # --- 6. Final OOF Generation ---
        log("Generating Final OOF...")
        folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
        oof_preds_linear = np.zeros(len(y), dtype=np.float32)
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
            train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
            
            df_train_meta = pd.DataFrame({'Tissue': train_tissues, 'project_id': train_groups, 'original_idx': train_idx})
            df_bal, _ = adaptive_sampler(df_train_meta, min_coh=min_cohorts, min_c=min_cap, max_c=max_cap, mult=median_mult, seed=seed + fold_idx)
            final_train_idx = df_bal['original_idx'].values
            
            model_fold = clone(best_model)
            model_fold.fit(X_selected[final_train_idx], y_trans[final_train_idx])
            
            # Predict
            # X_selected is memmap, so this reads from disk
            val_preds_trans = model_fold.predict(X_selected[val_idx])
            val_preds_linear = trans.inverse_transform(val_preds_trans)
            oof_preds_linear[val_idx] = val_preds_linear

        oof_metrics = compute_regression_metrics(y, oof_preds_linear)
        log(f"Stage 1 Final OOF Metrics: {oof_metrics}")

        # --- 7. Final Training ---
        log("Retraining final model...")
        df_full_meta = pd.DataFrame({'Tissue': tissues, 'project_id': groups, 'original_idx': np.arange(len(y))})
        df_full_bal, _ = adaptive_sampler(df_full_meta, min_coh=min_cohorts, min_c=min_cap, max_c=max_cap, mult=median_mult, seed=seed)
        final_idx = df_full_bal['original_idx'].values
        
        final_model = clone(best_model)
        final_model.fit(X_selected[final_idx], y_trans[final_idx])

        # --- 8. Orthogonalization ---
        log("Generating Orthogonalized Dictionary...")
        hallmark_dict_suffixed = defaultdict(list)
        available_set = set(feature_candidates)
        
        for h, cpgs in hallmark_dict.items():
            iter_cpgs = cpgs if isinstance(cpgs, list) else cpgs.values() if isinstance(cpgs, dict) else cpgs
            for cpg in iter_cpgs:
                if not isinstance(cpg, str): continue 
                for suffix in ["_beta", "_chalm", "_camda"]:
                    feat = f"{cpg}{suffix}"
                    if feat in available_set:
                        hallmark_dict_suffixed[h].append(feat)
        
        # Pass Memmap X to orthogonalization too!
        final_ortho_dict = orthogonalize_by_correlation(
            X=X_selected, 
            y=y_trans, 
            feature_names=feature_candidates, 
            hallmark_mapping=hallmark_dict_suffixed
        )
        
        # --- Save Artifacts ---
        log("Saving artifacts...")
        artifact_handler.save_global_model(final_model)
        artifact_handler.save("stage1_features", feature_candidates) 
        artifact_handler.save_orthogonalized_dict(final_ortho_dict)
        
        oof_df = pd.DataFrame({
            "sample_id": sample_ids,
            "age": y,
            "pred_age": oof_preds_linear,
            "residual": y - oof_preds_linear,
            "project_id": groups.values,
            "Tissue": tissues.values
        })
        
        if meta_sex is not None: oof_df["Sex"] = meta_sex
        if meta_healthy is not None: oof_df["Is_Healthy"] = meta_healthy
            
        artifact_handler.save_oof_predictions(oof_df)
        log(f"Stage 1 completed. Outputs in {output_dir}")

    finally:
        # --- CLEANUP ---
        if os.path.exists(temp_memmap_dir):
            try:
                # Close memmap references
                if 'X_selected' in locals():
                    del X_selected 
                if 'fp' in locals():
                    del fp
                gc.collect()
                shutil.rmtree(temp_memmap_dir)
                log(f"Cleaned up temp memmap storage: {temp_memmap_dir}")
            except Exception as e:
                log(f"Warning: Failed to clean temp dir {temp_memmap_dir}: {e}")

# Predict function unchanged...
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