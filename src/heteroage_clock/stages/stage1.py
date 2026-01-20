"""
heteroage_clock.stages.stage1

Stage 1: Global Anchor
Updates:
- [Fix] Robust handling of 'sample_id' when reading pickle files (handles unnamed indices).
- [Fix] Added process synchronization (locking) to prevent Race Conditions.
- [Fix] Workers wait for leader instead of crashing.
- [Major] Integrated Optuna for automatic hyperparameter tuning.
- [Optimization] Implements Sparse Inference.
- [Inference] predict_stage1 accepts raw input paths and preserves Metadata (Age/Tissue).
"""

import os
import json
import gc
import shutil
import joblib
import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from collections import defaultdict
from typing import List, Optional

from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.core.age_transform import AgeTransformer
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.optimization import optimize_elasticnet_optuna
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage1 import Stage1Artifact
from heteroage_clock.core.sampling import adaptive_sampler

# --- Helper: Robust ID Extractor ---
def _ensure_sample_id(df):
    """
    Robustly ensures a 'sample_id' column exists.
    If missing, tries to reset index and rename it.
    """
    if 'sample_id' not in df.columns:
        # Check for case-insensitive match
        for col in df.columns:
            if col.lower() == 'sample_id':
                df.rename(columns={col: 'sample_id'}, inplace=True)
                return df
                
        # If still missing, assume the Index is the ID
        df = df.reset_index()
        
        # If the index was named 'sample_id', we are done.
        # If not, we need to rename the column created from the index.
        if 'sample_id' not in df.columns:
            # If index had no name, pandas usually names it 'index'
            if 'index' in df.columns:
                df.rename(columns={'index': 'sample_id'}, inplace=True)
            else:
                # Fallback: Rename the first column (which comes from the index)
                # This handles cases where index was named "ID", "geo_accession", etc.
                first_col = df.columns[0]
                df.rename(columns={first_col: 'sample_id'}, inplace=True)
    
    return df

# --- Helper: Low-Memory Modality Processor ---
def process_and_dump_modality(name, path, target_samples, output_dir, suffix):
    """
    Loads one modality, aligns to target samples, imputes, and dumps to a temp joblib file.
    Returns: (temp_file_path, feature_names_list)
    """
    log(f"  > [Stream] Processing {name} from {path}...")
    
    # 1. Load Data
    df = pd.read_pickle(path)
    
    # 2. Standardize Metadata [FIXED]
    df = _ensure_sample_id(df)
    df['sample_id'] = df['sample_id'].astype(str)
    
    # 3. Align Samples
    df = df.set_index('sample_id')
    
    # Handle cases where target_samples might be missing in this modality
    common_samples = df.index.intersection(target_samples)
    log(f"    - Found {len(common_samples)}/{len(target_samples)} target samples.")
    
    df = df.reindex(target_samples)
    
    # 4. Identify Features (Exclude Metadata)
    meta_cols = {'project_id', 'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy', 'Sex_encoded', 'sample_id', 'sample_key'}
    
    # Select ONLY numeric columns first
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Filter out any numeric metadata that might have slipped through
    features = [c for c in df_numeric.columns if c not in meta_cols]
    
    if len(features) < len(df.columns) - len(meta_cols):
        log(f"    - Note: Filtered out non-numeric columns. Keeping {len(features)} numeric features.")

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
    log(f"    - Extracting {len(features)} features...")
    
    # Use df[features] which strictly contains only numeric columns
    X_mod = df[features].values.astype(np.float32)
    
    # Free memory immediately
    del df, df_numeric
    gc.collect()
    
    # Fast Imputation (Median)
    if np.isnan(X_mod).any():
        log(f"    - Imputing missing values (Median)...")
        col_medians = np.nanmedian(X_mod, axis=0)
        inds = np.where(np.isnan(X_mod))
        X_mod[inds] = np.take(col_medians, inds[1])
        X_mod = np.nan_to_num(X_mod, nan=0.0)
    
    # 7. Dump to Disk (Temp Joblib)
    temp_path = os.path.join(output_dir, f"temp_{name}.joblib")
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
    Train the Stage 1 model using Optuna for joint hyperparameter optimization.
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
    
    # [FIXED] Robust ID check
    pc_data = _ensure_sample_id(pc_data)
    pc_data['sample_id'] = pc_data['sample_id'].astype(str)
    
    target_samples = sorted(pc_data['sample_id'].unique())
    log(f"Established Anchor: {len(target_samples)} samples from PC file.")
    
    pc_data = pc_data.set_index('sample_id').reindex(target_samples).reset_index()
    
    # --- 3. Synchronized Data Assembly ---
    temp_dir = os.path.join(output_dir, "temp_stream_cache")
    os.makedirs(temp_dir, exist_ok=True)
    
    final_memmap_path = os.path.join(temp_dir, "X_final.dat")
    features_list_path = os.path.join(temp_dir, "features_list.pkl")
    lock_dir = os.path.join(temp_dir, "assembly_lock")
    
    is_data_ready = os.path.exists(final_memmap_path) and os.path.exists(features_list_path)
    
    if not is_data_ready:
        is_builder = False
        try:
            os.mkdir(lock_dir)
            is_builder = True
            log(">>> Acquired Lock. I am the Data Builder.")
        except FileExistsError:
            is_builder = False
            log(">>> Lock exists. Another worker is building data. I will wait.")
        
        if is_builder:
            try:
                log("Data matrix missing or incomplete. Assembling...")
                temp_files = []
                all_features = []
                
                tasks = [('beta', beta_path, '_beta'), ('chalm', chalm_path, '_chalm'), ('camda', camda_path, '_camda')]
                
                # A. Process each modality
                for name, path, suffix in tasks:
                    if path and os.path.exists(path):
                        t_path, feats = process_and_dump_modality(name, path, target_samples, temp_dir, suffix)
                        temp_files.append(t_path)
                        all_features.extend(feats)
                
                pc_cols = [c for c in pc_data.columns if c.startswith('RF_PC')]
                all_features.extend(pc_cols)
                
                # B. Assemble Final Memmap
                n_samples = len(target_samples)
                n_features = len(all_features)
                
                log(f"Writing Final X Matrix ({n_samples}x{n_features})...")
                X_final = np.memmap(final_memmap_path, dtype='float32', mode='w+', shape=(n_samples, n_features))
                
                current_col = 0
                for t_path in temp_files:
                    X_part = joblib.load(t_path) 
                    n_cols = X_part.shape[1]
                    X_final[:, current_col : current_col + n_cols] = X_part
                    current_col += n_cols
                    del X_part
                    gc.collect()
                    
                X_pc = pc_data[pc_cols].values.astype(np.float32)
                X_final[:, current_col:] = X_pc
                X_final.flush()
                del X_final 
                
                joblib.dump(all_features, features_list_path)
                log("Data Assembly Complete.")

            except Exception as e:
                log(f"CRITICAL ERROR during assembly: {e}")
                if os.path.exists(final_memmap_path): os.remove(final_memmap_path)
                if os.path.exists(features_list_path): os.remove(features_list_path)
                raise e
            finally:
                try: os.rmdir(lock_dir); log(">>> Released Lock.")
                except: pass
        else:
            wait_time = 0
            while not (os.path.exists(final_memmap_path) and os.path.exists(features_list_path)):
                time.sleep(10)
                wait_time += 10
                if wait_time > 7200: raise TimeoutError("Data assembly timed out.")
            log("Data assembly finished by leader.")
    else:
        log("Found existing valid data cache. Skipping assembly.")

    # --- LOAD DATA ---
    all_features = joblib.load(features_list_path)
    n_samples = len(target_samples)
    n_features = len(all_features)

    try:
        X_selected = np.memmap(final_memmap_path, dtype='float32', mode='r', shape=(n_samples, n_features))
        
        log("Loading metadata from Beta...")
        df_meta = pd.read_pickle(beta_path)
        # [FIXED] Robust ID handling for metadata too
        df_meta = _ensure_sample_id(df_meta)
        
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

        # --- Optimization Loop ---
        trans = AgeTransformer(adult_age=20)
        y_trans = trans.transform(y)

        search_config = {
            'min_cap_low': min_cap_low, 'min_cap_high': min_cap_high,
            'max_cap_low': max_cap_low, 'max_cap_high': max_cap_high,
            'median_mult_low': median_mult_low, 'median_mult_high': median_mult_high,
            'alpha_low': alpha_low, 'alpha_high': alpha_high,
            'l1_low': l1_low, 'l1_high': l1_high
        }

        best_model, trials_df = optimize_elasticnet_optuna(
            X=X_selected, y=y_trans, groups=groups, tissues=tissues, 
            output_dir=output_dir, trans_func=trans, n_trials=n_trials, n_jobs=n_jobs, n_splits=n_splits, 
            seed=seed, max_iter=max_iter, min_cohorts=min_cohorts, search_config=search_config,
            storage=storage, study_name=study_name
        )

        artifact_handler.save("Stage1_Optuna_Trials", trials_df)
        best_sampling = getattr(best_model, 'best_sampling_params_', {})
        
        # --- OOF & Final Model ---
        log("Generating Final OOF...")
        folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
        oof_preds_linear = np.zeros(len(y), dtype=np.float32)
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
            train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
            df_train_meta = pd.DataFrame({'Tissue': train_tissues, 'project_id': train_groups, 'original_idx': train_idx})
            
            df_bal, _ = adaptive_sampler(df_train_meta, min_coh=min_cohorts, seed=seed+fold_idx, **best_sampling)
            
            model_fold = clone(best_model)
            model_fold.fit(X_selected[df_bal['original_idx'].values], y_trans[df_bal['original_idx'].values])
            
            val_preds = model_fold.predict(X_selected[val_idx])
            oof_preds_linear[val_idx] = trans.inverse_transform(val_preds)

        log("Retraining final global model...")
        df_full_meta = pd.DataFrame({'Tissue': tissues, 'project_id': groups, 'original_idx': np.arange(len(y))})
        df_full_bal, _ = adaptive_sampler(df_full_meta, min_coh=min_cohorts, seed=seed, **best_sampling)
        
        final_model = clone(best_model)
        final_model.fit(X_selected[df_full_bal['original_idx'].values], y_trans[df_full_bal['original_idx'].values])

        # --- Dictionary Mapping ---
        log("Generating Feature Dictionary...")
        hallmark_dict_suffixed = defaultdict(list)
        available_set = set(all_features) 
        
        for h, cpgs in hallmark_dict.items():
            iter_cpgs = cpgs if isinstance(cpgs, list) else cpgs.values() if isinstance(cpgs, dict) else cpgs
            for cpg in iter_cpgs:
                if not isinstance(cpg, str): continue 
                for suffix in ["_beta", "_chalm", "_camda"]:
                    feat = f"{cpg}{suffix}"
                    if feat in available_set: hallmark_dict_suffixed[h].append(feat)
        
        final_ortho_dict = dict(hallmark_dict_suffixed)

        artifact_handler.save_global_model(final_model)
        artifact_handler.save("stage1_features", all_features) 
        artifact_handler.save_orthogonalized_dict(final_ortho_dict)
        
        oof_df = pd.DataFrame({
            "sample_id": sample_ids, "age": y, "pred_age": oof_preds_linear,
            "project_id": groups.values, "Tissue": tissues.values, "Sex": meta_sex, "Is_Healthy": meta_healthy
        })
        artifact_handler.save_oof_predictions(oof_df)
        log(f"Stage 1 completed. Outputs in {output_dir}")

    finally:
        if 'X_selected' in locals(): del X_selected 
        if 'X_final' in locals(): del X_final
        gc.collect()
        if not storage and os.path.exists(temp_dir): shutil.rmtree(temp_dir)

def predict_stage1(
    artifact_dir: str, 
    input_beta: str, 
    output_path: str, 
    input_chalm: str = None, 
    input_camda: str = None, 
    keep_mmap_path: str = None, 
    keep_meta_path: str = None
):
    """
    End-to-End Inference for Stage 1.
    [Updated] Now explicitly preserves 'Age', 'Tissue', and 'project_id' in output.
    """
    log(">>> [Stage 1] Starting End-to-End Inference...")

    # --- 1. Load Artifacts ---
    artifact_handler = Stage1Artifact(artifact_dir)
    model_pipeline = artifact_handler.load_global_model()
    target_features = artifact_handler.load("stage1_features")
    
    # --- 2. Prepare Data ---
    log(f"Loading metadata from {input_beta}...")
    df_meta = pd.read_pickle(input_beta)
    
    # [FIXED] Robust ID handling
    df_meta = _ensure_sample_id(df_meta)
    df_meta['sample_id'] = df_meta['sample_id'].astype(str)
    
    target_samples = df_meta['sample_id'].tolist()
    n_samples = len(target_samples)
    n_features = len(target_features)
    
    # [FIXED] Extract Metadata for Output
    true_age = np.full(n_samples, np.nan)
    for col in ['Age', 'age', 'True_Age', 'chronological_age']:
        if col in df_meta.columns:
            true_age = df_meta[col].values
            break
            
    tissue_vals = np.full(n_samples, "Unknown")
    for col in ['Tissue', 'tissue', 'Source', 'source']:
        if col in df_meta.columns:
            tissue_vals = df_meta[col].values
            break
            
    proj_vals = df_meta.get('project_id', np.full(n_samples, "Unknown"))

    # Determine temp paths
    temp_dir = os.path.dirname(output_path)
    os.makedirs(temp_dir, exist_ok=True)
    
    final_mmap_path = keep_mmap_path if keep_mmap_path else os.path.join(temp_dir, f"temp_s1_X_{os.getpid()}.dat")
    if keep_meta_path: df_meta.to_pickle(keep_meta_path)
    
    log(f"Assembling Input Matrix ({n_samples}x{n_features})...")
    X_final = np.memmap(final_mmap_path, dtype='float32', mode='w+', shape=(n_samples, n_features))
    feat_to_idx = {name: i for i, name in enumerate(target_features)}
    
    tasks = [('beta', input_beta, '_beta')]
    if input_chalm: tasks.append(('chalm', input_chalm, '_chalm'))
    if input_camda: tasks.append(('camda', input_camda, '_camda'))
    
    for name, path, suffix in tasks:
        if not path or not os.path.exists(path): continue
        log(f"Processing {name}...")
        df_mod = pd.read_pickle(path)
        
        df_mod = _ensure_sample_id(df_mod)
        df_mod['sample_id'] = df_mod['sample_id'].astype(str)
        df_mod = df_mod.set_index('sample_id').reindex(target_samples)
        
        col_map = {}
        data_cols = []
        for col in df_mod.columns:
            if col in ['sample_id', 'project_id', 'Tissue', 'Age', 'Sex', 'Is_Healthy']: continue
            if not np.issubdtype(df_mod[col].dtype, np.number): continue
            
            feat_name = f"{col}{suffix}"
            if feat_name in feat_to_idx:
                col_map[col] = feat_to_idx[feat_name]
                data_cols.append(col)
                
        if not data_cols: continue
            
        X_mod = df_mod[data_cols].values.astype(np.float32)
        if np.isnan(X_mod).any():
            col_medians = np.nanmedian(X_mod, axis=0)
            inds = np.where(np.isnan(X_mod))
            X_mod[inds] = np.take(col_medians, inds[1])
            X_mod = np.nan_to_num(X_mod, nan=0.0)
            
        target_indices = [col_map[c] for c in data_cols]
        X_final[:, target_indices] = X_mod
        
        del df_mod, X_mod
        gc.collect()

    X_final.flush()
    
    # --- 3. Run Prediction ---
    log("Running Prediction...")
    X_read = np.memmap(final_mmap_path, dtype='float32', mode='r', shape=(n_samples, n_features))
    
    try:
        if 'standardscaler' in model_pipeline.named_steps: scaler = model_pipeline.named_steps['standardscaler']
        else: scaler = model_pipeline.steps[0][1]
            
        if 'elasticnet' in model_pipeline.named_steps: estimator = model_pipeline.named_steps['elasticnet']
        else: estimator = model_pipeline.steps[-1][1]

        if hasattr(estimator, 'coef_'):
            full_coefs, intercept = estimator.coef_, estimator.intercept_
        else:
            full_coefs, intercept = estimator.best_estimator_.coef_, estimator.best_estimator_.intercept_
            
        full_coefs = np.array(full_coefs).flatten()
        active_indices = np.where(np.abs(full_coefs) > 1e-12)[0]
        
        final_preds_trans = np.zeros(n_samples, dtype=np.float32)
        if len(active_indices) > 0:
            active_means = scaler.mean_[active_indices]
            active_scales = scaler.scale_[active_indices]
            active_weights = full_coefs[active_indices]
            
            for start in range(0, n_samples, 4096):
                end = min(start + 4096, n_samples)
                X_batch = X_read[start:end][:, active_indices]
                X_batch_scaled = (X_batch - active_means) / active_scales
                final_preds_trans[start:end] = X_batch_scaled @ active_weights + intercept
        else:
            final_preds_trans = np.full(n_samples, intercept)

    except Exception as e:
        log(f"Sparse prediction failed ({e}). Fallback to full predict.")
        final_preds_trans = np.zeros(n_samples, dtype=np.float32)
        for start in range(0, n_samples, 1024):
            end = min(start + 1024, n_samples)
            final_preds_trans[start:end] = model_pipeline.predict(X_read[start:end])

    trans = AgeTransformer(adult_age=20)
    final_preds = trans.inverse_transform(final_preds_trans)
    
    # [FIXED] Save result with Metadata
    res_df = pd.DataFrame({
        "sample_id": target_samples,
        "Age": true_age,
        "Tissue": tissue_vals,
        "project_id": proj_vals,
        "pred_age_stage1": final_preds
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    res_df.to_csv(output_path, index=False)
    
    del X_read, X_final
    gc.collect()
    
    if not keep_mmap_path and os.path.exists(final_mmap_path):
        try: os.remove(final_mmap_path)
        except: pass
        
    log(f"Stage 1 Prediction Saved (with Metadata): {output_path}")