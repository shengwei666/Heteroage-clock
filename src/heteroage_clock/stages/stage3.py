"""
heteroage_clock.stages.stage3

Stage 3: Context-Aware Fusion (Ridge Regression with Optuna)
Updates:
- [Fix] Added robust column name handling for Stage 1 inputs.
- [Fix] EXPLICITLY EXCLUDES 'pred_resid_Global_Union' from input features.
- [Preserved] Uses Ridge Regression and Global+Tissue-Specific architecture.
- [Feature] Optuna auto-tunes Ridge regularization (alpha).
- [Distributed] Added NFS-Safe Journal Storage support.
- [Update] Output accumulates Stage 1 + Stage 2 + Stage 3 results.
"""

import os
import gc
import joblib
import pandas as pd
import numpy as np
import optuna
from typing import Optional
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from optuna.storages import JournalStorage, JournalFileStorage

from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.artifacts.stage3 import Stage3Artifact
from heteroage_clock.utils.logging import log

def train_stage3(
    output_dir: str,
    stage1_oof: str,
    stage2_oof: str,
    pc_path: str,
    # --- Optuna Config ---
    n_trials: int = 0, 
    alpha: float = 1.0, 
    min_samples_for_tissue: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    # Distributed
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Train Stage 3 fusion models (Ridge) to correct Stage 1 error.
    Optimizes Global Model via Optuna (Distributed-ready & NFS-Safe).
    """
    artifact_handler = Stage3Artifact(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    log(">>> [Stage 3 Training] Training residual fusion models (Ridge)...")
    
    # --- 1. Load Data ---
    df_s1 = pd.read_csv(stage1_oof)
    df_s2 = pd.read_csv(stage2_oof)
    
    # [FIX] Robust Column Renaming for Stage 1
    if 'pred_age_stage1' not in df_s1.columns:
        if 'Stage1_Pred' in df_s1.columns:
            df_s1.rename(columns={'Stage1_Pred': 'pred_age_stage1'}, inplace=True)
        elif 'pred_age' in df_s1.columns:
            df_s1.rename(columns={'pred_age': 'pred_age_stage1'}, inplace=True)
        elif 'Global_Pred_Age' in df_s1.columns:
            df_s1.rename(columns={'Global_Pred_Age': 'pred_age_stage1'}, inplace=True)
        else:
            raise KeyError(f"Could not find Stage 1 prediction column in {stage1_oof}. Available columns: {df_s1.columns.tolist()}")

    # [FIX] Robust Column Renaming for True Age
    if 'age' not in df_s1.columns and 'Age' in df_s1.columns:
        df_s1.rename(columns={'Age': 'age'}, inplace=True)
        
    # Auto-detect Expert columns (Stage 2 now adds 'pred_resid_' prefix)
    expert_cols = [c for c in df_s2.columns if c.startswith("pred_resid_")]
    if not expert_cols:
         # Fallback for backward compatibility
         expert_cols = [c for c in df_s2.columns if c.startswith("pred_age_") and c != 'pred_age_stage1']
         
    if not expert_cols:
         expert_cols = [c for c in df_s2.columns if c not in ['sample_id', 'age', 'Tissue', 'project_id', 'pred_age_stage1']]
         log(f"Warning: No 'pred_resid_' columns found. Using fallback columns: {len(expert_cols)}")

    # [CRITICAL UPDATE] Explicitly remove 'Global_Union' from input features
    # This prevents the naive ensemble from Stage 2 being used as a feature in Stage 3
    cols_to_remove = ['pred_resid_Global_Union', 'pred_age_Global_Union']
    original_count = len(expert_cols)
    expert_cols = [c for c in expert_cols if c not in cols_to_remove and 'Global_Union' not in c]
    
    if len(expert_cols) < original_count:
        log(f"Excluded Global_Union column(s). Features reduced from {original_count} to {len(expert_cols)}.")
    
    # Merge Logic:
    cols_to_use_from_s2 = ['sample_id'] + expert_cols
    cols_to_use_from_s2 = [c for c in cols_to_use_from_s2 if c not in df_s1.columns or c == 'sample_id']
    
    df_merged = pd.merge(
        df_s1, 
        df_s2[cols_to_use_from_s2], 
        on="sample_id", 
        how="inner"
    )
    
    pc_cols = []
    if pc_path and os.path.exists(pc_path):
        log(f"Loading PCs from {pc_path}...")
        df_pc = pd.read_csv(pc_path)
        pc_cols = [c for c in df_pc.columns if c.startswith('RF_PC')]
        df_merged = pd.merge(df_merged, df_pc[['sample_id'] + pc_cols], on="sample_id", how="inner")
    
    # --- 2. Define Targets ---
    df_merged["target_residual"] = df_merged["age"] - df_merged["pred_age_stage1"]
    
    y = df_merged["target_residual"].values.astype(np.float32)
    groups = df_merged["project_id"]
    tissues = df_merged["Tissue"]
    
    # Helper: Feature Matrix
    def get_feature_matrix(df, use_pcs, use_dummies):
        selected_cols = expert_cols.copy()
        if use_pcs: selected_cols += pc_cols
        X = df[selected_cols].copy()
        if use_dummies:
            dummies = pd.get_dummies(df['Tissue'], prefix='Tissue')
            X = pd.concat([X, dummies], axis=1)
        return X

    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
    
    best_alpha = alpha
    best_use_pcs = True if len(pc_cols) > 0 else False
    best_use_dummies = False

    # --- 3. Optuna Optimization ---
    if n_trials > 0:
        log(f"Starting Stage 3 Optuna Search ({n_trials} trials)...")
        
        def objective(trial):
            curr_alpha = trial.suggest_float("alpha", 0.1, 1000.0, log=True)
            curr_use_pcs = trial.suggest_categorical("use_pcs", [True, False]) if pc_cols else False
            curr_use_dummies = trial.suggest_categorical("use_tissue_dummies", [True, False])
            
            X_curr = get_feature_matrix(df_merged, curr_use_pcs, curr_use_dummies)
            fold_scores = []
            
            for train_idx, val_idx in folds:
                X_train, X_val = X_curr.iloc[train_idx], X_curr.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = Ridge(alpha=curr_alpha, random_state=seed)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                fold_scores.append(mean_absolute_error(y_val, preds))
                
            return np.mean(fold_scores)

        # Storage Support (NFS-Safe Logic)
        if storage and study_name:
            if storage.startswith("sqlite") or storage.startswith("mysql"):
                log(f"Using RDB Storage: {storage}")
                storage_backend = storage
            else:
                log(f"Using NFS-Safe Journal Storage: {storage}")
                storage_backend = JournalStorage(JournalFileStorage(storage))

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_backend,
                load_if_exists=True,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=seed)
            )
        else:
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
            
        study.optimize(objective, n_trials=n_trials)
        
        log(f"Optuna Finished. Best Global MAE: {study.best_value:.4f}")
        best_alpha = study.best_params.get('alpha', alpha)
        best_use_pcs = study.best_params.get('use_pcs', False)
        best_use_dummies = study.best_params.get('use_tissue_dummies', False)
    
    # --- 4. Final Training ---
    log(f"Training Final Models with Alpha={best_alpha:.2f}...")
    
    X_final_df = get_feature_matrix(df_merged, best_use_pcs, best_use_dummies)
    feature_names = X_final_df.columns.tolist()
    X_final = X_final_df.values.astype(np.float32)
    
    models = {}
    
    # Global
    global_model = Ridge(alpha=best_alpha, random_state=seed)
    global_model.fit(X_final, y)
    models['Global'] = global_model
    
    # Tissue Specific
    unique_tissues = np.unique(tissues)
    for tissue in unique_tissues:
        mask = (tissues == tissue)
        if np.sum(mask) >= min_samples_for_tissue:
            tissue_model = Ridge(alpha=best_alpha, random_state=seed)
            tissue_model.fit(X_final[mask], y[mask])
            models[tissue] = tissue_model
            
    # --- 5. Save Artifacts (Cumulative) ---
    artifact_handler.save_fusion_model(models)
    artifact_handler.save_feature_names(feature_names)
    
    config = {
        "use_pcs": best_use_pcs,
        "use_tissue_dummies": best_use_dummies,
        "pc_cols": pc_cols if best_use_pcs else []
    }
    artifact_handler.save("config", config)
    
    # OOF Generation
    final_preds = np.zeros(len(y))
    for i, t in enumerate(tissues):
        model = models.get(t, models['Global'])
        final_preds[i] = model.predict(X_final[i:i+1])
        
    final_age = df_merged["pred_age_stage1"] + final_preds
    
    # Accumulate results into df_merged
    df_merged["stage3_correction"] = final_preds
    df_merged["HeteroAge"] = final_age
    
    # Organize columns nicely
    first_cols = ["sample_id", "age", "HeteroAge", "pred_age_stage1", "stage3_correction", "Tissue", "project_id"]
    s2_cols = [c for c in df_merged.columns if c.startswith("pred_resid_") or c.startswith("pred_age_")]
    # Make sure we don't accidentally include the excluded union column in the output list if it exists in source
    # (Though we might want to keep it in the CSV for comparison, just not used in model)
    
    other_cols = [c for c in df_merged.columns if c not in first_cols + s2_cols]
    
    final_cols = first_cols + s2_cols + other_cols
    final_cols = [c for c in final_cols if c in df_merged.columns]
    
    res_df = df_merged[final_cols]
    res_df.to_csv(os.path.join(output_dir, "stage3_oof.csv"), index=False)
    
    log(f"Stage 3 Training Completed.")

def predict_stage3(artifact_dir, input_path, output_path):
    """
    Stage 3 Inference: Linear residual fusion to produce final HeteroAge.
    [Updated] Ensures Age and Tissue columns are preserved in output.
    """
    log(">>> [Stage 3] Running Residual Fusion Inference...")
    
    artifact_handler = Stage3Artifact(artifact_dir)
    models = artifact_handler.load_fusion_model()
    target_features = artifact_handler.load_feature_names()
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_pickle(input_path)
    
    # Robust Column Renaming
    if 'pred_age_stage1' not in df.columns:
        if 'Stage1_Pred' in df.columns:
            df.rename(columns={'Stage1_Pred': 'pred_age_stage1'}, inplace=True)
        elif 'pred_age' in df.columns:
            df.rename(columns={'pred_age': 'pred_age_stage1'}, inplace=True)
            
    base_age_col = "pred_age_stage1"
        
    # Reconstruct Feature Matrix for Context Fusion
    for f in target_features:
        if f not in df.columns:
            # Handle One-Hot Encoded Tissue columns
            if f.startswith("Tissue_"):
                tissue_name = f.replace("Tissue_", "")
                if "Tissue" in df.columns:
                    df[f] = (df["Tissue"] == tissue_name).astype(int)
                else:
                    df[f] = 0
            else:
                df[f] = 0.0

    X = df[target_features].values.astype(np.float32)
    tissues = df["Tissue"].values if "Tissue" in df.columns else np.array(["Global"] * len(df))
    
    s3_correction = np.zeros(len(df))
    unique_tissues = np.unique(tissues)
    
    # Apply Tissue-Specific Corrections
    for tissue in unique_tissues:
        mask = (tissues == tissue)
        model = models.get(tissue, models.get('Global'))
        if model:
            s3_correction[mask] = model.predict(X[mask])
            
    df["pred_residual_stage3"] = s3_correction
    df["HeteroAge"] = df[base_age_col] + df["pred_residual_stage3"]
    
    # --- [Key Update] Construct Final Output Columns ---
    desired_cols = [
        "sample_id", 
        "Age",                  # Metadata
        "Tissue",               # Metadata
        "HeteroAge",            # Final Result
        "pred_age_stage1",      # Intermediate
        "pred_residual_stage3", # Correction
        "project_id"
    ]
    
    # Include all Expert predictions (Hallmarks)
    expert_cols = [c for c in df.columns if c.startswith("pred_resid_") or c.startswith("pred_age_")]
    
    # Combine desired + experts + any other relevant columns
    final_output_cols = [c for c in desired_cols if c in df.columns] + \
                        [c for c in expert_cols if c not in desired_cols]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[final_output_cols].to_csv(output_path, index=False)
    
    del X, df, models
    gc.collect()
    log(f"  > Stage 3 results saved to {output_path} (Included Age & Tissue)")