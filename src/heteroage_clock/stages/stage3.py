"""
heteroage_clock.stages.stage3

Stage 3: Context-Aware Fusion (LightGBM).
Fusion model that combines Stage 1 (Anchor), Stage 2 (Experts), and Context (PCs/Tissue).

UPDATES:
1. [Fix] Robust merge strategy to handle overlapping metadata in Stage 2 output.
2. [Feature] Comprehensive Optuna Hyperparameter Tuning.
3. [Feature] GroupKFold Cross-Validation to prevent leakage.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage3 import Stage3Artifact

def train_stage3(
    output_dir: str,
    stage1_oof_path: str,
    stage2_oof_path: str,
    pc_path: str,
    # --- Default Hyperparameters (Used if n_trials=0) ---
    n_estimators: int = 3000,
    learning_rate: float = 0.01,
    num_leaves: int = 31,
    max_depth: int = -1,
    # --- Pipeline Control ---
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
    n_trials: int = 0,  # Set >0 to enable Optuna
    **kwargs  # Absorbs unused arguments
) -> None:
    """
    Train Stage 3 Fusion Model using LightGBM with robust Hyperparameter Tuning.
    """
    artifact_handler = Stage3Artifact(output_dir)
    
    # ==========================================================================
    # 1. Data Loading & Assembly (Robust Merge Fix)
    # ==========================================================================
    log(">>> [Stage 3] Loading Artifacts for Context Fusion...")
    
    # Load inputs
    s1 = pd.read_csv(stage1_oof_path)
    s2 = pd.read_csv(stage2_oof_path)
    pcs = pd.read_csv(pc_path)
    
    # --- [CRITICAL FIX] Smart Merge Strategy ---
    # Stage 2 output now contains metadata (Age, Tissue, project_id) which duplicates Stage 1.
    # We must filter s2 to ONLY include 'sample_id' and expert predictions before merging.
    
    # 1. Identify Expert Columns
    expert_cols = [c for c in s2.columns if c.startswith("pred_residual_")]
    cols_to_merge_s2 = ["sample_id"] + expert_cols
    
    # Safety: Filter s2 to avoid 'project_id_x' / 'project_id_y' issues
    s2_clean = s2[cols_to_merge_s2].copy()
    
    # 2. Merge Stage 1 (Anchor) + Stage 2 (Experts)
    # Use inner join to ensure alignment
    df = pd.merge(s1, s2_clean, on="sample_id", how="inner")
    
    # 3. Merge Context (PCs)
    # Filter PCs to avoid duplication, keep 'RF_PC' columns
    cols_to_use_pc = [c for c in pcs.columns if c not in df.columns or c == 'sample_id']
    df = pd.merge(df, pcs[cols_to_use_pc], on="sample_id", how="inner")
    
    # --- Column Standardization ---
    # Handle common case-sensitivity issues
    if 'Age' in df.columns and 'age' not in df.columns:
        df.rename(columns={'Age': 'age'}, inplace=True)
        
    # Ensure project_id exists (critical for GroupKFold)
    if 'project_id' not in df.columns:
        # Fallback: check for merged artifacts
        if 'project_id_x' in df.columns:
            df.rename(columns={'project_id_x': 'project_id'}, inplace=True)
        else:
            raise KeyError("Critical: 'project_id' column missing. Cannot perform GroupKFold.")
    
    # Define Features
    s2_features = [c for c in df.columns if c.startswith("pred_residual_")]
    pc_features = [c for c in df.columns if c.startswith("RF_PC")]
    
    feature_cols = ["pred_age"] + s2_features + pc_features
    
    # Handle Tissue (Categorical)
    if "Tissue" in df.columns:
        df["Tissue"] = df["Tissue"].astype("category")
        feature_cols.append("Tissue")
    elif "tissue" in df.columns:
        df["Tissue"] = df["tissue"].astype("category")
        feature_cols.append("Tissue")
    
    # Prepare X, y, groups
    X = df[feature_cols]
    y = df["age"]
    groups = df["project_id"]
    
    log(f"Data Assembly Complete: {len(X)} samples, {len(feature_cols)} features.")
    
    # ==========================================================================
    # 2. Hyperparameter Optimization (Optuna)
    # ==========================================================================
    # Base parameters
    best_params = {
        "objective": "regression",
        "metric": "l1",
        "verbosity": -1,
        "n_jobs": n_jobs,
        "random_state": seed,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth
    }

    if n_trials > 0:
        log(f"🚀 Starting Comprehensive Optuna Optimization ({n_trials} trials)...")
        
        def objective(trial):
            # Define Search Space
            param_grid = {
                "objective": "regression",
                "metric": "l1",
                "verbosity": -1,
                "n_jobs": n_jobs,
                "random_state": seed,
                "n_estimators": 1000, # Faster search
                
                # Structure
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                
                # Learning
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                
                # Regularization
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
                
                # Sampling
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            }
            
            scores = []
            gkf = GroupKFold(n_splits=3) # Fast CV
            
            for train_idx, val_idx in gkf.split(X, y, groups=groups):
                model = lgb.LGBMRegressor(**param_grid)
                model.fit(
                    X.iloc[train_idx], y.iloc[train_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                    eval_metric="l1",
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                preds = model.predict(X.iloc[val_idx])
                scores.append(np.mean(np.abs(y.iloc[val_idx] - preds)))
            
            return np.mean(scores)

        # Run Optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        log(f"🏆 Best Optuna Params: {study.best_params}")
        best_params.update(study.best_params)
        
        # Robust Finetuning
        final_lr = best_params["learning_rate"] * 0.5
        final_est = 3000
        best_params["learning_rate"] = final_lr
        best_params["n_estimators"] = final_est
        log(f"Applying Robust Strategy: LR adjusted to {final_lr:.5f}, Estimators to {final_est}")

    # ==========================================================================
    # 3. Final Cross-Validation (5-Fold GroupKFold)
    # ==========================================================================
    log(f"Starting Final Evaluation (5-Fold GroupKFold)...")
    
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])
        
    metrics = compute_regression_metrics(y, oof_preds)
    log(f"📊 Stage 3 Final OOF Metrics: {metrics}")
    
    # ==========================================================================
    # 4. Production Training & Saving
    # ==========================================================================
    log("Training Final Production Model on Full Dataset...")
    
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y)
    
    # Save Artifacts
    artifact_handler.save_fusion_model(final_model)
    artifact_handler.save("stage3_features", feature_cols)
    
    # Save Prediction Results
    out_df = df[["sample_id", "age", "Tissue", "project_id"]].copy()
    out_df["HeteroAge"] = oof_preds
    out_df["HeteroAge_Accel"] = out_df["HeteroAge"] - out_df["age"]
    out_df["Stage1_Pred"] = df["pred_age"]
    
    artifact_handler.save_final_predictions(out_df)
    log(f"✅ Stage 3 Completed Successfully. Outputs saved to {output_dir}")

def predict_stage3(artifact_dir, input_path, output_path):
    """
    Inference for Stage 3 using the saved Fusion Model.
    """
    artifact_handler = Stage3Artifact(artifact_dir)
    log(f"Loading Stage 3 model from {artifact_dir}...")
    
    model = artifact_handler.load_fusion_model()
    features = artifact_handler.load("stage3_features")
    
    # Load Input
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_pickle(input_path)
    
    # 1. Standardize
    if "pred_age_stage1" in df.columns and "pred_age" not in df.columns:
        df["pred_age"] = df["pred_age_stage1"]
    
    # 2. Handle Tissue
    if "Tissue" in df.columns:
        df["Tissue"] = df["Tissue"].astype("category")
    elif "tissue" in df.columns:
        df["Tissue"] = df["tissue"].astype("category")
        
    # 3. Handle Missing Context
    missing = [c for c in features if c not in df.columns]
    if missing:
        log(f"⚠️ Warning: Missing features for Stage 3 inference: {len(missing)} cols.")
        for c in missing: 
            df[c] = 0
            
    # 4. Predict
    X = df[features]
    preds = model.predict(X)
    
    df["HeteroAge"] = preds
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"Final predictions saved to {output_path}")