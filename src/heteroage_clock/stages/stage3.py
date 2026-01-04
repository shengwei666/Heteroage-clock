"""
heteroage_clock.stages.stage3

Stage 3: Context-Aware Fusion (LightGBM).
Fusion model that combines Stage 1 (Anchor), Stage 2 (Experts), and Context (PCs/Tissue).
Updates: Added **kwargs to robustly handle pipeline arguments.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage3 import Stage3Artifact

def train_stage3(
    output_dir: str,
    stage1_oof_path: str,
    stage2_oof_path: str,
    pc_path: str,
    n_estimators: int = 2000,
    learning_rate: float = 0.01,
    num_leaves: int = 31,
    max_depth: int = -1,
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
    **kwargs  # <--- [Critical] Absorbs unused args passed by pipeline (e.g. min_cap)
) -> None:
    """
    Train Stage 3 Fusion Model using LightGBM.
    """
    artifact_handler = Stage3Artifact(output_dir)
    
    # --- 1. Load Data ---
    log("Loading Stage 1 & Stage 2 OOFs...")
    s1 = pd.read_csv(stage1_oof_path)
    s2 = pd.read_csv(stage2_oof_path)
    pcs = pd.read_csv(pc_path)
    
    # Merge Stage 1 and Stage 2
    # s1 has [sample_id, age, pred_age, residual, project_id, Tissue]
    # s2 has [sample_id, pred_residual_Inflammation, ...]
    
    df = pd.merge(s1, s2, on="sample_id", how="inner")
    
    # Merge PCs (Context)
    # Check for duplicate columns to avoid merge errors
    cols_to_use = [c for c in pcs.columns if c not in df.columns or c == 'sample_id']
    df = pd.merge(df, pcs[cols_to_use], on="sample_id", how="inner")
    
    # --- 2. Prepare Features ---
    # Features: Stage 1 Prediction + Stage 2 Corrections + PCs + Tissue(Encoded)
    
    # Identify Stage 2 columns (residual predictions)
    s2_cols = [c for c in df.columns if c.startswith("pred_residual_")]
    # Identify PC columns
    pc_cols = [c for c in df.columns if c.startswith("RF_PC")]
    
    feature_cols = ["pred_age"] + s2_cols + pc_cols
    
    # Handle Tissue Encoding (Categorical) for LightGBM
    if "Tissue" in df.columns:
        df["Tissue"] = df["Tissue"].astype("category")
        feature_cols.append("Tissue")
        
    X = df[feature_cols]
    y = df["age"] # True Age
    
    # --- 3. LightGBM Training ---
    log(f"Training Fusion Model on {len(X)} samples with {len(feature_cols)} features...")
    
    params = {
        "objective": "regression",
        "metric": "l1",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "n_jobs": n_jobs,
        "random_state": seed,
        "verbose": -1
    }
    
    # Cross-Validation for OOF generation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))
    
    # LightGBM handles categorical features automatically if col type is 'category'
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)], 
            eval_metric="l1", 
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_va)
        
    metrics = compute_regression_metrics(y, oof_preds)
    log(f"Stage 3 OOF Metrics: {metrics}")
    
    # Final Fit on Full Data
    log("Training Final Stage 3 Model...")
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)
    
    # --- 4. Save Artifacts ---
    artifact_handler.save_fusion_model(final_model)
    artifact_handler.save("stage3_features", feature_cols)
    
    # Save Final Predictions (for analysis)
    out_df = df[["sample_id", "age", "Tissue", "project_id"]].copy()
    out_df["HeteroAge"] = oof_preds
    out_df["HeteroAge_Accel"] = out_df["HeteroAge"] - out_df["age"]
    
    artifact_handler.save_final_predictions(out_df)
    log(f"Stage 3 Completed. Outputs in {output_dir}")

def predict_stage3(artifact_dir, input_path, output_path):
    """
    Inference for Stage 3.
    """
    artifact_handler = Stage3Artifact(artifact_dir)
    log(f"Loading Stage 3 model from {artifact_dir}...")
    
    model = artifact_handler.load_fusion_model()
    features = artifact_handler.load("stage3_features")
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_pickle(input_path)
    
    # Standardize column names
    # pipeline.py might save as 'pred_age_stage1', but model expects 'pred_age'
    if "pred_age_stage1" in df.columns and "pred_age" not in df.columns:
        df["pred_age"] = df["pred_age_stage1"]
        
    # Ensure Tissue is category
    if "Tissue" in df.columns:
        df["Tissue"] = df["Tissue"].astype("category")
        
    # Check for missing features (e.g. PCs) and fill 0 if needed
    missing = [c for c in features if c not in df.columns]
    if missing:
        log(f"Warning: Missing features for Stage 3: {missing}")
        for c in missing: df[c] = 0
        
    X = df[features]
    preds = model.predict(X)
    
    df["HeteroAge"] = preds
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"Final predictions saved to {output_path}")