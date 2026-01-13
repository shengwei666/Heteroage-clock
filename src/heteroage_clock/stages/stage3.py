"""
heteroage_clock.stages.stage3

Stage 3: Context-Aware Fusion (Ridge Regression Strategy).
WINNER MODEL: Best performance (MAE ~5.57).

REASONING:
- Ridge Regression handles the multicollinearity of biological hallmarks better than Lasso/ElasticNet.
- It preserves the "team effort" of multiple experts rather than selecting just one.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage3 import Stage3Artifact

def train_stage3(
    output_dir: str,
    stage1_oof_path: str,
    stage2_oof_path: str,
    pc_path: str,
    # --- Ridge Hyperparameters ---
    alphas: tuple = (0.1, 1.0, 10.0, 100.0, 500.0, 1000.0), # Extended range for robustness
    min_samples_for_tissue: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> None:
    """
    Train Stage 3 using Tissue-Specific Ridge Regression.
    """
    artifact_handler = Stage3Artifact(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Data Loading & Assembly
    log(">>> [Stage 3] Loading Artifacts (Strategy: Ridge Regression - The Winner)...")
    
    s1 = pd.read_csv(stage1_oof_path)
    s2 = pd.read_csv(stage2_oof_path)
    
    # Standardization
    if 'Stage1_Pred' in s1.columns and 'pred_age' not in s1.columns: s1.rename(columns={'Stage1_Pred': 'pred_age'}, inplace=True)
    if 'Age' in s1.columns and 'age' not in s1.columns: s1.rename(columns={'Age': 'age'}, inplace=True)
        
    expert_cols_age = [c for c in s2.columns if c.startswith("pred_age_") and c != 'pred_age']
    expert_cols_res = [c for c in s2.columns if c.startswith("pred_residual_")]
    expert_cols = expert_cols_age + expert_cols_res

    # Merge
    cols_to_merge_s2 = ["sample_id"] + expert_cols
    df = pd.merge(s1, s2[cols_to_merge_s2], on="sample_id", how="inner")
    
    if 'project_id_x' in df.columns: df.rename(columns={'project_id_x': 'project_id'}, inplace=True)
    if "Tissue" not in df.columns and "tissue" in df.columns: df.rename(columns={'tissue': 'Tissue'}, inplace=True)

    # 2. Feature Engineering
    df["target_residual"] = df["age"] - df["pred_age"]
    y = df["target_residual"].values
    
    input_features = []
    for col in expert_cols:
        if col.startswith("pred_age_"):
            feat_name = f"diff_{col.replace('pred_age_', '')}"
            df[feat_name] = df[col] - df["pred_age"]
            input_features.append(feat_name)
        else:
            input_features.append(col)
            
    X = df[input_features].values
    tissues = df["Tissue"].values
    
    log(f"Training Ridge Models on {len(input_features)} features.")

    # 3. Tissue-Specific Ridge Regression
    models = {}
    weights_log = []
    
    # A. Global Model (Fallback)
    log("Training Global Ridge Model...")
    global_model = RidgeCV(alphas=alphas, scoring='neg_mean_absolute_error')
    global_model.fit(X, y)
    models['Global'] = global_model
    
    # B. Tissue-Specific Models
    unique_tissues = np.unique(tissues)
    final_preds = np.zeros(len(y))
    
    log(f"Training specific models for {len(unique_tissues)} tissues...")
    
    for tissue in unique_tissues:
        mask = (tissues == tissue)
        n_samples = np.sum(mask)
        
        if n_samples >= min_samples_for_tissue:
            X_t = X[mask]
            y_t = y[mask]
            
            # RidgeCV automatically selects the best alpha (Regularization strength)
            model = RidgeCV(alphas=alphas, scoring='neg_mean_absolute_error')
            model.fit(X_t, y_t)
            
            final_preds[mask] = model.predict(X_t)
            models[tissue] = model
            
            # Log weights for analysis
            weights = {k: v for k, v in zip(input_features, model.coef_)}
            weights['Tissue'] = tissue
            weights['Intercept'] = model.intercept_
            weights['Alpha'] = model.alpha_
            weights['N_Samples'] = n_samples
            weights_log.append(weights)
        else:
            final_preds[mask] = global_model.predict(X[mask])

    # 4. Results
    final_pred_age = df["pred_age"] + final_preds
    
    metrics = compute_regression_metrics(df["age"].values, final_pred_age.values)
    log(f"📊 Stage 3 (Ridge) Final Results:")
    log(f"   > MAE: {metrics['MAE']:.4f}")
    
    s1_mae = np.mean(np.abs(df["age"] - df["pred_age"]))
    log(f"   (vs Stage 1 MAE: {s1_mae:.4f})")
    
    if weights_log:
        pd.DataFrame(weights_log).to_csv(os.path.join(output_dir, "Stage3_Ridge_Weights.csv"), index=False)
    
    # Save Outputs
    artifact_handler.save_fusion_model(models)
    artifact_handler.save("stage3_features", input_features)
    
    out_df = df[["sample_id", "age", "Tissue", "project_id"]].copy()
    out_df["Stage1_Pred"] = df["pred_age"]
    out_df["Stage3_Residual"] = final_preds
    out_df["HeteroAge"] = final_pred_age
    out_df["HeteroAge_Accel"] = out_df["HeteroAge"] - out_df["age"]
    
    artifact_handler.save_final_predictions(out_df)
    log(f"✅ Stage 3 Completed. Outputs saved to {output_dir}")

def predict_stage3(artifact_dir, input_path, output_path):
    # Inference logic
    artifact_handler = Stage3Artifact(artifact_dir)
    log(f"Loading Stage 3 Ridge models from {artifact_dir}...")
    models = artifact_handler.load_fusion_model()
    features = artifact_handler.load("stage3_features")
    
    if input_path.endswith('.csv'): df = pd.read_csv(input_path)
    else: df = pd.read_pickle(input_path)
        
    if "pred_age_stage1" in df.columns: df["pred_age"] = df["pred_age_stage1"]
    if "Stage1_Pred" in df.columns: df["pred_age"] = df["Stage1_Pred"]
    if "Tissue" in df.columns: pass
    elif "tissue" in df.columns: df.rename(columns={'tissue': 'Tissue'}, inplace=True)
    
    for feat in features:
        if feat.startswith("diff_"):
            hallmark_col = "pred_age_" + feat.replace("diff_", "")
            if hallmark_col in df.columns: df[feat] = df[hallmark_col] - df["pred_age"]
            else: df[feat] = 0 
        elif feat not in df.columns: df[feat] = 0
            
    X = df[features].values
    tissues = df["Tissue"].values
    final_preds_res = np.zeros(len(df))
    
    unique_tissues = np.unique(tissues)
    for tissue in unique_tissues:
        mask = (tissues == tissue)
        X_t = X[mask]
        model = models.get(tissue, models['Global'])
        final_preds_res[mask] = model.predict(X_t)
    
    df["HeteroAge"] = df["pred_age"] + final_preds_res
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"Final predictions saved to {output_path}")