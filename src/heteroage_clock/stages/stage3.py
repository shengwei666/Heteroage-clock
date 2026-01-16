"""
heteroage_clock.stages.stage3

Stage 3: Context-Aware Fusion.
Simplified version: Only uses Stage 2 residuals to predict Stage 1 error.
Final Age = Stage 1 Prediction + Stage 3 Residual Correction.
"""

import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from heteroage_clock.artifacts.stage3 import Stage3Artifact
from heteroage_clock.utils.logging import log

def train_stage3(
    output_dir: str,
    stage1_oof_path: str,
    stage2_oof_path: str,
    alphas: tuple = (0.1, 1.0, 10.0, 100.0, 500.0, 1000.0),
    min_samples_for_tissue: int = 50,
    **kwargs
) -> None:
    """
    Train Stage 3 fusion models using expert residuals to correct Stage 1 error.
    """
    artifact_handler = Stage3Artifact(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    log(">>> [Stage 3 Training] Training simplified residual fusion models...")
    
    # Load Stage 1 and Stage 2 results
    df_s1 = pd.read_csv(stage1_oof_path)
    df_s2 = pd.read_csv(stage2_oof_path)
    
    # Ensure column names are standardized
    if 'pred_age_stage1' not in df_s1.columns and 'Stage1_Pred' in df_s1.columns:
        df_s1.rename(columns={'Stage1_Pred': 'pred_age_stage1'}, inplace=True)
        
    # Expert columns are the features for Stage 3
    expert_cols = [c for c in df_s2.columns if c.startswith("pred_residual_")]
    
    # Merge based on sample_id
    df = pd.merge(
        df_s1[['sample_id', 'age', 'pred_age_stage1', 'Tissue']], 
        df_s2[['sample_id'] + expert_cols], 
        on="sample_id"
    )
    
    # Target is the residual: True Age - Stage 1 Prediction
    df["target_residual"] = df["age"] - df["pred_age_stage1"]
    
    X = df[expert_cols].values.astype(np.float32)
    y = df["target_residual"].values.astype(np.float32)
    tissues = df["Tissue"].values
    
    models = {}
    
    # 1. Global Model (Fallback for small tissue samples)
    global_model = RidgeCV(alphas=alphas, scoring='neg_mean_absolute_error')
    global_model.fit(X, y)
    models['Global'] = global_model
    
    # 2. Tissue-Specific Models
    unique_tissues = np.unique(tissues)
    for tissue in unique_tissues:
        mask = (tissues == tissue)
        if np.sum(mask) >= min_samples_for_tissue:
            model = RidgeCV(alphas=alphas, scoring='neg_mean_absolute_error')
            model.fit(X[mask], y[mask])
            models[tissue] = model
            
    # Save Model Dictionary and Feature List
    artifact_handler.save_fusion_model(models)
    artifact_handler.save("stage3_features", expert_cols)
    log(f"Stage 3 training completed. Features used: {len(expert_cols)}")

def predict_stage3(artifact_dir, input_path, output_path):
    """
    Stage 3 Inference: Linear residual fusion to produce final HeteroAge.
    Calculates both the stage 3 correction and the final age.
    """
    log(">>> [Stage 3] Running Residual Fusion Inference...")
    
    # 1. Load Artifacts
    artifact_handler = Stage3Artifact(artifact_dir)
    fusion_models = artifact_handler.load_fusion_model()
    target_features = artifact_handler.load("stage3_features")
    
    # 2. Load Consolidated Input Data (Pickle from pipeline)
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_pickle(input_path)
        
    # Standardize base age column
    if "pred_age_stage1" in df.columns:
        base_age_col = "pred_age_stage1"
    elif "Stage1_Pred" in df.columns:
        base_age_col = "Stage1_Pred"
    else:
        raise KeyError("Stage 1 prediction column not found in input.")
    
    # Pad missing expert residuals with 0 if necessary
    for f in target_features:
        if f not in df.columns:
            df[f] = 0.0

    # 3. Perform Prediction for Stage 3 Correction
    X = df[target_features].values.astype(np.float32)
    tissues = df["Tissue"].values if "Tissue" in df.columns else np.array(["Global"] * len(df))
    
    # This is the "pred_residual_stage3"
    s3_correction = np.zeros(len(df))
    
    unique_tissues = np.unique(tissues)
    for tissue in unique_tissues:
        mask = (tissues == tissue)
        # Apply specific model or fallback to global
        model = fusion_models.get(tissue, fusion_models.get('Global'))
        if model:
            s3_correction[mask] = model.predict(X[mask])

    # 4. Integrate Results
    df["pred_residual_stage3"] = s3_correction
    df["HeteroAge"] = df[base_age_col] + df["pred_residual_stage3"]
    
    # Define columns for final output
    # Includes ID, S3 correction, and final result
    output_cols = ["sample_id", "pred_residual_stage3", "HeteroAge"]
    
    # Ensure intermediate stage 1 data is preserved if requested by pipeline
    if base_age_col in df.columns:
        output_cols.append(base_age_col)

    # 5. Export Results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[output_cols].to_csv(output_path, index=False)
    
    # 6. Memory Cleanup
    del X, df, fusion_models, target_features, s3_correction
    gc.collect()
    log(f"   > Stage 3 inference complete. Saved to: {os.path.basename(output_path)}")