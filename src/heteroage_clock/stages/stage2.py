"""
heteroage_clock.stages.stage2

Stage 2: Hallmark Experts
Uses 'Macro + Micro' optimization to ensure robust residual correction across tissues.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.base import clone

from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.data.assemble import assemble_features
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.optimization import tune_elasticnet_macro_micro
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage2 import Stage2Artifact

def train_stage2(project_root: str, output_dir: str, stage1_oof_path: str, stage1_dict_path: str, pc_path: str) -> None:
    artifact_handler = Stage2Artifact(output_dir)
    
    # --- 1. Load Data ---
    log(f"Loading Stage 1 OOF from {stage1_oof_path}...")
    stage1_oof = pd.read_csv(stage1_oof_path)
    if "residual" not in stage1_oof.columns: raise ValueError("Missing 'residual' in Stage 1 OOF")

    log(f"Loading Dictionary from {stage1_dict_path}...")
    if not os.path.exists(stage1_dict_path): raise FileNotFoundError(stage1_dict_path)
    hallmark_dict = pd.read_json(stage1_dict_path)
    
    log("Loading features...")
    raw_dir = os.path.join(project_root, "4.Data_assembly/Raw_Aligned_RefGuided")
    pc_data = pd.read_csv(pc_path)
    cpg_beta = pd.read_pickle(os.path.join(raw_dir, "Beta_Train_RefGuided_NoCpH.pkl"))
    chalm_data = pd.read_pickle(os.path.join(raw_dir, "Chalm_Train_RefGuided_NoCpH.pkl"))
    camda_data = pd.read_pickle(os.path.join(raw_dir, "Camda_Train_RefGuided_NoCpH.pkl"))

    # --- 2. Assemble ---
    assembled_data = assemble_features(cpg_beta, chalm_data, camda_data, pc_data, cpg_beta)
    
    # Merge and keep metadata for Splitting
    # Note: Stage 1 OOF now contains project_id/Tissue, so we can use them!
    train_df = pd.merge(assembled_data, stage1_oof, on="sample_id", how="inner", suffixes=("", "_oof"))
    
    # Ensure metadata availability
    if "project_id" in train_df.columns:
        groups = train_df["project_id"]
    elif "project_id_oof" in train_df.columns:
        groups = train_df["project_id_oof"]
    else:
        raise ValueError("Missing project_id for splitting")

    if "Tissue" in train_df.columns:
        tissues = train_df["Tissue"]
    elif "Tissue_oof" in train_df.columns:
        tissues = train_df["Tissue_oof"]
    else:
        raise ValueError("Missing Tissue for splitting")

    y_global = train_df["residual"].values
    
    # --- 3. Split ---
    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=5, seed=42)
    stage2_oof_df = train_df[["sample_id"]].copy()
    
    # --- 4. Train Hallmark Experts ---
    for hallmark, cpg_list in hallmark_dict.items():
        log(f"Processing Hallmark: {hallmark}...")
        
        # Select features
        relevant_cols = []
        for cpg in cpg_list:
            relevant_cols.extend([c for c in train_df.columns if c.startswith(f"{cpg}_")])
        
        if not relevant_cols:
            continue
            
        X = train_df[relevant_cols].values
        
        # --- Optimization (Macro + Micro) ---
        # Stage 2 predicts Residuals (Linear). No trans_func needed.
        best_model = tune_elasticnet_macro_micro(
            X=X, 
            y=y_global, 
            groups=groups, 
            tissues=tissues, 
            trans_func=None
        )
        
        # --- OOF Generation ---
        hallmark_oof = np.zeros(len(y_global))
        for train_idx, val_idx in folds:
            m = clone(best_model)
            m.fit(X[train_idx], y_global[train_idx])
            hallmark_oof[val_idx] = m.predict(X[val_idx])
            
        stage2_oof_df[f"pred_residual_{hallmark}"] = hallmark_oof
        
        metrics = compute_regression_metrics(y_global, hallmark_oof)
        log(f"  > Metrics: {metrics}")
        
        # --- Final Save ---
        final_model = clone(best_model)
        final_model.fit(X, y_global)
        
        artifact_handler.save_expert_model(hallmark, final_model)
        artifact_handler.save(f"stage2_{hallmark}_features", relevant_cols)

    artifact_handler.save_oof_corrections(stage2_oof_df)
    log(f"Stage 2 Completed.")


def predict_stage2(artifact_dir: str, input_path: str, output_path: str) -> None:
    """
    Predict using all trained Stage 2 Hallmark Expert models.
    """
    artifact_handler = Stage2Artifact(artifact_dir)
    
    log(f"Loading input data from {input_path}...")
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_pickle(input_path)
        
    output_df = pd.DataFrame()
    if "sample_id" in data.columns:
        output_df["sample_id"] = data["sample_id"]
        
    # Find Models
    model_files = glob.glob(os.path.join(artifact_dir, "stage2_*_expert_model.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No Stage 2 models found in {artifact_dir}")
        
    log(f"Found {len(model_files)} hallmark models.")
    
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
            for c in missing:
                data[c] = 0.0
        
        X = data[feature_cols].values
        preds = model.predict(X)
        
        output_df[f"pred_residual_{hallmark_name}"] = preds
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log(f"Stage 2 predictions saved to {output_path}")