"""
heteroage_clock.stages.stage1

Stage 1: Global Anchor

This stage establishes a global biological age baseline.
It uses 'Macro + Micro' optimization to find robust hyperparameters.
It saves metadata (project_id, Tissue) in OOF for downstream usage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.base import clone
from collections import defaultdict

from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.data.assemble import assemble_features, filter_and_impute
from heteroage_clock.core.age_transform import AgeTransformer
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.core.selection import select_features_internal
from heteroage_clock.core.optimization import tune_elasticnet_macro_micro
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage1 import Stage1Artifact

def train_stage1(project_root: str, output_dir: str, pc_path: str, dict_name: str, sweep_file: str = None) -> None:
    """
    Train the Stage 1 model using Custom 'Macro + Micro' Grid Search.
    """
    artifact_handler = Stage1Artifact(output_dir)
    
    # --- 1. Load Data ---
    log(f"Loading data from {project_root}...")
    feature_sets_dir = os.path.join(project_root, "4.Data_assembly/Feature_Sets")
    dict_path = os.path.join(feature_sets_dir, dict_name)
    
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Hallmark dictionary not found at {dict_path}")
        
    hallmark_dict = pd.read_json(dict_path)
    pc_data = pd.read_csv(pc_path)
    
    raw_aligned_dir = os.path.join(project_root, "4.Data_assembly/Raw_Aligned_RefGuided")
    cpg_beta = pd.read_pickle(os.path.join(raw_aligned_dir, "Beta_Train_RefGuided_NoCpH.pkl"))
    chalm_data = pd.read_pickle(os.path.join(raw_aligned_dir, "Chalm_Train_RefGuided_NoCpH.pkl"))
    camda_data = pd.read_pickle(os.path.join(raw_aligned_dir, "Camda_Train_RefGuided_NoCpH.pkl"))

    # --- 2. Assemble ---
    log("Assembling features...")
    assembled_data = assemble_features(cpg_beta, chalm_data, camda_data, pc_data, cpg_beta)
    assembled_data = filter_and_impute(assembled_data, features_to_keep=cpg_beta.columns.tolist())
    
    metadata_cols = ["sample_id", "age", "project_id", "Tissue", "Sex", "Is_Healthy", "Sex_encoded"]
    
    all_cols = assembled_data.columns.tolist()
    feature_candidates = [c for c in all_cols if c not in metadata_cols]
    
    X_full = assembled_data[feature_candidates].values
    y = assembled_data["age"].values
    groups = assembled_data["project_id"]
    tissues = assembled_data["Tissue"]
    sample_ids = assembled_data["sample_id"].values

    log(f"Data assembled. Shape: {X_full.shape}. Candidates: {len(feature_candidates)}")

    # --- 3. Transform Target ---
    trans = AgeTransformer(adult_age=20)
    y_trans = trans.transform(y)

    # --- 4. Feature Selection (Orthogonalization) ---
    log("Running Feature Orthogonalization...")
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

    selected_features = select_features_internal(X_full, y_trans, feature_candidates, hallmark_dict_suffixed)
    log(f"Selected {len(selected_features)} orthogonal features.")
    
    # Add Context features (PCs)
    pc_cols = [c for c in feature_candidates if c.startswith("RF_PC") or c == "Sex_encoded"]
    final_features = list(set(selected_features + pc_cols))
    
    # Re-build X
    final_indices = [feature_candidates.index(f) for f in final_features]
    X_selected = X_full[:, final_indices]
    
    log(f"Final Model Matrix: {X_selected.shape}")

    # --- 5. Optimization (Macro + Micro) ---
    log("Optimizing Hyperparameters (Macro + Micro)...")
    best_model = tune_elasticnet_macro_micro(
        X=X_selected, 
        y=y_trans, 
        groups=groups, 
        tissues=tissues, 
        trans_func=trans
    )

    # --- 6. Final OOF Generation ---
    log("Generating Final OOF with Best Model...")
    
    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=5, seed=42)
    
    if not folds:
        raise ValueError("Failed to create folds. Check data sufficiency.")

    oof_preds_linear = np.zeros(len(y))
    
    for train_idx, val_idx in folds:
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train_trans = y_trans[train_idx]
        
        model_fold = clone(best_model)
        model_fold.fit(X_train, y_train_trans)
        
        val_preds_trans = model_fold.predict(X_val)
        val_preds_linear = trans.inverse_transform(val_preds_trans)
        oof_preds_linear[val_idx] = val_preds_linear

    # Evaluate
    oof_metrics = compute_regression_metrics(y, oof_preds_linear)
    log(f"Stage 1 Final OOF Metrics: {oof_metrics}")

    # --- 7. Final Training & Saving ---
    log("Retraining final Global Anchor model...")
    final_model = clone(best_model)
    final_model.fit(X_selected, y_trans)

    log("Saving artifacts...")
    artifact_handler.save_global_model(final_model)
    artifact_handler.save("stage1_features", final_features) 
    
    # Save Ortho Dict
    final_ortho_dict = defaultdict(list)
    for f in selected_features:
        for h, feats in hallmark_dict_suffixed.items():
            if f in feats:
                final_ortho_dict[h].append(f)
                break
    artifact_handler.save_orthogonalized_dict(final_ortho_dict)
    
    # [CRITICAL] Save OOF with Metadata for downstream usage
    oof_df = pd.DataFrame({
        "sample_id": sample_ids,
        "age": y,
        "pred_age": oof_preds_linear,
        "residual": y - oof_preds_linear,
        "project_id": groups.values,
        "Tissue": tissues.values
    })
    
    if "Sex" in assembled_data.columns:
        oof_df["Sex"] = assembled_data["Sex"].values
    if "Is_Healthy" in assembled_data.columns:
        oof_df["Is_Healthy"] = assembled_data["Is_Healthy"].values
        
    artifact_handler.save_oof_predictions(oof_df)

    log(f"Stage 1 completed. Outputs in {output_dir}")


def predict_stage1(artifact_dir: str, input_path: str, output_path: str) -> None:
    """
    Predict using Stage 1 artifacts.
    """
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
        raise ValueError(f"Input missing {len(missing)} features.")
    
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