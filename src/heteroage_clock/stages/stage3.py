"""
heteroage_clock.stages.stage3

Stage 3: Context-Aware Fusion
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.base import clone
from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.core.age_transform import AgeTransformer
from heteroage_clock.core.splits import make_stratified_group_folds
from heteroage_clock.utils.logging import log
from heteroage_clock.artifacts.stage3 import Stage3Artifact

def train_stage3(
    output_dir: str, 
    stage1_oof_path: str, 
    stage2_oof_path: str, 
    pc_path: str,
    # Hyperparameters
    n_estimators: int = 2000,
    learning_rate: float = 0.01,
    num_leaves: int = 31,
    max_depth: int = -1,
    n_splits: int = 5,
    seed: int = 42
) -> None:
    """
    Train Stage 3 Meta-Learner.
    Accepts explicit paths and hyperparameters.
    """
    artifact_handler = Stage3Artifact(output_dir)
    
    # 1. Load Data
    log(f"Loading Stage 1 OOF from {stage1_oof_path}...")
    stage1_oof = pd.read_csv(stage1_oof_path)

    log(f"Loading Stage 2 OOF from {stage2_oof_path}...")
    stage2_oof = pd.read_csv(stage2_oof_path)

    log(f"Loading PCs from {pc_path}...")
    pc_data = pd.read_csv(pc_path)

    # 2. Merge Data
    merged_data = pd.merge(stage1_oof, stage2_oof, on="sample_id", how="inner")
    merged_data = pd.merge(merged_data, pc_data, on="sample_id", how="inner")
    
    if "Sex_encoded" not in merged_data.columns and "Sex" in merged_data.columns:
        merged_data["Sex_encoded"] = merged_data["Sex"].map({"F": 0, "M": 1, "Female": 0, "Male": 1}).fillna(0)
    
    # 3. Define Features
    feature_cols = []
    if "pred_age" in merged_data.columns:
        feature_cols.append("pred_age")
    
    hallmark_cols = [c for c in merged_data.columns if c.startswith("pred_residual_")]
    feature_cols.extend(hallmark_cols)
    
    pc_cols = [c for c in merged_data.columns if c.startswith("RF_PC")]
    feature_cols.extend(pc_cols)
    
    if "Sex_encoded" in merged_data.columns:
        feature_cols.append("Sex_encoded")
    
    log(f"Fusion Features ({len(feature_cols)}): Hallmarks={len(hallmark_cols)}, Context={len(pc_cols)}")

    X = merged_data[feature_cols]
    y = merged_data["age"]
    sample_ids = merged_data["sample_id"]
    
    # 4. Transform Target (Log-Linear)
    trans = AgeTransformer(adult_age=20)
    y_trans = trans.transform(y.values)

    # 5. Stratified Group K-Fold CV
    if "project_id" in merged_data.columns and "Tissue" in merged_data.columns:
        groups = merged_data["project_id"]
        tissues = merged_data["Tissue"]
        folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
    else:
        log("Warning: 'project_id' or 'Tissue' missing. Falling back to simple KFold.")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(kf.split(X, y_trans))

    oof_preds_linear = np.zeros(len(y))
    
    # Construct LightGBM params from arguments
    model_params = {
        'random_state': seed,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_trans = y_trans[train_idx]
        
        model_fold = lgb.LGBMRegressor(**model_params)
        model_fold.fit(X_train, y_train_trans)
        
        val_preds_trans = model_fold.predict(X_val)
        val_preds_linear = trans.inverse_transform(val_preds_trans)
        oof_preds_linear[val_idx] = val_preds_linear

    metrics = compute_regression_metrics(y, oof_preds_linear)
    log(f"Stage 3 OOF Metrics: {metrics}")

    # 6. Retrain Final Model
    log("Retraining final Meta-Learner...")
    final_model = lgb.LGBMRegressor(**model_params)
    final_model.fit(X, y_trans)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    artifact_handler.save_meta_learner_model(final_model)
    artifact_handler.save("stage3_features", feature_cols)
    artifact_handler.save_attention_importance(importance_df)
    
    oof_df = pd.DataFrame({
        "sample_id": sample_ids,
        "age": y,
        "final_pred_age": oof_preds_linear,
        "final_residual": y - oof_preds_linear
    })
    artifact_handler.save_oof_predictions(oof_df)
    log(f"Stage 3 completed.")

# predict_stage3 logic remains same
def predict_stage3(artifact_dir: str, input_path: str, output_path: str) -> None:
    artifact_handler = Stage3Artifact(artifact_dir)
    log(f"Loading Stage 3 model from {artifact_dir}...")
    model = artifact_handler.load_meta_learner_model()
    feature_cols = artifact_handler.load("stage3_features")
    
    log(f"Loading merged input data from {input_path}...")
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    else:
        data = pd.read_pickle(input_path)

    if "Sex_encoded" in feature_cols and "Sex_encoded" not in data.columns:
        if "Sex" in data.columns:
             data["Sex_encoded"] = data["Sex"].map({"F": 0, "M": 1, "Female": 0, "Male": 1}).fillna(0)
        else:
             data["Sex_encoded"] = 0
        
    if "pred_age_stage1" in data.columns and "pred_age" not in data.columns:
        data["pred_age"] = data["pred_age_stage1"]

    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        for c in missing: data[c] = 0.0
            
    X = data[feature_cols]
    preds_trans = model.predict(X)
    
    trans = AgeTransformer(adult_age=20)
    preds_linear = trans.inverse_transform(preds_trans)
    
    output_df = pd.DataFrame()
    if "sample_id" in data.columns:
        output_df["sample_id"] = data["sample_id"]
        
    output_df["final_pred_age"] = preds_linear
    
    if "age" in data.columns:
        output_df["age"] = data["age"]
        output_df["final_residual"] = output_df["age"] - output_df["final_pred_age"]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log(f"Stage 3 Final Predictions saved to {output_path}")