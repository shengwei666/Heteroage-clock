"""
heteroage_clock.stages.stage1

Stage 1: Global Anchor
"""

import os
import json
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
    # Hyperparameters updated to support lists and parallel
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
    # New Intelligent Sampling Params
    min_cohorts: int = 2,
    min_cap: int = 30,
    max_cap: int = 600,
    median_mult: float = 1.0,
    **kwargs
) -> None:
    """
    Train the Stage 1 model.
    Accepts explicit file paths and all hyperparameters.
    Now supports parallel processing, hyperparameter lists, and intelligent down-sampling.
    """
    artifact_handler = Stage1Artifact(output_dir)
    
    # --- 1. Load Data ---
    log(f"Loading Hallmark Dictionary from {dict_path}...")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Hallmark dictionary not found at {dict_path}")
    
    # Fixed: Use json.load for robust dictionary loading
    with open(dict_path, 'r') as f:
        hallmark_dict = json.load(f)

    log(f"Loading Context (PCs) from {pc_path}...")
    pc_data = pd.read_csv(pc_path)
    
    log(f"Loading Omics Data...")
    cpg_beta = pd.read_pickle(beta_path)
    chalm_data = pd.read_pickle(chalm_path)
    camda_data = pd.read_pickle(camda_path)

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

    # --- 4. Training Strategy: Full Set ---
    log(f"Stage 1 Strategy: Training on FULL feature set ({len(feature_candidates)} features).")
    X_selected = X_full 
    final_features = feature_candidates

    # --- 5. Optimization (Macro + Micro) ---
    log("Optimizing Hyperparameters (Macro + Micro) with Intelligent Sampling...")
    # Updated to pass through new list, parallel, and sampling parameters
    best_model = tune_elasticnet_macro_micro(
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
        # Sampling params passed here
        min_cohorts=min_cohorts,
        min_cap=min_cap,
        max_cap=max_cap,
        median_mult=median_mult
    )

    # --- 6. Final OOF Generation (Asymmetric) ---
    log("Generating Final OOF (Asymmetric: Balanced Train, Full Val)...")
    folds = make_stratified_group_folds(groups=groups, tissues=tissues, n_splits=n_splits, seed=seed)
    oof_preds_linear = np.zeros(len(y))
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # --- A. Intelligent Down-sampling on Train Set ---
        # Reconstruct DataFrame for sampler
        train_tissues = tissues.iloc[train_idx].values if hasattr(tissues, 'iloc') else tissues[train_idx]
        train_groups = groups.iloc[train_idx].values if hasattr(groups, 'iloc') else groups[train_idx]
        
        df_train_meta = pd.DataFrame({
            'Tissue': train_tissues,
            'project_id': train_groups,
            'original_idx': train_idx 
        })
        
        # Apply adaptive sampler
        df_bal, _ = adaptive_sampler(
            df_train_meta, 
            min_coh=min_cohorts,
            min_c=min_cap,
            max_c=max_cap,
            mult=median_mult,
            seed=seed + fold_idx # Vary seed per fold
        )
        final_train_idx = df_bal['original_idx'].values
        
        # --- B. Train on Balanced Data ---
        model_fold = clone(best_model)
        model_fold.fit(X_selected[final_train_idx], y_trans[final_train_idx])
        
        # --- C. Predict on Full Validation Data ---
        val_preds_trans = model_fold.predict(X_selected[val_idx])
        val_preds_linear = trans.inverse_transform(val_preds_trans)
        oof_preds_linear[val_idx] = val_preds_linear

    oof_metrics = compute_regression_metrics(y, oof_preds_linear)
    log(f"Stage 1 Final OOF Metrics (Full Data): {oof_metrics}")

    # --- 7. Final Training (Balanced Full Set) ---
    # We want the final artifact to be fair and unbiased, so we balance the FULL dataset
    log("Retraining final Global Anchor model on Balanced Full Data...")
    
    df_full_meta = pd.DataFrame({
        'Tissue': tissues,
        'project_id': groups,
        'original_idx': np.arange(len(y))
    })
    
    df_full_bal, _ = adaptive_sampler(
        df_full_meta, 
        min_coh=min_cohorts,
        min_c=min_cap,
        max_c=max_cap,
        mult=median_mult,
        seed=seed
    )
    final_idx = df_full_bal['original_idx'].values
    
    final_model = clone(best_model)
    final_model.fit(X_selected[final_idx], y_trans[final_idx])

    # --- 8. Orthogonalization (Correlation Ranking) ---
    log("Generating Correlation-based Orthogonalized Dictionary for Stage 2...")
    
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
    
    # Note: Orthogonalization typically uses the full dataset correlations to be robust
    final_ortho_dict = orthogonalize_by_correlation(
        X=X_full, 
        y=y_trans, 
        feature_names=feature_candidates, 
        hallmark_mapping=hallmark_dict_suffixed
    )
    
    log(f"Orthogonalized Dictionary generated.")

    # --- Save Artifacts ---
    log("Saving artifacts...")
    artifact_handler.save_global_model(final_model)
    artifact_handler.save("stage1_features", final_features) 
    artifact_handler.save_orthogonalized_dict(final_ortho_dict)
    
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

# predict_stage1 logic remains same
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