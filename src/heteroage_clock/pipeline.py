"""
heteroage_clock.pipeline

High-level orchestration for training and inference.
Updates: Explicitly handles Optuna parameters (n_trials) for Stage 3.
"""

import os
import pandas as pd
from heteroage_clock.utils.logging import log

# Import core implementation
from heteroage_clock.stages.stage1 import train_stage1 as _train_stage1, predict_stage1 as _predict_stage1
from heteroage_clock.stages.stage2 import train_stage2 as _train_stage2, predict_stage2 as _predict_stage2
from heteroage_clock.stages.stage3 import train_stage3 as _train_stage3, predict_stage3 as _predict_stage3


# ==============================================================================
# Training Pipeline Wrappers
# ==============================================================================

def train_stage1(
    output_dir, pc_path, dict_path, beta_path, chalm_path, camda_path, 
    sweep_file=None, alpha_start=-4.0, alpha_end=-0.5, n_alphas=30, 
    l1_ratio=0.5, alphas=None, l1_ratios=None, n_splits=5, seed=42, 
    max_iter=2000, n_jobs=-1, project_root=None,
    **kwargs 
):
    """
    Wrapper for Stage 1 Training: Global Anchor.
    """
    log("=== Pipeline: Starting Stage 1 Training (Global Anchor) ===")
    _train_stage1(
        output_dir=output_dir,
        pc_path=pc_path,
        dict_path=dict_path,
        beta_path=beta_path,
        chalm_path=chalm_path,
        camda_path=camda_path,
        sweep_file=sweep_file,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        n_alphas=n_alphas,
        l1_ratio=l1_ratio,
        alphas=alphas,
        l1_ratios=l1_ratios,
        n_splits=n_splits,
        seed=seed,
        max_iter=max_iter,
        n_jobs=n_jobs,
        **kwargs
    )
    log("=== Pipeline: Stage 1 Training Completed ===\n")


def train_stage2(
    output_dir, stage1_oof, stage1_dict, pc_path, beta_path, chalm_path, camda_path,
    alpha_start=-4.0, alpha_end=-0.5, n_alphas=30, l1_ratio=0.5, 
    alphas=None, l1_ratios=None, n_splits=5, seed=42, max_iter=2000, 
    n_jobs=-1, project_root=None,
    **kwargs 
):
    """
    Wrapper for Stage 2 Training: Hallmark Experts.
    """
    log("=== Pipeline: Starting Stage 2 Training (Hallmark Experts) ===")
    _train_stage2(
        output_dir=output_dir,
        stage1_oof_path=stage1_oof,
        stage1_dict_path=stage1_dict,
        pc_path=pc_path,
        beta_path=beta_path,
        chalm_path=chalm_path,
        camda_path=camda_path,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        n_alphas=n_alphas,
        l1_ratio=l1_ratio,
        alphas=alphas,
        l1_ratios=l1_ratios,
        n_splits=n_splits,
        seed=seed,
        max_iter=max_iter,
        n_jobs=n_jobs,
        **kwargs
    )
    log("=== Pipeline: Stage 2 Training Completed ===\n")


def train_stage3(
    output_dir, stage1_oof, stage2_oof, pc_path,
    n_estimators=2000, learning_rate=0.01, num_leaves=31, 
    max_depth=-1, n_splits=5, seed=42, n_jobs=-1, project_root=None,
    n_trials=0,
    use_tissue_dummies=False,
    **kwargs 
):
    """
    Wrapper for Stage 3 Training: Context-Aware Fusion.
    Updated to support Optuna tuning arguments.
    """
    log("=== Pipeline: Starting Stage 3 Training (Context-Aware Fusion) ===")
    
    if n_trials > 0:
        log(f"   -> Optuna Tuning Enabled: {n_trials} trials")
        
    _train_stage3(
        output_dir=output_dir,
        stage1_oof_path=stage1_oof,
        stage2_oof_path=stage2_oof,
        pc_path=pc_path,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        n_splits=n_splits,
        seed=seed,
        n_jobs=n_jobs,
        n_trials=n_trials,
        use_tissue_dummies=use_tissue_dummies,
        **kwargs
    )
    log("=== Pipeline: Stage 3 Training Completed ===\n")


# ==============================================================================
# Inference Pipeline Wrappers
# ==============================================================================

def predict_stage1(artifact_dir, input_path, output_path):
    log("--- Pipeline: Running Stage 1 Inference ---")
    _predict_stage1(artifact_dir, input_path, output_path)

def predict_stage2(artifact_dir, input_path, output_path):
    log("--- Pipeline: Running Stage 2 Inference ---")
    _predict_stage2(artifact_dir, input_path, output_path)

def predict_stage3(artifact_dir, input_path, output_path):
    log("--- Pipeline: Running Stage 3 Inference ---")
    _predict_stage3(artifact_dir, input_path, output_path)

def predict_pipeline(artifact_dir, input_path, output_path):
    log("=== Pipeline: Starting Full Inference Pipeline ===")
    
    base_out_dir = os.path.dirname(output_path)
    os.makedirs(base_out_dir, exist_ok=True)
    
    s1_out = os.path.join(base_out_dir, "intermediate_stage1_preds.csv")
    s2_out = os.path.join(base_out_dir, "intermediate_stage2_preds.csv")
    merged_input_s3 = os.path.join(base_out_dir, "intermediate_stage3_input.csv")
    
    # 1. Stage 1
    s1_artifacts = os.path.join(artifact_dir, "stage1")
    predict_stage1(s1_artifacts, input_path, s1_out)
    
    # 2. Stage 2
    s2_artifacts = os.path.join(artifact_dir, "stage2")
    predict_stage2(s2_artifacts, input_path, s2_out)
    
    # 3. Merge
    log("--- Pipeline: Merging intermediate results for Stage 3 ---")
    if input_path.endswith('.csv'):
        df_in = pd.read_csv(input_path)
    else:
        df_in = pd.read_pickle(input_path)
        
    s1_df = pd.read_csv(s1_out)
    s2_df = pd.read_csv(s2_out)
    
    if "sample_id" not in df_in.columns:
        raise ValueError("Input data missing 'sample_id' column.")
        
    merged = df_in.merge(s1_df[['sample_id', 'pred_age_stage1']], on="sample_id", how="left")
    s2_cols = [c for c in s2_df.columns if c != 'sample_id']
    merged = merged.merge(s2_df[['sample_id'] + s2_cols], on="sample_id", how="left")
    
    merged.to_csv(merged_input_s3, index=False)
    
    # 4. Stage 3
    s3_artifacts = os.path.join(artifact_dir, "stage3")
    predict_stage3(s3_artifacts, merged_input_s3, output_path)
    
    # Cleanup
    if os.path.exists(s1_out): os.remove(s1_out)
    if os.path.exists(s2_out): os.remove(s2_out)
    if os.path.exists(merged_input_s3): os.remove(merged_input_s3)
    
    log(f"=== Pipeline: Full Inference Completed. Final results at {output_path} ===")