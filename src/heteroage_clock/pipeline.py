"""
heteroage_clock.pipeline

This module orchestrates the training and inference pipelines.
It acts as a high-level interface that calls the specific logic defined in
heteroage_clock.stages.*.
"""

import os
import pandas as pd
from heteroage_clock.utils.logging import log

# Import actual implementations from stages
from heteroage_clock.stages.stage1 import train_stage1 as _train_stage1, predict_stage1 as _predict_stage1
from heteroage_clock.stages.stage2 import train_stage2 as _train_stage2, predict_stage2 as _predict_stage2
from heteroage_clock.stages.stage3 import train_stage3 as _train_stage3, predict_stage3 as _predict_stage3


# ==============================================================================
# Training Pipeline Wrappers
# ==============================================================================

def train_stage1(project_root, output_dir, pc_path, dict_name, sweep_file=None):
    """
    Wrapper for Stage 1 Training: Global Anchor.
    """
    log("=== Pipeline: Starting Stage 1 Training (Global Anchor) ===")
    _train_stage1(
        project_root=project_root,
        output_dir=output_dir,
        pc_path=pc_path,
        dict_name=dict_name,
        sweep_file=sweep_file
    )
    log("=== Pipeline: Stage 1 Training Completed ===\n")


def train_stage2(project_root, output_dir, pc_path, stage1_oof, stage1_dict):
    """
    Wrapper for Stage 2 Training: Hallmark Experts.
    """
    log("=== Pipeline: Starting Stage 2 Training (Hallmark Experts) ===")
    _train_stage2(
        project_root=project_root,
        output_dir=output_dir,
        stage1_oof_path=stage1_oof,
        hallmark_dict_path=os.path.join(project_root, "4.Data_assembly/Feature_Sets", stage1_dict),
        pc_path=pc_path
    )
    log("=== Pipeline: Stage 2 Training Completed ===\n")


def train_stage3(stage1_oof, stage2_oof, pc_path, output_dir, project_root=None):
    """
    Wrapper for Stage 3 Training: Context-Aware Fusion.
    """
    log("=== Pipeline: Starting Stage 3 Training (Context-Aware Fusion) ===")
    _train_stage3(
        project_root=project_root, # Passed for consistency logging
        output_dir=output_dir,
        stage1_oof_path=stage1_oof,
        stage2_oof_path=stage2_oof,
        pc_path=pc_path
    )
    log("=== Pipeline: Stage 3 Training Completed ===\n")


# ==============================================================================
# Inference Pipeline Wrappers
# ==============================================================================

def predict_stage1(artifact_dir, input_path, output_path):
    """
    Wrapper for Stage 1 Prediction.
    """
    log("--- Pipeline: Running Stage 1 Inference ---")
    _predict_stage1(
        artifact_dir=artifact_dir,
        input_path=input_path,
        output_path=output_path
    )


def predict_stage2(artifact_dir, input_path, output_path):
    """
    Wrapper for Stage 2 Prediction.
    """
    log("--- Pipeline: Running Stage 2 Inference ---")
    _predict_stage2(
        artifact_dir=artifact_dir,
        input_path=input_path,
        output_path=output_path
    )


def predict_stage3(artifact_dir, input_path, output_path):
    """
    Wrapper for Stage 3 Prediction.
    """
    log("--- Pipeline: Running Stage 3 Inference ---")
    _predict_stage3(
        artifact_dir=artifact_dir,
        input_path=input_path,
        output_path=output_path
    )


def predict_pipeline(artifact_dir, input_path, output_path):
    """
    Run the FULL inference pipeline (Stage 1 -> Stage 2 -> Stage 3).
    
    This function handles the data flow between stages automatically:
    1. Run Stage 1 -> produces pred_age
    2. Run Stage 2 -> produces pred_residual_* (hallmarks)
    3. Merge outputs from 1 & 2 + Input Context
    4. Run Stage 3 -> produces final_pred_age
    
    Args:
        artifact_dir (str): Root directory containing subdirectories 'stage1', 'stage2', 'stage3' with artifacts.
        input_path (str): Path to original input data (must contain features for all stages).
        output_path (str): Path to save the FINAL predictions.
    """
    log("=== Pipeline: Starting Full Inference Pipeline ===")
    
    # Define intermediate paths
    # We create a temporary or intermediate folder structure to hold stage outputs
    base_out_dir = os.path.dirname(output_path)
    s1_out = os.path.join(base_out_dir, "intermediate_stage1_preds.csv")
    s2_out = os.path.join(base_out_dir, "intermediate_stage2_preds.csv")
    merged_input_s3 = os.path.join(base_out_dir, "intermediate_stage3_input.csv")
    
    # 1. Stage 1 Inference
    # Assumes artifacts are in artifact_dir/stage1
    s1_artifacts = os.path.join(artifact_dir, "stage1")
    predict_stage1(s1_artifacts, input_path, s1_out)
    
    # 2. Stage 2 Inference
    # Assumes artifacts are in artifact_dir/stage2
    s2_artifacts = os.path.join(artifact_dir, "stage2")
    predict_stage2(s2_artifacts, input_path, s2_out)
    
    # 3. Merge Data for Stage 3
    # Stage 3 needs: Stage 1 preds + Stage 2 preds + Context (from original input)
    log("--- Pipeline: Merging intermediate results for Stage 3 ---")
    
    # Load original input to get Context (PCs, Sex, etc.)
    if input_path.endswith('.csv'):
        df_in = pd.read_csv(input_path)
    else:
        df_in = pd.read_pickle(input_path)
        
    s1_df = pd.read_csv(s1_out)
    s2_df = pd.read_csv(s2_out)
    
    # Merge on sample_id
    # Note: Ensure all inputs have 'sample_id'
    merged = df_in.merge(s1_df, on="sample_id").merge(s2_df, on="sample_id")
    
    # Save merged file for Stage 3 input
    merged.to_csv(merged_input_s3, index=False)
    
    # 4. Stage 3 Inference
    # Assumes artifacts are in artifact_dir/stage3
    s3_artifacts = os.path.join(artifact_dir, "stage3")
    predict_stage3(s3_artifacts, merged_input_s3, output_path)
    
    # Optional: Cleanup intermediate files
    # if os.path.exists(s1_out): os.remove(s1_out)
    # if os.path.exists(s2_out): os.remove(s2_out)
    # if os.path.exists(merged_input_s3): os.remove(merged_input_s3)
    
    log(f"=== Pipeline: Full Inference Completed. Final results at {output_path} ===")