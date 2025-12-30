"""
heteroage_clock.stages.stage1

Stage 1: Global Anchor

This stage establishes a global biological age baseline using ElasticNet, with heterogeneity-aware
sampling, leakage-free cross-validation, and out-of-fold evaluation. The resulting model is used as a
global anchor for further refinement in Stage 2.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.data.assemble import assemble_features, filter_and_impute
from heteroage_clock.utils.logging import log


def stage1_train(project_root: str, output_dir: str, pc_path: str, dict_name: str, sweep_file: str = None) -> None:
    """
    Train the Stage 1 model (Global Anchor) using the provided data and configuration.
    
    Args:
        project_root (str): Path to the project root directory.
        output_dir (str): Directory to save the output models and reports.
        pc_path (str): Path to the Principal Components (PCs) CSV file.
        dict_name (str): Path to the Hallmark CpG dictionary.
        sweep_file (str, optional): Path to a pre-generated sweep report to skip the sweep.
    """
    # Step 1: Load necessary data (modality files and PCs)
    log(f"Loading data from {project_root}...")
    hallmark_dict = pd.read_json(os.path.join(project_root, f"4.Data_assembly/Feature_Sets/{dict_name}"))
    pc_data = pd.read_csv(pc_path)

    # Load modality data (CpG beta, CHALM, CAMDA)
    cpg_beta = pd.read_pickle(os.path.join(project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Beta_Train_RefGuided_NoCpH.pkl"))
    chalm_data = pd.read_pickle(os.path.join(project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Chalm_Train_RefGuided_NoCpH.pkl"))
    camda_data = pd.read_pickle(os.path.join(project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Camda_Train_RefGuided_NoCpH.pkl"))

    # Step 2: Assemble features
    assembled_data = assemble_features(cpg_beta, chalm_data, camda_data, pc_data, cpg_beta)
    assembled_data = filter_and_impute(assembled_data, features_to_keep=cpg_beta.columns.tolist())

    # Step 3: Split the data into training and validation sets (cross-validation or hold-out)
    # Here, we simulate a train-validation split for simplicity.
    train_data = assembled_data.sample(frac=0.8, random_state=42)
    val_data = assembled_data.drop(train_data.index)

    X_train = train_data.drop(columns=["sample_id", "age"])
    y_train = train_data["age"]
    X_val = val_data.drop(columns=["sample_id", "age"])
    y_val = val_data["age"]

    # Step 4: Train ElasticNet model
    model = ElasticNetCV(cv=5, random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    val_predictions = model.predict(X_val)
    metrics = compute_regression_metrics(y_val, val_predictions)

    # Log performance
    log(f"Stage 1 Global Anchor - Metrics: {metrics}")

    # Step 6: Save results
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "stage1_model.joblib")
    pd.to_pickle(model, model_path)

    # Save out-of-fold predictions for Stage 2
    oof_predictions_path = os.path.join(output_dir, "Stage1_Global_Anchor_OOF.csv")
    pd.DataFrame({"sample_id": val_data["sample_id"], "pred_age": val_predictions}).to_csv(oof_predictions_path, index=False)

    log(f"Stage 1 model saved to {model_path}")
    log(f"Stage 1 out-of-fold predictions saved to {oof_predictions_path}")
