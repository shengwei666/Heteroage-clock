"""
heteroage_clock.stages.stage2

Stage 2: Hallmark Experts

This stage focuses on individual biological hallmarks, training separate expert models for each hallmark
using Stage 1 residuals. Each hallmark model is a separate ElasticNet regression model trained on the residuals
after Stage 1's global prediction.
"""

import os
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.utils.logging import log


def stage2_train(project_root: str, output_dir: str, stage1_oof_path: str, hallmark_dict_path: str) -> None:
    """
    Train Stage 2 expert models for individual biological hallmarks using Stage 1 residuals as the target.
    
    Args:
        project_root (str): Path to the project root directory.
        output_dir (str): Directory to save the output models and reports.
        stage1_oof_path (str): Path to the Stage 1 OOF predictions CSV file.
        hallmark_dict_path (str): Path to the Hallmark CpG dictionary file.
    """
    # Step 1: Load necessary data
    log(f"Loading Stage 1 OOF predictions from {stage1_oof_path}...")
    stage1_oof = pd.read_csv(stage1_oof_path)

    log(f"Loading Hallmark dictionary from {hallmark_dict_path}...")
    hallmark_dict = pd.read_json(hallmark_dict_path)

    # Step 2: Process each hallmark
    for hallmark, cpgs in hallmark_dict.items():
        log(f"Training hallmark model: {hallmark} with CpGs {cpgs}")

        # Prepare data for the hallmark model
        hallmark_data = stage1_oof[["sample_id", "pred_age", "age"] + cpgs]
        hallmark_data["residual"] = hallmark_data["age"] - hallmark_data["pred_age"]

        # Feature and target preparation
        X = hallmark_data[cpgs]
        y = hallmark_data["residual"]

        # Step 3: Train ElasticNet model for this hallmark
        model = ElasticNetCV(cv=5, random_state=42)
        model.fit(X, y)

        # Step 4: Evaluate the model
        val_predictions = model.predict(X)
        metrics = compute_regression_metrics(y, val_predictions)

        # Log performance
        log(f"Hallmark {hallmark} - Metrics: {metrics}")

        # Step 5: Save results
        os.makedirs(output_dir, exist_ok=True)
        hallmark_model_path = os.path.join(output_dir, f"stage2_{hallmark}_model.joblib")
        pd.to_pickle(model, hallmark_model_path)

        # Save predictions
        hallmark_predictions_path = os.path.join(output_dir, f"Stage2_{hallmark}_Corrections.csv")
        pd.DataFrame({"sample_id": hallmark_data["sample_id"], "pred_residual": val_predictions}).to_csv(hallmark_predictions_path, index=False)

        log(f"Hallmark {hallmark} model saved to {hallmark_model_path}")
        log(f"Hallmark {hallmark} predictions saved to {hallmark_predictions_path}")
