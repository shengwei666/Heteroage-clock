"""
heteroage_clock.stages.stage3

Stage 3: Context-Aware Fusion

This stage fuses Stage 1 global predictions and Stage 2 hallmark corrections using a LightGBM model,
incorporating biological context (such as Tissue, Sex, and PCs). The result is a final biological age prediction.
"""

import os
import pandas as pd
import lightgbm as lgb
from heteroage_clock.core.metrics import compute_regression_metrics
from heteroage_clock.utils.logging import log


def stage3_train(project_root: str, output_dir: str, stage1_oof_path: str, stage2_oof_path: str, pc_path: str) -> None:
    """
    Train the Stage 3 model using LightGBM, fusing Stage 1 and Stage 2 outputs with context features.
    
    Args:
        project_root (str): Path to the project root directory.
        output_dir (str): Directory to save the output models and reports.
        stage1_oof_path (str): Path to Stage 1 OOF predictions CSV.
        stage2_oof_path (str): Path to Stage 2 OOF predictions CSV.
        pc_path (str): Path to Principal Components (PCs) CSV file.
    """
    # Step 1: Load necessary data
    log(f"Loading Stage 1 OOF predictions from {stage1_oof_path}...")
    stage1_oof = pd.read_csv(stage1_oof_path)

    log(f"Loading Stage 2 OOF predictions from {stage2_oof_path}...")
    stage2_oof = pd.read_csv(stage2_oof_path)

    log(f"Loading Principal Components (PCs) from {pc_path}...")
    pc_data = pd.read_csv(pc_path)

    # Step 2: Merge data (Stage 1 OOF, Stage 2 OOF, and PC context)
    merged_data = stage1_oof.merge(stage2_oof, on="sample_id", suffixes=("_stage1", "_stage2"))
    merged_data = merged_data.merge(pc_data, on="sample_id", how="left")

    # Step 3: Prepare data for LightGBM
    features = ["pred_age_stage1", "pred_residual_stage2"] + [col for col in pc_data.columns if col.startswith("RF_PC")]
    X = merged_data[features]
    y = merged_data["age"]

    # Step 4: Train LightGBM model
    model = lgb.LGBMRegressor(random_state=42, n_estimators=2000, learning_rate=0.01)
    model.fit(X, y)

    # Step 5: Evaluate the model
    val_predictions = model.predict(X)
    metrics = compute_regression_metrics(y, val_predictions)

    # Log performance
    log(f"Stage 3 (Fusion) - Metrics: {metrics}")

    # Step 6: Save results
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "stage3_model.joblib")
    pd.to_pickle(model, model_path)

    # Save final predictions
    final_predictions_path = os.path.join(output_dir, "Stage3_Final_Predictions.csv")
    pd.DataFrame({"sample_id": merged_data["sample_id"], "final_pred_age": val_predictions}).to_csv(final_predictions_path, index=False)

    log(f"Stage 3 model saved to {model_path}")
    log(f"Stage 3 final predictions saved to {final_predictions_path}")
