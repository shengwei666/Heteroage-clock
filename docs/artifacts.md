# Artifacts Documentation for heteroage-clock

## Overview

Artifacts are serialized files generated during the training process of the `heteroage-clock` pipeline. These artifacts are critical for reproducibility and are used for inference. This documentation outlines the structure and contents of the artifacts generated at each stage of the pipeline.

### Key Artifacts
- **Stage 1**: Global baseline model using ElasticNet with leakage-free cross-validation.
- **Stage 2**: Hallmark-specific expert models that correct the Stage 1 predictions.
- **Stage 3**: Context-aware fusion model using LightGBM to combine Stage 2 corrections with biological context features.

Each stage produces different outputs, including model weights, out-of-fold predictions, and metadata. The artifacts are stored in a structured directory and are essential for inference.

---

## 1. Stage 1 Artifacts

Stage 1 generates artifacts related to the global anchor model:

### Key Outputs:

- **`model.joblib`**: 
  - The trained ElasticNet model for Stage 1.
  - This model is trained on the global anchor features and is used to generate the initial biological age prediction (Global Anchor).

- **`Stage1_Sweep_Report.csv`**: 
  - Contains the results of the hyperparameter sweep for the Stage 1 model.
  - Includes metrics like `MicroMAE`, `MacroMAE`, and `Score` for each configuration.

- **`Stage1_Global_Anchor_OOF.csv`**: 
  - Out-of-fold (OOF) predictions from Stage 1.
  - These are used as inputs to Stage 2 for further refinement.

- **`Stage1_Orthogonalized_Hallmark_Dict.json`**: 
  - A JSON file mapping hallmark names to features after orthogonalization.
  - This file helps map the CpG features from Stage 1 to the hallmark-specific residuals in Stage 2.

- **`Stage1_Global_Model_Weights.csv`**: 
  - Contains the weights of the features used by the global model.
  - Provides insights into which features (CpGs) were most influential in the Stage 1 model.

### Example File Structure:

```bash
/path/to/output_stage1/artifacts/stage1/
├─ model.joblib
├─ Stage1_Sweep_Report.csv
├─ Stage1_Global_Anchor_OOF.csv
├─ Stage1_Orthogonalized_Hallmark_Dict.json
└─ Stage1_Global_Model_Weights.csv
```
---

## 2. Stage 2 Artifacts
Stage 2 generates artifacts related to the hallmark-specific expert models. These models refine the biological age prediction by focusing on specific biological domains (e.g., inflammation, metabolism).

### Key Outputs:

- **`hallmarks/*.joblib`**: 
  - One .joblib model per hallmark.
  - Each hallmark model is an ElasticNet model trained on the residuals from Stage 1.
  - These models correct the global predictions from Stage 1 based on domain-specific features.

- **`Stage2_Expert_OOF_Corrections.csv`**: 
  - Out-of-fold (OOF) corrections from the hallmark-specific expert models.
  - These corrections are used in Stage 3 for further fusion with context features.

- **`Stage2_Expert_Weights.csv`**: 
  - Contains the feature weights for each hallmark model.
  - Helps to understand the importance of different biological markers in the expert models.

- **`Stage2_Performance_Summary.csv`**: 
  - Contains performance metrics like MAE and R2 scores for each hallmark model.
  - Provides a summary of the expert models' accuracy and performance.

- **`Stage2_Manifest.json`**: 
  - A manifest file listing all available models and their respective metadata.
  - This file is required for efficient loading and management of Stage 2 models during inference.

### Example File Structure:

```bash
/path/to/output_stage2/artifacts/stage2/
├─ manifest.json
├─ hallmarks/
│   ├─ Hallmark_Inflammation.joblib
│   ├─ Hallmark_Metabolism.joblib
│   └─ Hallmark_Stress.joblib
├─ Stage2_Expert_OOF_Corrections.csv
├─ Stage2_Expert_Weights.csv
└─ Stage2_Performance_Summary.csv
```

---

## 3. Stage 3 Artifacts
Stage 3 generates the final fusion model, which integrates the Stage 2 expert corrections with biological context features (e.g., Tissue, Sex, Principal Components). This model is trained using LightGBM to produce the final biological age prediction and feature importance rankings.

### Key Outputs:

- **`model.joblib`**: 
  - The trained LightGBM model used for context-aware fusion.
  - This model combines Stage 2 expert corrections with tissue, sex, and PC features to generate the final biological age prediction.

- **`Stage3_Final_Predictions_OOF.csv`**: 
  - Out-of-fold (OOF) predictions from Stage 3.
  - These predictions represent the final biological age estimations after the fusion of Stage 2 corrections and context features.

- **`Stage3_Attention_Importance.csv`**: 
  - Contains feature importance values (gain) for the LightGBM model.
  - These values indicate which features (e.g., Tissue, Sex, specific hallmark corrections) contributed most to the final predictions.

- **`Stage3_Artifacts.json`**: 
  - A metadata file storing the configuration and parameters used in Stage 3.
  - Helps track the model parameters and provides essential information for reproducibility.

### Example File Structure:

```bash
/path/to/output_stage3/artifacts/stage3/
├─ model.joblib
├─ Stage3_Final_Predictions_OOF.csv
├─ Stage3_Attention_Importance.csv
└─ Stage3_Artifacts.json
```

---

## Using Artifacts for Inference
To perform inference, you need to assemble the artifacts from all three stages into a directory structure that the pipeline can use for prediction:

### Prepare Artifact Directory
```bash
# Create the necessary subdirectories for each stage:
mkdir -p /path/to/full_artifacts/stage1 /path/to/full_artifacts/stage2 /path/to/full_artifacts/stage3

# Copy the corresponding artifacts from each stage:
cp -r /path/to/output_stage1/artifacts/stage1/* /path/to/full_artifacts/stage1/
cp -r /path/to/output_stage2/artifacts/stage2/* /path/to/full_artifacts/stage2/
cp -r /path/to/output_stage3/artifacts/stage3/* /path/to/full_artifacts/stage3/
```
### Run Inference
Once the artifact directory is prepared, you can run the prediction pipeline:
```bash
heteroage pipeline-predict \
  --artifact-dir /path/to/full_artifacts \
  --input /path/to/master.parquet \
  --out /path/to/final_predictions.csv
```
This will produce the final biological age predictions by using the models trained across all three stages.

---

## Reproducibility and Determinism
All stages of the `heteroage-clock` pipeline are designed to be fully reproducible. The following practices ensure consistency:

1. Serialized Artifacts: All models and preprocessing steps (e.g., imputation, scaling) are serialized into files, which are used during inference to guarantee the same behavior across runs.
2. Column Contracts: The input features required by each model are saved and validated at inference time. Missing columns are handled by filling them with NaN and imputing based on training statistics.
3. Preprocessing Consistency: During inference, columns with only NaN values are forced to zero, which ensures column alignment between training and inference data.

By following these practices, you can be confident that results will be consistent and reproducible, provided the same input data and artifacts are used.

---

## Troubleshooting
### High Memory Usage
The training pipeline can be memory-intensive, especially during full data merging. Consider the following solutions:
- Use Parquet for large datasets: This can significantly reduce memory overhead.
- Chunked processing: If you are constructing a master table externally, consider processing it in chunks to avoid memory overload.

### Missing Expert Correction Columns
If Stage 3 fails due to missing expert correction columns, ensure that:
- Stage 2 artifacts are properly generated.
- The merged dataset for Stage 3 includes columns ending with `_Correction`.

### Missing RF_PC Columns
Stage 1 and Stage 3 both depend on the RF_PC columns (e.g., `RF_PC1`, `RF_PC2`, etc.). Ensure that your input table includes these columns or the models will behave unpredictably.

---

## Conclusion
The artifacts generated by each stage of the pipeline are essential for running the inference pipeline and ensuring the reproducibility of your results. By following the above guidelines and ensuring the correct artifact setup, you can confidently use the `heteroage-clock` for biological age prediction across different datasets and scenarios.