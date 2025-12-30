# CLI Documentation for heteroage-clock

## Overview

The `heteroage-clock` CLI allows users to interact with the biological age prediction pipeline through simple commands. This CLI supports all three stages of the pipeline and is designed to be efficient and easy to use for both training and inference.

In addition to the main training commands for each stage, the CLI provides functionalities for preprocessing, model prediction, and more. This documentation covers the available commands, their arguments, and usage examples.

---

## 1. Command Structure

The CLI follows a hierarchical structure with three main categories of commands:

- **Stage 1 Commands**: For training the global anchor model.
- **Stage 2 Commands**: For training hallmark-specific expert models.
- **Stage 3 Commands**: For performing context-aware fusion using LightGBM.

Additionally, the CLI offers options for inference and using serialized model artifacts.

---

## 2. Available Commands

### 2.1. Stage 1: Global Anchor

The **Stage 1** command is used for training the global baseline model and generating the required outputs for the next stage.

#### Command:

```bash
heteroage stage1-train --project-root <PROJECT_ROOT> --output-dir <OUTPUT_DIR> --pc-path <PC_PATH> --dict-name <DICT_NAME> --sweep-file <SWEEP_FILE>
```
Arguments:

- --project-root: Path to the root directory containing modality PKLs and the hallmark dictionary.

- --output-dir: Path where the outputs of Stage 1 will be saved.

- --pc-path: Path to the CSV file containing the Principal Component (PC) covariates (columns should start with RF_PC).

- --dict-name: Name of the hallmark dictionary file (e.g., Hallmark_CpG_Dict_Final.json).

- --sweep-file: Optional argument to provide an existing sweep report and skip the sweep phase.

Example Usage:
```bash
heteroage stage1-train \
  --project-root /path/to/project_root \
  --output-dir /path/to/output_stage1 \
  --pc-path /path/to/Global_Healthy_RF_PCs.csv \
  --dict-name Hallmark_CpG_Dict_Final.json
```
This command will perform the following:

- Run a hyperparameter sweep over sampling parameters.
- Execute leakage-free cross-validation.
- Save trained models and reports in the specified output directory.

### 2.2. Stage 2: Hallmark Experts

The **Stage 2** command trains the hallmark-specific expert models, using the residuals from Stage 1 predictions as the target for training.

#### Command:

```bash
heteroage stage2-train --project-root <PROJECT_ROOT> --output-dir <OUTPUT_DIR> --pc-path <PC_PATH> --stage1-oof <STAGE1_OOF> --stage1-dict <STAGE1_DICT>
```
Arguments:

- --project-root: Path to the root directory containing the Stage 1 outputs and modality PKLs.
- --output-dir: Path where the outputs of Stage 2 will be saved.
- --pc-path: Path to the CSV file containing the Principal Component (PC) covariates.
- --stage1-oof: Path to the Stage 1 OOF (Out-of-Fold) predictions.
- --stage1-dict: Path to the Stage 1 orthogonalized hallmark dictionary.

Example Usage:
```bash
heteroage stage2-train \
  --project-root /path/to/project_root \
  --output-dir /path/to/output_stage2 \
  --pc-path /path/to/Global_Healthy_RF_PCs.csv \
  --stage1-oof /path/to/output_stage1/Stage1_Global_Anchor_OOF.csv \
  --stage1-dict /path/to/output_stage1/Stage1_Orthogonalized_Hallmark_Dict.json
```
This command will perform the following:

- Train hallmark-specific expert models using Stage 1 residuals.
- Generate out-of-fold (OOF) corrections and expert weights.
- Save trained models and performance reports.

### 2.3. Stage 3: Context-Aware Fusion

The **Stage 3** command performs the final stage of the pipeline, which fuses the Stage 2 corrections with context features using LightGBM.

#### Command:

```bash
heteroage stage3-train --output-dir <OUTPUT_DIR> --stage1-oof <STAGE1_OOF> --stage2-oof <STAGE2_OOF> --pc-path <PC_PATH>
```
Arguments:

- --output-dir: Path where the outputs of Stage 3 will be saved.
- --stage1-oof: Path to the Stage 1 OOF predictions.
- --stage2-oof: Path to the Stage 2 expert corrections.
- --pc-path: Path to the CSV file containing the Principal Component (PC) covariates.

Example Usage:
```bash
heteroage stage3-train \
  --output-dir /path/to/output_stage3 \
  --stage1-oof /path/to/output_stage1/Stage1_Global_Anchor_OOF.csv \
  --stage2-oof /path/to/output_stage2/Stage2_Expert_OOF_Corrections.csv \
  --pc-path /path/to/Global_Healthy_RF_PCs.csv
```
This command will perform the following:

- Merge Stage 1 and Stage 2 outputs along with context features.
- Train the final LightGBM model using cross-validation.
- Generate OOF predictions and feature importance (gain).
- Save the trained model and performance metrics.


## 3. Inference Commands
After training the models, you can use them for inference. There are options for making predictions at each stage, or you can run the entire pipeline end-to-end.

### 3.1. Pipeline Prediction (Stage 1 + Stage 2 + Stage 3)
To run the full pipeline and get the final biological age prediction, use the following command:
#### Command:
```bash
heteroage pipeline-predict --artifact-dir <ARTIFACT_DIR> --input <INPUT_FILE> --out <OUTPUT_FILE>
```
Arguments:
- --artifact-dir: Path to the directory containing all the trained models from Stage 1, Stage 2, and Stage 3.
- --input: Path to the input file (in CSV or Parquet format). This file must contain the required columns, including sample_id, Tissue, Sex, RF_PC columns, and CpG features.
- --out: Path where the final predictions will be saved.

Example Usage:
```bash
heteroage pipeline-predict \
  --artifact-dir /path/to/full_artifacts \
  --input /path/to/master.parquet \
  --out /path/to/final_predictions.csv
```
This will produce the final biological age predictions from the entire pipeline.

- Merge Stage 1 and Stage 2 outputs along with context features.
- Train the final LightGBM model using cross-validation.
- Generate OOF predictions and feature importance (gain).
- Save the trained model and performance metrics.

### 3.2. Stage-only Prediction
You can also run individual stages for inference. Below are commands for Stage 1, Stage 2, and Stage 3.

#### Stage 1 Only:
#### Command:
```bash
heteroage stage1-predict --artifact-dir <ARTIFACT_DIR> --input <INPUT_FILE> --out <OUTPUT_FILE>
```

#### Stage 2 Only:
#### Command:
```bash
heteroage stage2-predict --artifact-dir <ARTIFACT_DIR> --input <INPUT_FILE> --out <OUTPUT_FILE>
```

#### Stage 3 Only:
#### Command:
```bash
heteroage stage3-predict --artifact-dir <ARTIFACT_DIR> --input <INPUT_FILE> --out <OUTPUT_FILE>
```
For Stage 3, the input file must already include Stage 1 predictions and Stage 2 corrections.

## 4. Helper Commands
### 4.1. Checking Installed Version
To check the installed version of the heteroage-clock package, use:
```bash
heteroage --version
```

### 4.2. Help Command
To get help with any command, use the `--help` option:
```bash
heteroage stage1-train --help
heteroage stage2-train --help
heteroage stage3-train --help
heteroage pipeline-predict --help
```

## 5. Conclusion
The `heteroage-clock` CLI provides a flexible interface for training, evaluating, and performing inference with biological age prediction models. By using the provided commands, you can quickly set up the pipeline, train the models, and make predictions for biological age across different datasets. Ensure that you have the appropriate input files and artifacts in place to achieve accurate and reproducible results.