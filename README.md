<div align="center">

# 🕰️ HeteroAge-Clock

**An industrial-grade, three-stage biological age prediction pipeline designed for high-performance and reproducible aging research.**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Pipeline Architecture](#-1-pipeline-architecture) •
[Installation](#-2-installation) •
[Quick Start](#-4-quick-start-training) •
[Docs](docs/)

</div>

---

`heteroage-clock` is derived directly from specialized research scripts, decoupling training from inference to allow for seamless deployment of biological age models through serialized artifacts.

## 📖 Table of Contents
- [ ✨ 1. Pipeline Architecture](#-1-pipeline-architecture)
- [ ✨ 2. Installation](#-2-installation)
- [ ✨ 3. Data Requirements](#-3-data-requirements)
- [ ✨ 4. Quick Start: Training](#-4-quick-start-training)
- [ ✨ 5. Quick Start: Inference](#-5-quick-start-inference)
- [ ✨ 6. Repository Layout](#-6-repository-layout)
- [ ✨ 7. Key Features](#-7-key-features--reproducibility)

## ✨ 1. Pipeline Architecture

The system utilizes a hierarchical approach to balance global population trends with specific biological variations:

* **Stage 1 (Global Anchor)**: Establishes a robust global baseline using ElasticNet with heterogeneity-aware balanced sampling. It utilizes leakage-free group Cross-Validation (CV) and Out-of-Fold (OOF) evaluation to produce deployable artifacts.
* **Stage 2 (Hallmark Experts)**: Consists of independent, hallmark-specific residual expert models (ElasticNet). These are trained to correct the global anchor based on specific biological domains (e.g., inflammation, metabolism).
* **Stage 3 (Context-Aware Fusion)**: A LightGBM meta-learner that fuses Stage 2 expert corrections with biological context features (Tissue, Sex, Principal Components) to produce the final biological age and feature importance ranking.

---

## ✨ 2. Installation
### 🛠️ Tech Stack & Core Dependencies

| Category | Library | Purpose |
| :--- | :--- | :--- |
| **Machine Learning** | `scikit-learn` | ElasticNet engine for Stage 1 & 2. |
| **Gradient Boosting** | `LightGBM` | Meta-learner for Stage 3 fusion. |
| **Data Processing** | `pandas`, `numpy` | High-performance matrix manipulation. |
| **Serialization** | `joblib`, `pickle` | Artifact deployment and model persistence. |
| **Bioinformatics** | `pyarrow` | Optimized Parquet I/O for large methylome data. |

### Option A: Install from Source
If you want to use the latest version or modify the code, install in **editable mode**:

```bash
# 1. Clone the repository
git clone https://github.com/shengwei666/Heteroage-clock.git
cd Heteroage-clock

# 2. Install dependencies and the package in editable mode
#conda create -n heteroage-clock-env python=3.8
#conda activate heteroage-clock-env

pip install -e .

#heteroage --help
#heteroage stage1-train --help
```
**Note:** The `-e` flag stands for "editable". It allows you to modify the source code in `src/` and have the changes take effect immediately without re-installing.

### Option B: Install as a Standard Package
If you just want to use the tool without modifying it:

```bash
# 1. Clone the repository
git clone https://github.com/shengwei666/Heteroage-clock.git
cd Heteroage-clock

# 2. Install the package
#conda create -n heteroage-clock-env python=3.8
#conda activate heteroage-clock-env

pip install .
#heteroage --help
```
---
## ✨3. Data Requirements

### 3.1 Project Root Layout
The pipeline expects a specific directory structure inside your `--project-root`.

| File Type | Default Path (Relative to Project Root) |
| :--- | :--- |
| **Feature Dict** | `4.Data_assembly/Feature_Sets/Hallmark_CpG_Dict_Final.json` |
| **Beta Matrix** | `4.Data_assembly/Raw_Aligned_RefGuided/Beta_Train_RefGuided_NoCpH.pkl` |
| **Chalm Matrix** | `4.Data_assembly/Raw_Aligned_RefGuided/Chalm_Train_RefGuided_NoCpH.pkl` |
| **Camda Matrix** | `4.Data_assembly/Raw_Aligned_RefGuided/Camda_Train_RefGuided_NoCpH.pkl` |

### 3.2 Metadata Files

#### 📂 PC File (Required)
The `--pc-path` argument must point to a CSV file containing the following columns:

* `sample_id` (Unique identifier)
* `Tissue` (Tissue type)
* `RF_PC*` (One or more Principal Component columns, e.g., `RF_PC1`, `RF_PC2`...)

#### 📂 Modality PKL Files (Required for Training)
Each Pickle file (containing a pandas DataFrame) must include the following metadata and features:

* **Metadata:** `sample_id`, `project_id`, `Tissue`, `Age`, `Sex`, `Is_Healthy`
* **Features:** CpG columns (named by CpG ID, e.g., `cg00000029`)

> ⚠️ **Important Note:**
> The training pipeline strictly filters samples where `Is_Healthy == True`. Ensure your training data contains this column and strictly labels healthy controls as `True`.

---

## ✨ 4. Quick Start: Training

The training process supports Parallel Grid Search and Hyperparameter Lists to significantly reduce computation time.

---

### Stage 1: Global Anchor Training
The Global Anchor establishes the baseline biological age.

**Main Tasks:**
* **Full-Set ElasticNet**: Learns from the union of all modalities (Beta, CHALM, CAMDA) to capture a comprehensive aging signal.
* **Macro + Micro Optimization**: Performs a grid search for the best regularization (α and l1_ratio) that balances global accuracy (Micro) and cross-dataset generalizability (Macro).
* **Stable Orthogonalization**: Uses Pearson Correlation Ranking to assign overlapping CpG sites to the most biologically relevant Hallmark, ensuring Stage 2 experts are independent.

```bash
# Execute Stage 1 Training with Parallel Grid Search
heteroage stage1-train \
  --output-dir ./results/stage1 \
  --pc-path ./data/RefGuided_PCs.csv \
  --dict-path ./data/Hallmark_CpG_Dict_Final.json \
  --beta-path ./data/Beta_Train.pkl \
  --chalm-path ./data/Chalm_Train.pkl \
  --camda-path ./data/Camda_Train.pkl \
  --alphas 0.1 0.01 0.001 \
  --l1-ratios 0.5 0.7 0.9 \
  --n-jobs -1
```
**📂 Key Outputs:**

| File / Directory | Description |
| :--- | :--- |
| `stage1_grid_search_report.csv` | Performance metrics for all tested hyperparameter combinations. |
| `stage1_orthogonalized_dict.json` | Refined CpG-to-Hallmark mapping based on Stage 1 feature ranking. |
| `stage1_oof_predictions.csv` | Out-of-Fold predictions used as the training baseline for Stage 2. |
| `artifacts/stage1/` | **Deployable Directory**: Contains serialized ElasticNet model and feature indices. |

### Stage 2: Hallmark Experts Training
Trains Hallmark-specific expert models to capture biological variations (residuals).

**Main Tasks:**
* **Hallmark Mapping**: Utilizes the orthogonalized dictionary derived from Stage 1 to group CpG sites.
* **Residual Prediction**: Each expert model is trained to predict the difference (residual) between the chronological age and the Stage 1 global prediction.
* **Leakage-free CV**: Employs robust Cross-Validation to generate Out-of-Fold (OOF) corrections.

```bash
# Execute Stage 2 Training with Parallel Expert Tuning
heteroage stage2-train \
  --output-dir ./results/stage2 \
  --stage1-oof ./results/stage1/stage1_oof_predictions.csv \
  --stage1-dict ./results/stage1/stage1_orthogonalized_dict.json \
  --pc-path ./data/RefGuided_PCs.csv \
  --beta-path ./data/Beta_Train.pkl \
  --chalm-path ./data/Chalm_Train.pkl \
  --camda-path ./data/Camda_Train.pkl \
  --alphas 0.1 0.01 \
  --n-jobs -1
```
**📂 Key Outputs:**

| File / Directory | Description |
| :--- | :--- |
| `Stage2_Hallmark_OOF.csv` | Hallmark-specific residual corrections used as input features for the Stage 3 Meta-Learner. |
| `stage2_{hallmark}_expert_model.joblib` | Serialized ElasticNet weights for each biological hallmark expert. |
| `stage2_{hallmark}_features.csv` | Detailed list of CpG features assigned to each specific hallmark expert. |
| `artifacts/stage2/` | **Deployable Directory**: Contains all serialized expert models and their corresponding feature indices. |

> [!NOTE]
> Stage 2 is critical for capturing domain-specific aging signals (e.g., inflammation, metabolism, or epigenetic drift) that are often masked in a single-stage global model.


### Stage 3: Context-Aware Fusion
Final fusion using LightGBM, incorporating tissue and sex context.

**Main Tasks:**
* **Data Integration**: Merges Stage 1 OOF predictions, Stage 2 expert corrections, and Principal Component (PC) context.
* **Robust Validation**: Implements leakage-free **GroupKFold** cross-validation based on `project_id` and `Tissue`.
* **Interpretability**: Generates feature importance (Gain) to reveal which biological drivers contribute most to the final age.
* **Meta-Serialization**: Produces the final deployable meta-learner artifacts.

```bash
# Execute Stage 3 Training
heteroage stage3-train \
  --output-dir ./results/stage3 \
  --stage1-oof ./results/stage1/stage1_oof_predictions.csv \
  --stage2-oof ./results/stage2/Stage2_Hallmark_OOF.csv \
  --pc-path ./data/RefGuided_PCs.csv \
  --n-jobs -1
```
**📂 Key Outputs:**

| File / Directory | Description |
| :--- | :--- |
| `Stage3_Final_Predictions_Train.csv` | The final integrated "HeteroAge" predictions for the training cohort (Out-of-Fold). |
| `Stage3_Feature_Importance.csv` | Ranking of feature contributions (Gain/Split) across all Hallmarks and PC contexts. |
| `artifacts/stage3/` | **Deployable Directory**: Contains the serialized LightGBM meta-learner and feature indices. |

> [!TIP]
> **Biological Interpretability:** The `Stage3_Attention_Importance.csv` is the "brain" of the clock. It reveals whether the model is prioritizing specific biological hallmarks (e.g., Inflammation) or tissue-specific contexts when calculating biological age.
---

## ✨ 5. Quick Start: Inference

Inference is designed to be **fully decoupled** from the training environment. By utilizing serialized artifacts, the pipeline ensures that biological age predictions are consistent across different datasets and compute environments.

---

### 5.1 Artifact Preparation
Before running inference, consolidate the artifacts generated from all three training stages into a unified directory structure.

```bash
# 1. Create a centralized artifact directory
mkdir -p ./deploy_model/stage1 ./deploy_model/stage2 ./deploy_model/stage3

# 2. Collect artifacts from training outputs
cp -r /path/to/output_stage1/artifacts/stage1/* ./deploy_model/stage1/
cp -r /path/to/output_stage2/artifacts/stage2/* ./deploy_model/stage2/
cp -r /path/to/output_stage3/artifacts/stage3/* ./deploy_model/stage3/
```

### 5.2 End-to-End Pipeline Prediction

The `pipeline-predict` command orchestrates the full three-stage inference flow: **Global Anchor** → **Hallmark Experts** → **Contextual Fusion**. This is the recommended way to generate final age predictions.

**Input Table Requirements:**
The input file (CSV or Parquet) must contain the following schema:

| Category | Columns / Pattern | Description |
| :--- | :--- | :--- |
| **Metadata** | `sample_id`, `Tissue`, `Sex` | Essential biological and administrative context. |
| **Context** | `RF_PC*` (e.g., `RF_PC1`, `RF_PC2`) | Principal Components used for tissue-specific adjustment. |
| **Features** | `{cpg_id}_beta`, `{cpg_id}_chalm`,  `{cpg_id}_camda` | CpG features suffixed by modality as per trained artifacts. |

**Run Prediction:**
```bash
heteroage pipeline-predict \
  --artifact-dir ./deploy_model \
  --input /path/to/master.parquet \
  --out /path/to/final_predictions.csv
```

### 🎯 5.3 Granular Stage-Specific Prediction

For modular analysis, debugging, or research into specific biological hallmarks, you can execute individual stages independently using their respective serialized artifacts.

#### Stage 1: Global Anchor Prediction
Generates the baseline biological age based on global population trends.
```bash
heteroage stage1-predict \
  --artifact-dir ./deploy_model/stage1 \
  --input /path/to/master.parquet \
  --out /path/to/stage1_predictions.csv
```

#### Stage 2: Hallmark Experts Prediction
Calculates independent biological deviations (residuals) for each specific Hallmark.

```bash
# Generate hallmark-specific residuals
heteroage stage2-predict \
  --artifact-dir ./deploy_model/stage2 \
  --input /path/to/master.parquet \
  --out /path/to/stage2_corrections.csv
```

#### Stage 3: Context-Aware Fusion Prediction
Fuses the baseline global trends and hallmark-specific corrections with biological context (Tissue, Sex, PCs) to produce the final "HeteroAge" prediction.

> [!IMPORTANT]
> **Data Dependency**: Stage 3 requires an input table that has already been enriched with Stage 2 outputs. Ensure your input file contains columns starting with `pred_residual_` (e.g., `pred_residual_Inflammation`) as generated by Stage 2.

```bash
# Generate final fused biological age
heteroage stage3-predict \
  --artifact-dir ./deploy_model/stage3 \
  --input /path/to/merged_stage1_stage2.parquet \
  --out /path/to/stage3_predictions.csv
```

---

## ✨ 6. Repository Layout
The project follows a modular **src-layout** to ensure clear separation between core logic, pipeline orchestration, and artifact management.

```text
heteroage-clock/
├── .github/                # CI/CD workflows and GitHub actions
├── docs/                   # Detailed technical documentation & CLI specs
├── src/
│   └── heteroage_clock/    # Core package source
│       ├── core/           # Mathematical logic & modeling engine
│       ├── stages/         # Stage 1, 2, and 3 implementation logic
│       ├── artifacts/      # Model serialization & artifact handlers
│       ├── data/           # Data loading & validation schemas
│       ├── utils/          # Logging & helper functions
│       └── cli.py          # Command-line interface entry point
├── tests/                  # Unit and integration test suite
├── .pre-commit-config.yaml # Linting & formatting hooks (Black, Flake8)
├── pyproject.toml          # Build system and dependency management
└── Makefile                # Shortcuts for installation and testing
```

---

## ✨ 7. Key Features & Reproducibility

`heteroage-clock` is built with a "production-first" mindset, ensuring that biological insights are backed by rigorous engineering and reproducible workflows.

| Feature | Implementation Detail |
| :--- | :--- |
| **High-Performance Parallelism** | Leverages `joblib` for multi-core hyperparameter grid searching and LightGBM multi-threading, significantly reducing training time. |
| **Flexible Hyperparameter Search** | Supports both automated range generation and explicit user-defined lists for `alphas` and `l1-ratios` via CLI. |
| **Leakage-Free Evaluation** | All CV splits are strictly grouped by `project_id` and stratified by `Tissue`, preventing over-fitting and ensuring cross-study generalizability. |
| **Deterministic Inference** | Every stage serializes its entire environment—including **Imputers, Age Transformers, Selected Column Indices**, and model weights—to ensure bit-identical results. |

---

## 🛠️ Troubleshooting

### ⚠️ High Memory Usage
Processing large-scale DNA methylation matrices can be resource-intensive. 
* **Use Parquet**: Convert large CSV/PKL files to Parquet format for faster I/O and lower memory overhead.
* **Manage Parallelism**: When training on very large datasets, high `--n-jobs` values may lead to memory exhaustion (OOM). Adjust according to available RAM.
* **Feature Subsetting**: Only load the CpG sites defined in your `Feature_Dict` or trained artifacts.

### ⚠️ Missing Correction Columns
Stage 3 requires inputs from Stage 2. 
* If you are running Stage 3 independently, ensure your input table contains Hallmark correction columns suffixed with **`_Correction`** (e.g., `Inflammation_Correction`).

### ⚠️ Missing RF_PC Columns
The pipeline relies heavily on Principal Components for tissue-specific normalization.
* Ensure your PC metadata file and inference master tables contain columns matching the **`RF_PC*`** pattern (e.g., `RF_PC1`, `RF_PC2`, etc.).
