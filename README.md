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
pip install -e .
```
**Note:** The `-e` flag stands for "editable". It allows you to modify the source code in `src/` and have the changes take effect immediately without re-installing.

### Option B: Install as a Standard Package
If you just want to use the tool without modifying it:

```bash
# 1. Clone the repository
git clone https://github.com/shengwei666/Heteroage-clock.git
cd Heteroage-clock

# 2. Install the package
pip install .
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

The training process is executed in a decoupled, stage-by-stage workflow. This design ensures that each component can be validated independently, producing both biological insights and deterministic inference artifacts.

---

### Stage 1: Global Anchor Training
The Global Anchor establishes the baseline biological age.

**Main Tasks:**
* **Full-Set ElasticNet**: Learns from the union of all modalities (Beta, CHALM, CAMDA) to capture a comprehensive aging signal.
* **Macro + Micro Optimization**: Performs a grid search for the best regularization (α) that balances global accuracy (Micro) and cross-dataset generalizability (Macro).
* **Stable Orthogonalization**: Uses Pearson Correlation Ranking to assign overlapping CpG sites to the most biologically relevant Hallmark, ensuring Stage 2 experts are independent.
* **Artifact Generation**: Saves the global model, the stable hallmark dictionary, and Out-of-Fold (OOF) residuals.

```bash
# Execute Stage 1 Training
heteroage stage1-train \
  --output-dir ./results/stage1 \
  --pc-path ./data/RefGuided_PCs.csv \
  --dict-path ./data/Hallmark_CpG_Dict_Final.json \
  --beta-path ./data/Beta_Train.pkl \
  --chalm-path ./data/Chalm_Train.pkl \
  --camda-path ./data/Camda_Train.pkl \
  --l1-ratio 0.5 \
  --n-alphas 30
```
**📂 Key Outputs:**

| File / Directory | Description |
| :--- | :--- |
| `Stage1_Sweep_Report.csv` | Performance metrics and logs for all parameter sweep configurations. |
| `Stage1_Orthogonalized_Hallmark_Dict.json` | Refined CpG-to-Hallmark mapping based on Stage 1 feature importance. |
| `Stage1_Global_Anchor_OOF.csv` | Out-of-Fold predictions used as the training baseline for Stage 2. |
| `Stage1_Global_Model_Weights.csv` | CSV export of the finalized ElasticNet coefficients. |
| `artifacts/stage1/` | **Deployable Directory**: Contains serialized preprocessors and model artifacts. |

> [!TIP]
> **Efficiency Gain:** You can bypass the computationally intensive parameter sweep by reusing an existing report. This will trigger the final training phase using the best parameters found in that report.

```bash
# Skip sweep and train using the best parameters from a previous run
heteroage stage1-train \
  --project-root /path/to/project_root \
  --output-dir /path/to/output_stage1 \
  --pc-path /path/to/Global_Healthy_RF_PCs.csv \
  --sweep-file /path/to/output_stage1/Stage1_Sweep_Report.csv
```

### Stage 2: Hallmark Experts Training

Stage 2 focuses on **Residual Learning**. It trains independent "Expert" models (ElasticNetCV) to capture biological variations that the Global Anchor might miss, specifically focusing on defined biological hallmarks.

**Main Tasks:**
* **Hallmark Mapping**: Utilizes the orthogonalized dictionary derived from Stage 1 to group CpG sites.
* **Residual Prediction**: Each expert model is trained to predict the difference (residual) between the chronological age and the Stage 1 global prediction.
* **Leakage-free CV**: Employs robust Cross-Validation to generate Out-of-Fold (OOF) corrections.
* **Expert Serialization**: Fits final models on the full dataset and exports them as deployable artifacts.

```bash
# Execute Stage 2 Training
heteroage stage2-train \
  --project-root /path/to/project_root \
  --output-dir /path/to/output_stage2 \
  --pc-path /path/to/Global_Healthy_RF_PCs.csv \
  --stage1-oof /path/to/output_stage1/Stage1_Global_Anchor_OOF.csv \
  --stage1-dict /path/to/output_stage1/Stage1_Orthogonalized_Hallmark_Dict.json
```
**📂 Key Outputs:**

| File / Directory | Description |
| :--- | :--- |
| `Stage2_Expert_OOF_Corrections.csv` | Hallmark-specific residual corrections used as input features for the Stage 3 Meta-Learner. |
| `Stage2_Expert_Weights.csv` | Detailed coefficients for each hallmark expert model (ElasticNet weights). |
| `Stage2_Performance_Summary.csv` | Regression metrics (R², MAE) for each hallmark, allowing for domain-specific auditing. |
| `artifacts/stage2/` | **Deployable Directory**: Contains the hallmark manifest and all serialized expert artifacts. |

> [!NOTE]
> Stage 2 is critical for capturing domain-specific aging signals (e.g., inflammation, metabolism, or epigenetic drift) that are often masked in a single-stage global model.


### Stage 3: Context-Aware Fusion

The final stage employs a **LightGBM Meta-Learner** to integrate the global baseline with specific hallmark deviations. By incorporating biological context (Tissue, Sex, and PCs), it produces a refined, "heterogeneity-aware" biological age prediction.

**Main Tasks:**
* **Data Integration**: Merges Stage 1 OOF predictions, Stage 2 expert corrections, and Principal Component (PC) context.
* **Robust Validation**: Implements leakage-free **GroupKFold** cross-validation based on `project_id`.
* **Gradient Boosting**: Executes LightGBM training with early stopping to prevent over-fitting.
* **Interpretability**: Generates feature importance (Gain) to reveal which biological drivers contribute most to the final age.
* **Meta-Serialization**: Produces the final deployable meta-learner artifacts.

```bash
# Execute Stage 3 Training
heteroage stage3-train \
  --output-dir /path/to/output_stage3 \
  --stage1-oof /path/to/output_stage1/Stage1_Global_Anchor_OOF.csv \
  --stage2-oof /path/to/output_stage2/Stage2_Expert_OOF_Corrections.csv \
  --pc-path /path/to/Global_Healthy_RF_PCs.csv
```
**📂 Key Outputs:**

| File / Directory | Description |
| :--- | :--- |
| `Stage3_Final_Predictions_OOF.csv` | The final integrated "HeteroAge" predictions for the training cohort (Out-of-Fold). |
| `Stage3_Attention_Importance.csv` | Ranking of feature contributions (Gain/Split) across all Hallmarks and PC contexts. |
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
| **Features** | `{cpg_id}_beta`, `{cpg_id}_chalm` | CpG features suffixed by modality as per trained artifacts. |

**Run Prediction:**
```bash
heteroage pipeline-predict \
  --artifact-dir /path/to/full_artifacts \
  --input /path/to/master.parquet \
  --out /path/to/final_predictions.csv
```

### 🎯 5.3 Granular Stage-Specific Prediction

For modular analysis, debugging, or research into specific biological hallmarks, you can execute individual stages independently using their respective serialized artifacts.

#### Stage 1: Global Anchor Prediction
Generates the baseline biological age based on global population trends.
```bash
heteroage stage1-predict \
  --artifact-dir /path/to/full_artifacts/stage1 \
  --input /path/to/master.parquet \
  --out /path/to/stage1_predictions.csv
```

#### Stage 2: Hallmark Experts Prediction
Calculates independent biological deviations (residuals) for each specific Hallmark (e.g., Inflammation, Metabolism, Epigenetic Drift).

```bash
# Generate hallmark-specific residuals
heteroage stage2-predict \
  --artifact-dir /path/to/full_artifacts/stage2 \
  --input /path/to/master.parquet \
  --out /path/to/stage2_corrections.csv
```

#### Stage 3: Context-Aware Fusion Prediction
Fuses the baseline global trends and hallmark-specific corrections with biological context (Tissue, Sex, PCs) to produce the final "HeteroAge" prediction.

> [!IMPORTANT]
> **Data Dependency**: Stage 3 requires an input table that has already been enriched with Stage 2 outputs. Ensure your input file contains columns suffixed with `_Correction` (e.g., `Inflammation_Correction`) along with the required Principal Components.

```bash
# Generate final fused biological age
heteroage stage3-predict \
  --artifact-dir /path/to/full_artifacts/stage3 \
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
| **Leakage-Free Evaluation** | All CV splits are strictly **grouped by `project_id`** and **stratified by `Tissue`**, preventing over-fitting and ensuring cross-study generalizability. |
| **Deterministic Inference** | Every stage serializes its entire environment—including **Imputers, Scalers, Selected Column Indices**, and model weights—to ensure bit-identical results across different environments. |
| **Robust Data Handling** | Automated handling of missing values: all-NaN columns are zero-filled to maintain matrix dimensions, while partial NaNs are imputed via median values. |
| **Deployment-Ready** | Decoupled architecture allows Stage 3 to run using only serialized artifacts, eliminating the need for retraining or access to the original training cohort. |

---

## 🛠️ Troubleshooting

### ⚠️ High Memory Usage
Processing large-scale DNA methylation matrices can be resource-intensive. 
* **Use Parquet**: Convert large CSV/PKL files to Parquet format for faster I/O and lower memory overhead.
* **Feature Subsetting**: Only load the CpG sites defined in your `Feature_Dict` or trained artifacts.
* **Chunking**: If memory is limited, build master tables in chunks rather than loading the entire beta matrix at once.

### ⚠️ Missing Correction Columns
Stage 3 requires inputs from Stage 2. 
* If you are running Stage 3 independently, ensure your input table contains Hallmark correction columns suffixed with **`_Correction`** (e.g., `Inflammation_Correction`).

### ⚠️ Missing RF_PC Columns
The pipeline relies heavily on Principal Components for tissue-specific normalization.
* Ensure your PC metadata file and inference master tables contain columns matching the **`RF_PC*`** pattern (e.g., `RF_PC1`, `RF_PC2`, etc.).
