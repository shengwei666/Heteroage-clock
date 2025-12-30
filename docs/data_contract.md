# Data Contract for heteroage-clock

## Overview

The **data contract** defines the structure and format of the data used in the `heteroage-clock` pipeline. It ensures that all stages of the pipeline (Stage 1, Stage 2, and Stage 3) receive data in the expected format, with all required fields available for processing. This document outlines the required and optional columns for both training and inference, as well as any data preprocessing steps that must be followed.

---

## 1. Input Data Format

### 1.1. Training Data

Training data for the pipeline consists of the following core components:

- **Modality PKL Files**: These contain the biological data associated with CpG sites (e.g., Beta, Chalm, Camda).
- **PC CSV File**: This contains the Principal Components (PCs) used for context-aware features in the pipeline.
- **Hallmark Dictionary**: This file maps hallmark names to their associated CpG sites.

#### 1.1.1. Modality PKL Files

Each modality (Beta, Chalm, Camda) must be provided as a `.pkl` file. Each file should contain the following columns:

- `sample_id`: Unique identifier for each sample.
- `project_id`: Identifier for the project that the sample belongs to.
- `Tissue`: The tissue type for each sample.
- `Age`: The age of the sample (used for training and evaluation).
- `Sex`: Gender of the sample (used for context-aware features).
- `Is_Healthy`: A boolean flag indicating whether the sample is healthy (`True` or `False`).
- **CpG columns**: For each CpG site, there should be a corresponding column containing the methylation data for that site. Each column name should include the CpG ID (e.g., `cg00000029`).

#### Example:
```csv
sample_id,project_id,Tissue,Age,Sex,Is_Healthy,cg00000029,cg00000031,cg00000032
sample_1,proj_1,Brain,50,F,True,0.23,0.34,0.45
sample_2,proj_1,Heart,45,M,True,0.12,0.56,0.33
```

#### 1.1.2. PC CSV File
The PC CSV file contains the following required columns:

- `sample_id`: Unique identifier for each sample.
- `Tissue`:The tissue type for each sample.
- **RF_PC columns**: These are the principal components generated from dimensionality reduction techniques. Each column name should start with `RF_PC` (e.g., `RF_PC1`, `RF_PC2`, etc.).

#### Example:
```csv
sample_id,Tissue,RF_PC1,RF_PC2,RF_PC3
sample_1,Brain,0.12,-0.45,0.78
sample_2,Heart,-0.56,0.23,0.90
```

---

## 2. Inference Data Format
For inference, the data structure required is similar to the training data but may differ slightly in the available columns.

### 2.1. Master Table
The **master table** is a CSV or Parquet file used as input for inference. It must contain the following columns:

- `sample_id`: Unique identifier for each sample.
- `Tissue`: The tissue type for each sample.
- `Sex`: Gender of the sample.
- **RF_PC columns**: Must include all the principal components used during training (e.g., `RF_PC1`, `RF_PC2`, etc.).
- **CpG columns**: Must include CpG features used during training. These columns should be named `{cpg_id}_beta`, `{cpg_id}_chalm`, `{cpg_id}_camda`.

**Required Columns**:
- `sample_id`
- `Tissue`
- `Sex`
- **RF_PC columns**
- **CpG columns**: `{cpg_id}_beta`, `{cpg_id}_chalm`, `{cpg_id}_camda` (for each CpG used in the model)

Example:
```csv
sample_id,Tissue,Sex,RF_PC1,RF_PC2,RF_PC3,cg00000029_beta,cg00000031_chalm,cg00000032_camda
sample_1,Brain,F,0.12,-0.45,0.78,0.23,0.34,0.45
sample_2,Heart,M,-0.56,0.23,0.90,0.12,0.56,0.33
```

**Optional Columns**:
- `Age`: If present, the heteroage-clock pipeline will output residuals during inference and allow for the evaluation of predictions against true biological age.

Example:
```csv
sample_id,Tissue,Sex,Age,RF_PC1,RF_PC2,RF_PC3,cg00000029_beta,cg00000031_chalm,cg00000032_camda
sample_1,Brain,F,50,0.12,-0.45,0.78,0.23,0.34,0.45
sample_2,Heart,M,45,-0.56,0.23,0.90,0.12,0.56,0.33
```

---

## 3. Data Preprocessing and Alignment
To ensure that data is correctly processed and aligned across all stages of the pipeline, the following steps must be followed:
- **Missing Data Handling**: If any columns are missing from the input data, they will be filled with NaN values. The pipeline automatically imputes missing data using median imputation during preprocessing.
- **Zero Filling for All-NaN Columns**: Columns that are entirely NaN during training or inference will be filled with zeros to maintain alignment across data.
- **Feature Alignment**: During training, only the features that are present in both the training and inference datasets will be used. Any missing features in the inference data will be filled with zeros and imputed accordingly.
- **Scaling**: All features used in the pipeline will be scaled using standard scaling (mean = 0, standard deviation = 1) during preprocessing.

---

## 4. Data Contract Summary
The data contract ensures that the following requirements are met for both training and inference:
- **Sample-level identification**: Every sample must have a unique `sample_id`.
- **Tissue type**: Each sample must include a `Tissue` column, used for stratification and grouping.
- **Biological data (CpGs)**: Each sample must contain methylation data for various CpG sites, labeled by CpG ID.
- **Context features (PCs)**: Principal component features must be provided to contextualize the data for the pipeline.
- **Consistency in data columns**: The columns provided in both training and inference datasets must be consistent to ensure proper feature alignment.

## 5. Troubleshooting Data Issues
### 5.1. Missing Columns
If certain expected columns are missing from the input dataset, the pipeline will either:

- Raise an error if critical columns (such as `sample_id`, `Tissue`, or `Age`) are missing, or
- Fill missing columns with NaN values for imputation.

### 5.2. Incorrect Column Names

Ensure that all columns follow the correct naming conventions:

- **CpG columns** must follow the format `{cpg_id}_beta`, `{cpg_id}_chalm`, `{cpg_id}_camda`.
- **RF_PC columns** must start with `RF_PC` followed by the component number (e.g., `RF_PC1`, `RF_PC2`, ...).

### 5.3. Inconsistent Data Formats

Ensure that the data is consistently formatted:

- For CSV inputs, ensure no extra spaces or special characters in column names.
- For Parquet files, ensure they conform to the expected schema with correct column types.

## 6. Conclusion

This data contract defines the required data structure and preprocessing steps necessary for the `heteroage-clock` pipeline. By adhering to this contract, users can ensure that the data flows smoothly through the pipeline and that predictions are accurate and reproducible.

For any additional questions or clarifications on data format, please refer to the Repository's Documentation.