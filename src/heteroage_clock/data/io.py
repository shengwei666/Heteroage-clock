"""
heteroage_clock.data.assemble

Data assembly and feature extraction utilities for combining multiple modalities
and covariates into a single dataframe that is ready for model training.

The module includes functionality to load and combine:
- CpG beta values
- CHALM data
- CAMDA data
- Principal Component features (RF_PC)
- Covariates like Sex, Tissue, and Age

It ensures that all required features are properly aligned for model consumption.
"""

import pandas as pd
import numpy as np


def assemble_features(cpg_beta: pd.DataFrame, chalm_data: pd.DataFrame, camda_data: pd.DataFrame, pc_data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Assembles various modalities and covariates into a single feature dataframe for model training.

    Args:
        cpg_beta (pd.DataFrame): CpG beta values.
        chalm_data (pd.DataFrame): CHALM data.
        camda_data (pd.DataFrame): CAMDA data.
        pc_data (pd.DataFrame): Principal component data.
        metadata (pd.DataFrame): Metadata containing sample information (e.g., Tissue, Sex, Age).

    Returns:
        pd.DataFrame: The combined feature set.
    """
    # Merge all modality data into a single dataframe
    df_combined = metadata.merge(cpg_beta, left_on='sample_id', right_index=True, how='left')
    df_combined = df_combined.merge(chalm_data, left_on='sample_id', right_index=True, how='left')
    df_combined = df_combined.merge(camda_data, left_on='sample_id', right_index=True, how='left')
    df_combined = df_combined.merge(pc_data, left_on='sample_id', right_index=True, how='left')

    return df_combined


def filter_and_impute(df: pd.DataFrame, features_to_keep: list, impute_strategy: str = 'median') -> pd.DataFrame:
    """
    Filters the dataframe to keep only the relevant features and imputes missing values.

    Args:
        df (pd.DataFrame): The dataframe to filter and impute.
        features_to_keep (list): List of features to keep in the dataframe.
        impute_strategy (str): The strategy for imputing missing values. Default is 'median'.

    Returns:
        pd.DataFrame: The filtered and imputed dataframe.
    """
    df_filtered = df[features_to_keep]

    # Impute missing values
    if impute_strategy == 'median':
        df_filtered = df_filtered.fillna(df_filtered.median())
    elif impute_strategy == 'mean':
        df_filtered = df_filtered.fillna(df_filtered.mean())
    else:
        df_filtered = df_filtered.fillna(0)  # Fallback to zero imputation

    return df_filtered
