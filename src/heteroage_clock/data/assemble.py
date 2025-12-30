"""
heteroage_clock.data.assemble

Data assembly and feature extraction utilities for combining multiple modalities
and covariates into a single dataframe that is ready for model training.
"""

import pandas as pd
import numpy as np
from heteroage_clock.utils.logging import log

def assemble_features(
    cpg_beta: pd.DataFrame, 
    chalm_data: pd.DataFrame, 
    camda_data: pd.DataFrame, 
    pc_data: pd.DataFrame, 
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Assembles various modalities and covariates into a single feature dataframe.
    
    Args:
        cpg_beta: CpG beta values matrix.
        chalm_data: CHALM values matrix.
        camda_data: CAMDA values matrix.
        pc_data: Principal Components.
        metadata: DataFrame containing sample info. 
                  (Can be the same object as cpg_beta, logic will extract only metadata cols).

    Returns:
        pd.DataFrame: The combined DataFrame with sample_id alignment.
    """
    # 1. Prepare Metadata
    # If the metadata DF is actually a feature matrix (e.g. cpg_beta), 
    # we only want to keep the non-feature columns to avoid duplication.
    # Define standard metadata columns to look for
    meta_candidates = ['sample_id', 'age', 'Age', 'sex', 'Sex', 'tissue', 'Tissue', 'project_id', 'is_healthy', 'Is_Healthy']
    
    # Ensure sample_id is a column
    df_meta = metadata.copy()
    if 'sample_id' not in df_meta.columns:
        df_meta.index.name = 'sample_id'
        df_meta = df_meta.reset_index()
    
    # Filter to keep only metadata columns
    # We keep 'sample_id' plus any column that matches our candidates or is NOT a float feature
    # A simple heuristic: Keep known metadata columns + sample_id
    cols_to_keep = [c for c in df_meta.columns if c in meta_candidates or c.lower() in [m.lower() for m in meta_candidates]]
    if 'sample_id' not in cols_to_keep:
        cols_to_keep.append('sample_id')
        
    df_base = df_meta[cols_to_keep].copy()
    
    # Standardize column names (lowercase 'age', 'sex' for consistency if needed, but let's keep original for now)
    if 'Age' in df_base.columns and 'age' not in df_base.columns:
        df_base.rename(columns={'Age': 'age'}, inplace=True)

    # 2. Define modalities to merge
    # We use a list of (name, df) tuples
    modalities = [
        ('beta', cpg_beta),
        ('chalm', chalm_data),
        ('camda', camda_data),
        ('pc', pc_data)
    ]

    log(f"Assembling features for {len(df_base)} samples...")

    for name, df_mod in modalities:
        if df_mod is None:
            continue
            
        # Prepare modality df
        df_m = df_mod.copy()
        if 'sample_id' not in df_m.columns:
            df_m.index.name = 'sample_id'
            df_m = df_m.reset_index()
            
        # If this modality IS the metadata source (same object), drop metadata cols to avoid suffix duplication
        # Exception: keep sample_id for merging
        if df_mod is metadata:
            # Drop columns that are already in df_base (except sample_id)
            cols_overlap = [c for c in df_m.columns if c in df_base.columns and c != 'sample_id']
            df_m = df_m.drop(columns=cols_overlap)

        # Merge
        # Inner join ensures we only keep samples present in ALL required modalities (and PCs)
        # This is crucial for valid training data
        before_len = len(df_base)
        df_base = pd.merge(df_base, df_m, on='sample_id', how='inner')
        after_len = len(df_base)
        
        if after_len < before_len:
            log(f"  > Merging {name}: dropped {before_len - after_len} samples (common samples: {after_len})")

    return df_base


def filter_and_impute(df: pd.DataFrame, features_to_keep: list = None, impute_strategy: str = 'median') -> pd.DataFrame:
    """
    Filters the dataframe features and imputes missing values.
    
    Args:
        df: Input DataFrame.
        features_to_keep: List of specific feature columns to retain. 
                          If None, keeps all columns.
        impute_strategy: 'median', 'mean', or 'zero'.
    """
    # 1. Feature Filtering
    if features_to_keep:
        # Always keep metadata
        meta_cols = ['sample_id', 'age', 'project_id', 'Tissue', 'Sex', 'Is_Healthy', 'Sex_encoded']
        # Find which metadata cols exist in current df
        existing_meta = [c for c in meta_cols if c in df.columns]
        
        # Determine features that actually exist in df
        valid_features = [f for f in features_to_keep if f in df.columns]
        
        # Combine
        cols_final = list(set(existing_meta + valid_features))
        df_filtered = df[cols_final].copy()
    else:
        df_filtered = df.copy()

    # 2. Imputation
    # Only impute numeric columns, exclude ID/Metadata
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    # Exclude Age/Target from imputation if they are numeric
    cols_to_impute = [c for c in numeric_cols if c not in ['age', 'sample_id', 'Sex_encoded']]

    if cols_to_impute.empty:
        return df_filtered

    if impute_strategy == 'median':
        df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(df_filtered[cols_to_impute].median())
    elif impute_strategy == 'mean':
        df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(df_filtered[cols_to_impute].mean())
    else:
        df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(0)

    return df_filtered