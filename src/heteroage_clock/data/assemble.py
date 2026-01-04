"""
heteroage_clock.data.assemble

Data assembly and feature extraction utilities.
Includes robust handling for all-NaN columns to support Full-Set training.
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
    [Fixed]: Robustly handles metadata duplication to prevent 'project_id_x' errors.
    """
    # 1. Prepare Metadata (Base DataFrame)
    # We use the provided metadata df (usually cpg_beta) as the anchor.
    meta_candidates = ['sample_id', 'age', 'Age', 'sex', 'Sex', 'tissue', 'Tissue', 'project_id', 'is_healthy', 'Is_Healthy']
    
    df_base = metadata.copy()
    if 'sample_id' not in df_base.columns:
        if df_base.index.name == 'sample_id':
            df_base = df_base.reset_index()
        else:
            raise ValueError("Metadata dataframe missing 'sample_id' column or index.")
    
    # 2. Define modalities to merge
    # Note: We skip 'beta' if it is the same object as metadata to save time/memory,
    # but if they are different, we process it.
    modalities = [
        ('beta', cpg_beta),
        ('chalm', chalm_data),
        ('camda', camda_data),
        ('pc', pc_data)
    ]

    log(f"Assembling features for {len(df_base)} samples...")

    for name, df_mod in modalities:
        if df_mod is None or df_mod.empty:
            continue
            
        # Optimization: If this modality IS the metadata object, we don't need to merge it again
        # provided we already took all columns. But to be safe, we usually merge.
        # If it's the exact same object, merging inner on sample_id is redundant but harmless 
        # IF we handle columns correctly.
        if df_mod is metadata:
             # If we initialized df_base FROM metadata, we already have these features.
             # We can skip merging metadata with itself.
             continue

        df_m = df_mod.copy()
        if 'sample_id' not in df_m.columns:
            if df_m.index.name == 'sample_id':
                df_m = df_m.reset_index()
            else:
                log(f"Warning: Modality {name} missing sample_id. Skipping.")
                continue
            
        # --- [CRITICAL FIX] Robust Column Deduplication ---
        # Before merging, check for ANY columns that already exist in df_base.
        # We MUST drop them from df_m to prevent suffix creation (_x, _y).
        # We only keep 'sample_id' for the merge key.
        
        existing_cols = set(df_base.columns)
        new_cols = set(df_m.columns)
        
        # Calculate intersection excluding merge key
        cols_overlap = list(existing_cols.intersection(new_cols) - {'sample_id'})
        
        if cols_overlap:
            # log(f"  > {name}: Dropping {len(cols_overlap)} duplicate columns (e.g., {cols_overlap[:3]}...)")
            df_m = df_m.drop(columns=cols_overlap)

        # Merge (Inner Join)
        before_len = len(df_base)
        df_base = pd.merge(df_base, df_m, on='sample_id', how='inner')
        after_len = len(df_base)
        
        if after_len < before_len:
            log(f"  > Merging {name}: dropped {before_len - after_len} samples (common: {after_len})")

    # Final cleanup: Standardize 'age' column name if needed
    if 'Age' in df_base.columns and 'age' not in df_base.columns:
        df_base.rename(columns={'Age': 'age'}, inplace=True)

    return df_base


def filter_and_impute(df: pd.DataFrame, features_to_keep: list = None, impute_strategy: str = 'median') -> pd.DataFrame:
    """
    Filters features and imputes missing values.
    [CRITICAL]: Includes robust handling for all-NaN columns for Full-Set training.
    """
    # 1. Feature Filtering
    if features_to_keep:
        meta_cols = ['sample_id', 'age', 'project_id', 'Tissue', 'Sex', 'Is_Healthy', 'Sex_encoded']
        existing_meta = [c for c in meta_cols if c in df.columns]
        valid_features = [f for f in features_to_keep if f in df.columns]
        cols_final = list(set(existing_meta + valid_features))
        df_filtered = df[cols_final].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return df_filtered

    # --- [NEW] All-NaN Column Guard ---
    # Essential for 240k feature sets where some modalities might be empty for some subsets
    numeric_df = df_filtered.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        all_nan_cols = numeric_df.columns[numeric_df.isna().all()].tolist()
        
        if all_nan_cols:
            log(f"  > Warning: Found {len(all_nan_cols)} all-NaN columns. Filling with 0.0.")
            df_filtered[all_nan_cols] = 0.0

    # 2. Regular Imputation
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    cols_to_impute = [c for c in numeric_cols if c not in ['age', 'sample_id', 'Sex_encoded']]

    if not cols_to_impute:
        return df_filtered

    if impute_strategy == 'median':
        df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(df_filtered[cols_to_impute].median())
    elif impute_strategy == 'mean':
        df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(df_filtered[cols_to_impute].mean())
    else:
        df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(0)
    
    # Final fallback for lingering NaNs
    if df_filtered[cols_to_impute].isna().any().any():
         df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(0)

    return df_filtered