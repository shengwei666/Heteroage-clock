"""
heteroage_clock.data.assemble

Data assembly and feature extraction utilities.
Updates:
- [Memory Optimization]: Auto-downcast to float32 during assembly to reduce merge overhead.
- Robust column deduplication.
- Handling of all-NaN columns for stability.
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
    Includes memory optimizations (float32 casting) and robust merge handling.
    """
    # 1. Prepare Metadata (Base DataFrame)
    # Use copy to avoid modifying original
    df_base = metadata.copy()
    
    # Ensure sample_id is a column
    if 'sample_id' not in df_base.columns:
        if df_base.index.name == 'sample_id':
            df_base = df_base.reset_index()
        else:
            raise ValueError("Metadata dataframe missing 'sample_id' column or index.")
            
    # Normalize sample_id to string to ensure consistent merging
    df_base['sample_id'] = df_base['sample_id'].astype(str)
    
    # 2. Define modalities
    # Order matters: we merge these INTO the metadata
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
            
        # Optimization: If this modality IS the metadata object itself (e.g. beta passed as metadata),
        # skip merging to save time/memory.
        if df_mod is metadata:
             continue

        # Create a working copy
        df_m = df_mod.copy()
        
        # Ensure sample_id exists and is string
        if 'sample_id' not in df_m.columns:
            if df_m.index.name == 'sample_id':
                df_m = df_m.reset_index()
            else:
                log(f"Warning: Modality {name} missing sample_id. Skipping.")
                continue
        
        df_m['sample_id'] = df_m['sample_id'].astype(str)

        # --- [MEMORY OPTIMIZATION] Force float32 ---
        # Identify float64 columns and downcast to float32 BEFORE merge
        # This reduces memory usage of the merged dataframe by 50%
        float64_cols = df_m.select_dtypes(include=['float64']).columns
        if len(float64_cols) > 0:
            # log(f"  > {name}: Downcasting {len(float64_cols)} float64 columns to float32...")
            df_m[float64_cols] = df_m[float64_cols].astype(np.float32)

        # --- Robust Column Deduplication ---
        # Prevent '_x', '_y' suffixes by dropping duplicate columns from the incoming dataframe.
        # We only keep 'sample_id' for the merge key.
        existing_cols = set(df_base.columns)
        new_cols = set(df_m.columns)
        cols_overlap = list(existing_cols.intersection(new_cols) - {'sample_id'})
        
        if cols_overlap:
            df_m = df_m.drop(columns=cols_overlap)

        # Merge (Inner Join)
        before_len = len(df_base)
        df_base = pd.merge(df_base, df_m, on='sample_id', how='inner')
        after_len = len(df_base)
        
        if after_len < before_len:
            log(f"  > Merging {name}: dropped {before_len - after_len} samples (common: {after_len})")

    # Final cleanup: Ensure age column naming standard
    if 'Age' in df_base.columns and 'age' not in df_base.columns:
        df_base.rename(columns={'Age': 'age'}, inplace=True)
        
    # Final safety: Convert any remaining float64 in df_base to float32
    # This ensures the final object passed to Stage 1/2 is memory efficient
    final_f64 = df_base.select_dtypes(include=['float64']).columns
    if len(final_f64) > 0:
        df_base[final_f64] = df_base[final_f64].astype(np.float32)

    return df_base

def filter_and_impute(df: pd.DataFrame, features_to_keep: list = None, impute_strategy: str = 'median') -> pd.DataFrame:
    """
    Filters features and imputes missing values.
    Includes robust handling for all-NaN columns.
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

    # 2. All-NaN Guard (Critical for 240k feature sets)
    # If a column is entirely NaN, fill it with 0.0 immediately to prevent imputation errors
    numeric_df = df_filtered.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        all_nan_cols = numeric_df.columns[numeric_df.isna().all()].tolist()
        if all_nan_cols:
            log(f"  > Warning: Found {len(all_nan_cols)} all-NaN columns. Filling with 0.0.")
            # Use float32 zero
            df_filtered[all_nan_cols] = np.float32(0.0)

    # 3. Imputation
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    cols_to_impute = [c for c in numeric_cols if c not in ['age', 'sample_id', 'Sex_encoded']]

    if not cols_to_impute:
        return df_filtered

    # Compute stats using float32 to stay efficient
    if impute_strategy == 'median':
        fill_vals = df_filtered[cols_to_impute].median()
    elif impute_strategy == 'mean':
        fill_vals = df_filtered[cols_to_impute].mean()
    else:
        fill_vals = 0.0
        
    df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(fill_vals)
    
    # Fallback for any lingering NaNs (e.g., if median itself was NaN)
    if df_filtered[cols_to_impute].isna().any().any():
         df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(0.0)

    return df_filtered