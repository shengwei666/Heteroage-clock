"""
heteroage_clock.data.assemble

Data assembly and feature extraction utilities.
Optimized for high memory efficiency by removing redundant copies and 
enforcing float32 precision early in the process.
"""

import pandas as pd
import numpy as np
import gc
import sys
from heteroage_clock.utils.logging import log

def assemble_features(
    cpg_beta: pd.DataFrame, 
    chalm_data: pd.DataFrame, 
    camda_data: pd.DataFrame, 
    pc_data: pd.DataFrame, 
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Assembles various modalities into a single dataframe.
    Memory Strategy: 
    1. Removed redundant df.copy() calls.
    2. Immediate downcasting to float32.
    3. Manual garbage collection between merges.
    """
    # 1. Prepare Metadata (Use the reference, only copy if necessary for indexing)
    df_base = metadata
    if 'sample_id' not in df_base.columns:
        if df_base.index.name == 'sample_id':
            df_base = df_base.reset_index()
    df_base['sample_id'] = df_base['sample_id'].astype(str)
    
    # 2. Define modalities to iterate through
    modalities = [
        ('beta', cpg_beta, '_beta'),
        ('chalm', chalm_data, '_chalm'),
        ('camda', camda_data, '_camda'),
        ('pc', pc_data, '') 
    ]

    log(f"Assembling features for {len(df_base)} samples...")
    meta_cols = {'sample_id', 'project_id', 'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy', 'Sex_encoded'}

    for name, df_mod, suffix in modalities:
        if df_mod is None or df_mod.empty: 
            continue
        if df_mod is metadata: 
            continue

        # Process each modality dataframe
        if 'sample_id' not in df_mod.columns:
            if df_mod.index.name == 'sample_id': 
                df_mod = df_mod.reset_index()
        df_mod['sample_id'] = df_mod['sample_id'].astype(str)

        # Auto-Suffixing features to avoid collisions
        if suffix:
            rename_map = {}
            for col in df_mod.columns:
                if col not in meta_cols and not col.endswith(suffix):
                    rename_map[col] = f"{col}{suffix}"
            if rename_map:
                df_mod.rename(columns=rename_map, inplace=True)

        # Force float32 precision to save 50% RAM
        f64_cols = df_mod.select_dtypes(include=['float64']).columns
        if len(f64_cols) > 0:
            df_mod[f64_cols] = df_mod[f64_cols].astype(np.float32)

        # Drop overlapping columns except sample_id before merge
        existing_cols = set(df_base.columns)
        new_cols = set(df_mod.columns)
        overlap = list(existing_cols.intersection(new_cols) - {'sample_id'})
        if overlap:
            df_mod.drop(columns=overlap, inplace=True)

        # Perform inner merge
        df_base = pd.merge(df_base, df_mod, on='sample_id', how='inner')
        
        # Immediate cleanup of temporary pointers
        gc.collect()

    # Final cleanup of data types
    if 'Age' in df_base.columns and 'age' not in df_base.columns:
        df_base.rename(columns={'Age': 'age'}, inplace=True)
    
    final_f64 = df_base.select_dtypes(include=['float64']).columns
    if len(final_f64) > 0:
        df_base[final_f64] = df_base[final_f64].astype(np.float32)

    return df_base

def filter_and_impute(df: pd.DataFrame, features_to_keep: list = None, impute_strategy: str = 'median') -> pd.DataFrame:
    """
    Filters features and imputes missing values using chunked processing to prevent OOM.
    """
    # 1. Feature Filtering: Only keep what is required
    if features_to_keep:
        meta_cols = ['sample_id', 'age', 'project_id', 'Tissue', 'Sex', 'Is_Healthy', 'Sex_encoded']
        existing_meta = [c for c in meta_cols if c in df.columns]
        valid_features = [f for f in features_to_keep if f in df.columns]
        df_filtered = df[list(set(existing_meta + valid_features))]
    else:
        df_filtered = df 

    if df_filtered.empty: 
        return df_filtered

    # 2. Identify Numeric Columns for Imputation
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    cols_to_impute = [c for c in numeric_cols if c not in ['age', 'sample_id', 'Sex_encoded', 'Age', 'project_id']]

    if not cols_to_impute:
        return df_filtered

    log(f"Starting chunked imputation for {len(cols_to_impute)} columns (Chunk Size: 1000)...")
    gc.collect()

    # 3. Chunked In-place Imputation
    chunk_size = 1000
    total_cols = len(cols_to_impute)
    
    for i in range(0, total_cols, chunk_size):
        chunk = cols_to_impute[i:i+chunk_size]
        try:
            # Calculate and fill per chunk to minimize RAM peak
            if impute_strategy == 'median':
                fill_vals = df_filtered[chunk].astype(np.float32).median()
            elif impute_strategy == 'mean':
                fill_vals = df_filtered[chunk].astype(np.float32).mean()
            else:
                fill_vals = 0.0
                
            df_filtered[chunk] = df_filtered[chunk].fillna(fill_vals)
            
            # Fallback for columns containing only NaNs
            if df_filtered[chunk].isna().any().any():
                 df_filtered[chunk] = df_filtered[chunk].fillna(0.0)
        except Exception as e:
            log(f"Warning: Error processing imputation chunk at index {i}: {e}")
             
        if i % 10000 == 0:
            log(f"   Processed {i}/{total_cols} columns...")
            sys.stdout.flush()
            gc.collect()

    log("Data imputation completed successfully.")
    return df_filtered