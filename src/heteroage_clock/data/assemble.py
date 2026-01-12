"""
heteroage_clock.data.assemble

Data assembly and feature extraction utilities.
Updates:
- [Critical Fix] REMOVED df.copy() to prevent memory doubling (In-place operation).
- [Critical Fix] Reduced chunk size to 1000 for extreme safety.
- Auto-suffixing to prevent data loss.
- Auto-downcast to float32.
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
    Assembles various modalities into a single dataframe with AUTO-SUFFIXING.
    """
    # 1. Prepare Metadata
    df_base = metadata.copy()
    if 'sample_id' not in df_base.columns:
        if df_base.index.name == 'sample_id':
            df_base = df_base.reset_index()
    df_base['sample_id'] = df_base['sample_id'].astype(str)
    
    # 2. Define modalities
    modalities = [
        ('beta', cpg_beta, '_beta'),
        ('chalm', chalm_data, '_chalm'),
        ('camda', camda_data, '_camda'),
        ('pc', pc_data, '') 
    ]

    log(f"Assembling features for {len(df_base)} samples...")
    meta_cols = {'sample_id', 'project_id', 'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy', 'Sex_encoded'}

    for name, df_mod, suffix in modalities:
        if df_mod is None or df_mod.empty: continue
        if df_mod is metadata: continue

        df_m = df_mod.copy()
        if 'sample_id' not in df_m.columns:
            if df_m.index.name == 'sample_id': df_m = df_m.reset_index()
        df_m['sample_id'] = df_m['sample_id'].astype(str)

        # Auto-Suffixing
        if suffix:
            rename_map = {}
            for col in df_m.columns:
                if col not in meta_cols and not col.endswith(suffix):
                    rename_map[col] = f"{col}{suffix}"
            if rename_map:
                log(f"  > {name}: Adding suffix '{suffix}' to {len(rename_map)} columns...")
                df_m.rename(columns=rename_map, inplace=True)

        # Force float32
        float64_cols = df_m.select_dtypes(include=['float64']).columns
        if len(float64_cols) > 0:
            df_m[float64_cols] = df_m[float64_cols].astype(np.float32)

        # Robust Deduplication
        existing_cols = set(df_base.columns)
        new_cols = set(df_m.columns)
        cols_overlap = list(existing_cols.intersection(new_cols) - {'sample_id'})
        if cols_overlap:
            df_m.drop(columns=cols_overlap, inplace=True)

        # Merge
        before_len = len(df_base)
        df_base = pd.merge(df_base, df_m, on='sample_id', how='inner')
        after_len = len(df_base)
        if after_len < before_len:
            log(f"  > Merging {name}: dropped {before_len - after_len} samples (common: {after_len})")

    # Cleanup
    if 'Age' in df_base.columns and 'age' not in df_base.columns:
        df_base.rename(columns={'Age': 'age'}, inplace=True)
    
    final_f64 = df_base.select_dtypes(include=['float64']).columns
    if len(final_f64) > 0:
        df_base[final_f64] = df_base[final_f64].astype(np.float32)

    return df_base

def filter_and_impute(df: pd.DataFrame, features_to_keep: list = None, impute_strategy: str = 'median') -> pd.DataFrame:
    """
    Filters features and imputes missing values using CHUNKED IN-PLACE processing.
    """
    # 1. Feature Filtering
    if features_to_keep:
        meta_cols = ['sample_id', 'age', 'project_id', 'Tissue', 'Sex', 'Is_Healthy', 'Sex_encoded']
        existing_meta = [c for c in meta_cols if c in df.columns]
        valid_features = [f for f in features_to_keep if f in df.columns]
        cols_final = list(set(existing_meta + valid_features))
        
        # Here we assume df can be subsetted. 
        # Ideally, we modify df in place, but subsetting creates a copy anyway.
        # However, filtering REDUCES size, so this copy is safe/beneficial.
        df_filtered = df[cols_final] # Not .copy() explicit, pandas decides view/copy
        # Force garbage collection of the old 'df' reference if possible from caller side?
        # We can't clear caller's variable, but we return a new one.
    else:
        # [CRITICAL CHANGE] Do NOT use df.copy() here. Use reference.
        # We are modifying in-place to save 50GB+ RAM.
        df_filtered = df 

    if df_filtered.empty: return df_filtered

    # 2. Identify Numeric Columns
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    cols_to_impute = [c for c in numeric_cols if c not in ['age', 'sample_id', 'Sex_encoded', 'Age', 'project_id']]

    if not cols_to_impute:
        return df_filtered

    log(f"  > Starting Chunked Imputation for {len(cols_to_impute)} columns (Chunk=1000)...")
    
    # Explicit GC before the big loop to clear any fragmentation
    gc.collect()

    # 4. Chunked Imputation
    # Reduced chunk size to 1000 to be extremely safe against allocation overhead
    chunk_size = 1000
    total_cols = len(cols_to_impute)
    
    for i in range(0, total_cols, chunk_size):
        chunk = cols_to_impute[i:i+chunk_size]
        
        try:
            # Calculate stats for just this chunk
            if impute_strategy == 'median':
                fill_vals = df_filtered[chunk].astype(np.float32).median()
            elif impute_strategy == 'mean':
                fill_vals = df_filtered[chunk].astype(np.float32).mean()
            else:
                fill_vals = 0.0
                
            # Fill NA in place for this chunk
            df_filtered[chunk] = df_filtered[chunk].fillna(fill_vals)
            
            # Fallback for all-NaN columns in this chunk (fill with 0)
            if df_filtered[chunk].isna().any().any():
                 df_filtered[chunk] = df_filtered[chunk].fillna(0.0)
        except Exception as e:
            log(f"Warning: Error processing chunk {i}: {e}")
             
        # Explicit GC to be safe
        if i % 10000 == 0:
            log(f"    - Imputed {i}/{total_cols} columns...")
            sys.stdout.flush() # Force log output immediately
            gc.collect()

    log("  > Imputation completed.")
    return df_filtered