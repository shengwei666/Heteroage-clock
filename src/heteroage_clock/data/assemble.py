"""
heteroage_clock.data.assemble

Data assembly and feature extraction utilities.
Updates:
- [Critical] Auto-suffixing to prevent data loss during merge (e.g. adding _chalm, _camda).
- [Memory] Auto-downcast to float32.
- Robust handling of all-NaN columns.
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
    Assembles various modalities into a single dataframe with AUTO-SUFFIXING.
    This prevents data loss when different modalities share the same raw column names (e.g. cg IDs).
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
    
    # 2. Define modalities with their expected suffixes
    # Structure: (name, dataframe, suffix_to_add)
    # Note: PCs usually don't need a suffix or have distinct names like RF_PC1
    modalities = [
        ('beta', cpg_beta, '_beta'),
        ('chalm', chalm_data, '_chalm'),
        ('camda', camda_data, '_camda'),
        ('pc', pc_data, '') 
    ]

    log(f"Assembling features for {len(df_base)} samples...")

    # Define metadata columns to exclude from renaming
    meta_cols = {'sample_id', 'project_id', 'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy', 'Sex_encoded'}

    for name, df_mod, suffix in modalities:
        if df_mod is None or df_mod.empty:
            continue
            
        # Optimization: If this modality IS the metadata object itself (e.g. beta passed as metadata),
        # we skip merging, BUT we might still need to rename columns if they lack suffixes.
        # However, usually metadata/base is kept as is. 
        if df_mod is metadata:
             continue

        # Create a working copy
        df_m = df_mod.copy()
        
        # Ensure sample_id exists
        if 'sample_id' not in df_m.columns:
            if df_m.index.name == 'sample_id':
                df_m = df_m.reset_index()
            else:
                log(f"Warning: Modality {name} missing sample_id. Skipping.")
                continue
        
        df_m['sample_id'] = df_m['sample_id'].astype(str)

        # --- [CRITICAL FIX] Auto-Suffixing ---
        # If columns don't already have the suffix, add it!
        if suffix:
            rename_map = {}
            for col in df_m.columns:
                # Rename if it's not a metadata column and doesn't already have the suffix
                if col not in meta_cols and not col.endswith(suffix):
                    rename_map[col] = f"{col}{suffix}"
            
            if rename_map:
                log(f"  > {name}: Adding suffix '{suffix}' to {len(rename_map)} columns (e.g., {list(rename_map.keys())[0]} -> {list(rename_map.values())[0]})...")
                df_m.rename(columns=rename_map, inplace=True)

        # --- [MEMORY OPTIMIZATION] Force float32 ---
        float64_cols = df_m.select_dtypes(include=['float64']).columns
        if len(float64_cols) > 0:
            df_m[float64_cols] = df_m[float64_cols].astype(np.float32)

        # --- Robust Column Deduplication ---
        # Now that we've renamed features, overlaps should only be metadata columns (like project_id, Tissue)
        # We drop them from the incoming dataframe to avoid _x, _y duplication
        existing_cols = set(df_base.columns)
        new_cols = set(df_m.columns)
        cols_overlap = list(existing_cols.intersection(new_cols) - {'sample_id'})
        
        if cols_overlap:
            # log(f"  > Dropping {len(cols_overlap)} overlapping metadata columns from {name} to prevent duplicates.")
            df_m.drop(columns=cols_overlap, inplace=True)

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
    final_f64 = df_base.select_dtypes(include=['float64']).columns
    if len(final_f64) > 0:
        df_base[final_f64] = df_base[final_f64].astype(np.float32)

    return df_base

def filter_and_impute(df: pd.DataFrame, features_to_keep: list = None, impute_strategy: str = 'median') -> pd.DataFrame:
    """
    Filters features and imputes missing values.
    Includes robust handling for all-NaN columns to prevent crashes.
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

    # 2. All-NaN Guard (Critical)
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
    # Note: We calculate median on float32 casted data to save RAM during the calculation
    if impute_strategy == 'median':
        fill_vals = df_filtered[cols_to_impute].astype(np.float32).median()
    elif impute_strategy == 'mean':
        fill_vals = df_filtered[cols_to_impute].astype(np.float32).mean()
    else:
        fill_vals = 0.0
        
    df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(fill_vals)
    
    # Fallback for any lingering NaNs
    if df_filtered[cols_to_impute].isna().any().any():
         df_filtered[cols_to_impute] = df_filtered[cols_to_impute].fillna(0.0)

    return df_filtered