"""
heteroage_clock.pipeline
Coordinating the 3-Stage Training and Inference.
Updated: Extended memory-saving strategies (Selective Loading, Float32, GC) to inference.
"""

import os
import pandas as pd
import numpy as np
import gc
import joblib
from heteroage_clock.utils.logging import log
from heteroage_clock.stages.stage1 import train_stage1, predict_stage1
from heteroage_clock.stages.stage2 import train_stage2, predict_stage2
from heteroage_clock.stages.stage3 import train_stage3, predict_stage3

def _standardize_ids(df):
    if 'sample_id' in df.columns:
        if not df['sample_id'].duplicated().any():
            return df
    
    for col in ['gsm', 'id', 'Unnamed: 0']:
        if col in df.columns and not df[col].duplicated().any():
            df = df.rename(columns={col: 'sample_id'})
            return df

    df = df.copy()
    df = df.reset_index().rename(columns={df.index.name if df.index.name else 'index': 'sample_id'})
    df['sample_id'] = df['sample_id'].astype(str)
    if df['sample_id'].duplicated().any():
        df = df.drop_duplicates(subset=['sample_id'], keep='first')
    return df

def _get_sparse_numeric_features(artifact_dir):
    numeric_features = set()
    meta_keys = {'sample_id', 'project_id', 'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy'}
    
    # Stage 1
    s1_feat_path = os.path.join(artifact_dir, "stage1", "stage1_features.pkl")
    if os.path.exists(s1_feat_path):
        all_s1_feats = joblib.load(s1_feat_path)
        numeric_features.update([f for f in all_s1_feats if f not in meta_keys])

    # Stage 2
    s2_dir = os.path.join(artifact_dir, "stage2")
    if os.path.exists(s2_dir):
        feat_files = [f for f in os.listdir(s2_dir) if "features.joblib" in f or "features.pkl" in f]
        for f_file in feat_files:
            feats = joblib.load(os.path.join(s2_dir, f_file))
            numeric_features.update([f for f in feats if f not in meta_keys])
    
    return sorted(list(numeric_features))

def _merge_to_memmap_sparse(input_path, input_chalm, input_camda, input_pc, output_dir, numeric_features):
    log("--- Data Assembly: Building Sparse Numeric Matrix ---")
    raw_main = pd.read_pickle(input_path) if input_path.endswith('.pkl') else pd.read_csv(input_path)
    main_df = _standardize_ids(raw_main)
    
    sample_ids = main_df['sample_id'].astype(str).values
    X = np.memmap(os.path.join(output_dir, f"inference_{os.getpid()}.mmap"), 
                  dtype='float32', mode='w+', shape=(len(sample_ids), len(numeric_features)))
    X[:] = 0.0 
    
    feat_to_idx = {f: i for i, f in enumerate(numeric_features)}
    
    def fill_modality(path, suffix):
        if not path or not os.path.exists(path): return
        log(f"   Streaming modality: {os.path.basename(path)}")
        data = _standardize_ids(pd.read_pickle(path) if path.endswith('.pkl') else pd.read_csv(path))
        data = data.set_index('sample_id').reindex(sample_ids)
        
        meta_ignore = {'Tissue', 'Age', 'age', 'Sex', 'Is_Healthy', 'sample_id', 'project_id'}
        rename_map = {c: (c if c.startswith('RF_PC') else f"{c}{suffix}") 
                      for c in data.columns if c not in meta_ignore}
        
        data.rename(columns=rename_map, inplace=True)
        active_cols = [c for c in data.columns if c in feat_to_idx]
        if active_cols:
            target_indices = [feat_to_idx[c] for c in active_cols]
            X[:, target_indices] = data[active_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype('float32')
        del data; gc.collect()

    fill_modality(input_path, "_beta")
    fill_modality(input_chalm, "_chalm")
    fill_modality(input_camda, "_camda")
    fill_modality(input_pc, "") 
    X.flush()
    
    # CRITICAL: Preserve 'age' in meta_df for plotting
    meta_path = os.path.join(output_dir, f"meta_{os.getpid()}.pkl")
    meta_keys = ['sample_id', 'Tissue', 'Age', 'age', 'Sex', 'project_id', 'Is_Healthy']
    final_meta_cols = [c for c in meta_keys if c in main_df.columns]
    main_df[final_meta_cols].to_pickle(meta_path)
    
    return X.filename, meta_path

def train_pipeline(output_dir, pc_path, dict_path, beta_path, chalm_path, camda_path, **kwargs):
    """
    Sequentially trains Stage 1, Stage 2, and Stage 3.
    """
    log("=== Pipeline: Starting Full Training Pipeline ===")
    
    s1_dir = os.path.join(output_dir, "stage1")
    train_stage1(
        output_dir=s1_dir, pc_path=pc_path, dict_path=dict_path,
        beta_path=beta_path, chalm_path=chalm_path, camda_path=camda_path, **kwargs
    )
    
    s1_oof = os.path.join(s1_dir, "stage1_oof_predictions.csv")
    s1_dict = os.path.join(s1_dir, "stage1_orthogonalized_dict.joblib")
    
    s2_dir = os.path.join(output_dir, "stage2")
    train_stage2(
        output_dir=s2_dir, stage1_oof_path=s1_oof, stage1_dict_path=s1_dict,
        pc_path=pc_path, beta_path=beta_path, chalm_path=chalm_path, camda_path=camda_path, **kwargs
    )
    
    s2_oof = os.path.join(s2_dir, "stage2_oof_corrections.csv")
    
    s3_dir = os.path.join(output_dir, "stage3")
    train_stage3(
        output_dir=s3_dir, stage1_oof_path=s1_oof, stage2_oof_path=s2_oof, pc_path=pc_path, **kwargs
    )
    log(f"=== Pipeline: Full Training Finished ===")

def predict_pipeline(artifact_dir, input_path, output_path, **kwargs):
    log("=== Pipeline: Starting Full-Output Inference ===")
    mmap_path, meta_path = None, None
    s1_out, s2_out, s3_out = [output_path + f".s{i}.csv" for i in [1, 2, 3]]
    
    try:
        numeric_features = _get_sparse_numeric_features(artifact_dir)
        mmap_path, meta_path = _merge_to_memmap_sparse(input_path, kwargs.get('input_chalm'), 
                                                       kwargs.get('input_camda'), kwargs.get('input_pc'), 
                                                       os.path.dirname(output_path), numeric_features)
        
        # Execute Stages
        predict_stage1(os.path.join(artifact_dir, "stage1"), mmap_path, meta_path, s1_out, numeric_features)
        predict_stage2(os.path.join(artifact_dir, "stage2"), mmap_path, meta_path, s2_out, numeric_features)
        
        # Merge S1 and S2 for S3 input
        df_meta = pd.read_pickle(meta_path)
        df_s1 = pd.read_csv(s1_out)
        df_s2 = pd.read_csv(s2_out)
        
        s3_input_df = pd.merge(df_meta, df_s1, on='sample_id').merge(df_s2, on='sample_id')
        s3_temp_path = output_path + ".s3_in.pkl"
        s3_input_df.to_pickle(s3_temp_path)
        
        predict_stage3(os.path.join(artifact_dir, "stage3"), s3_temp_path, s3_out)
        
        # FINAL GRAND MERGE
        log("--- Finalizing Comprehensive Output ---")
        df_s3 = pd.read_csv(s3_out)
        final_df = pd.merge(s3_input_df, df_s3[['sample_id', 'pred_residual_stage3', 'HeteroAge']], on='sample_id')
        
        # Calculate Stage 1 Residual
        age_col = 'age' if 'age' in final_df.columns else 'Age'
        if age_col in final_df.columns:
            final_df['pred_residual_stage1'] = final_df[age_col] - final_df['pred_age_stage1']
        
        final_df.to_csv(output_path, index=False)
        log(f"=== Success: Full results saved to {output_path} ===")
        
    except Exception as e:
        log(f"❌ Failed: {e}"); raise e
    finally:
        for f in [s1_out, s2_out, s3_out, s3_temp_path, mmap_path, meta_path]:
            if f and os.path.exists(f): os.remove(f)
        gc.collect()