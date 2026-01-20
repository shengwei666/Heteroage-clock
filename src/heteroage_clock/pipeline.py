"""
heteroage_clock.pipeline

End-to-end Prediction Pipeline.
Updates:
- [Critical] Implements 'InferenceDataAssembler' to build the mandatory Memmap for Stage 1 & 2.
- [Feature] Orchestrates the new explicit-path inference flow.
- [Optimization] Memory-safe handling of multi-modal inputs via chunked processing.
"""

import os
import shutil
import gc
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional

# Import prediction functions from specific stages
from heteroage_clock.stages.stage1 import predict_stage1
from heteroage_clock.stages.stage2 import predict_stage2
from heteroage_clock.stages.stage3 import predict_stage3
from heteroage_clock.artifacts.stage1 import Stage1Artifact
from heteroage_clock.utils.logging import log

class InferenceDataAssembler:
    """
    Responsible for aligning raw input files to the model's feature space
    and creating a memory-mapped binary file for efficient inference.
    """
    def __init__(self, output_dir: str, feature_names: List[str]):
        self.output_dir = output_dir
        self.feature_names = feature_names
        # Create a fast lookup map: Feature Name -> Column Index
        self.feat_map = {f: i for i, f in enumerate(feature_names)}
        self.n_features = len(feature_names)
        self.mmap_path = os.path.join(output_dir, "X_inference.dat")
        
    def assemble(self, main_df: pd.DataFrame, 
                 chalm_path: Optional[str] = None, 
                 camda_path: Optional[str] = None, 
                 pc_path: Optional[str] = None) -> str:
        """
        Builds X matrix on disk.
        Args:
            main_df: The primary input dataframe (Beta values + Metadata).
            chalm_path: Path to Chalm pickle/csv.
            camda_path: Path to Camda pickle/csv.
            pc_path: Path to PC CSV.
        Returns:
            Path to the generated Memmap file.
        """
        n_samples = len(main_df)
        log(f"Assembling Inference Matrix: {n_samples} samples x {self.n_features} features")
        
        # 1. Create Writeable Memmap (Initialize with Zeros)
        # Zeros effectively act as mean-imputation for StandardScaler centered data.
        X_mmap = np.memmap(self.mmap_path, dtype='float32', mode='w+', shape=(n_samples, self.n_features))
        
        # 2. Process Modalities
        # We enforce the standard suffixes used in training: _beta, _chalm, _camda
        
        # --- A. Process Main Input (Beta) ---
        log("  > Merging Beta modality (from main input)...")
        self._fill_modality(X_mmap, main_df, suffix="_beta")
        
        # --- B. Process Chalm ---
        if chalm_path and os.path.exists(chalm_path):
            self._process_file(X_mmap, chalm_path, main_df['sample_id'], suffix="_chalm")
            
        # --- C. Process Camda ---
        if camda_path and os.path.exists(camda_path):
            self._process_file(X_mmap, camda_path, main_df['sample_id'], suffix="_camda")
            
        # --- D. Process PCs ---
        if pc_path and os.path.exists(pc_path):
            # PCs usually have exact names (no suffix needed if names match training feature list)
            self._process_file(X_mmap, pc_path, main_df['sample_id'], suffix="") 
            
        # Flush changes to disk
        X_mmap.flush()
        # Explicitly delete the memmap object to close the file handle
        del X_mmap
        gc.collect()
        
        return self.mmap_path

    def _process_file(self, X_mmap, path, target_sample_ids, suffix):
        """Loads a file, aligns samples, and fills the memmap."""
        log(f"  > Merging modality from {os.path.basename(path)}...")
        
        # Load File
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_pickle(path)
            
        # Standardize ID column
        if 'sample_id' not in df.columns and df.index.name == 'sample_id':
            df = df.reset_index()
            
        if 'sample_id' not in df.columns:
            log(f"Warning: 'sample_id' not found in {os.path.basename(path)}. Skipping merge.")
            return

        df['sample_id'] = df['sample_id'].astype(str)
        
        # Align rows to main_df order (Left Join / Reindex)
        df = df.set_index('sample_id').reindex(target_sample_ids).reset_index()
        
        self._fill_modality(X_mmap, df, suffix)
        
        del df
        gc.collect()

    def _fill_modality(self, X_mmap, df, suffix):
        """
        Maps columns from df to X_mmap indices based on feature_names and suffix.
        Uses chunked assignment to be memory safe.
        """
        # 1. Map DF columns to Memmap Indices
        col_to_idx = {}
        
        if suffix:
            # Check if "ColName" + "Suffix" exists in our feature list
            for col in df.columns:
                target_name = f"{col}{suffix}"
                if target_name in self.feat_map:
                    col_to_idx[col] = self.feat_map[target_name]
        else:
            # Exact match check
            for col in df.columns:
                if col in self.feat_map:
                    col_to_idx[col] = self.feat_map[col]
                    
        if not col_to_idx:
            return

        # 2. Bulk Assignment in Chunks
        src_cols = list(col_to_idx.keys())
        dst_idxs = list(col_to_idx.values())
        
        n_rows = len(df)
        chunk_size = 5000 # Process 5000 rows at a time
        
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            
            # Extract values: (Chunk_Size, N_Matched_Cols)
            vals = df.iloc[start:end][src_cols].values.astype(np.float32)
            
            # Assign to Memmap: X[rows, specific_cols]
            # Numpy memmap supports basic fancy indexing for assignment
            X_mmap[start:end, dst_idxs] = vals

def predict_pipeline(
    artifact_dir: str,
    input_path: str,
    output_path: str,
    input_chalm: Optional[str] = None,
    input_camda: Optional[str] = None,
    input_pc: Optional[str] = None
):
    """
    Executes the full HeteroAge-Clock inference pipeline.
    1. Assemble Data (Beta + Modalities) -> Memmap
    2. Stage 1 Inference -> OOF/Resid
    3. Stage 2 Inference -> Expert Resids
    4. Stage 3 Inference -> Final Fusion
    """
    log(">>> Starting End-to-End Inference Pipeline")
    
    # 0. Setup Temp Workspace
    # We create a temp folder next to the output file to store intermediate artifacts
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_inference_workspace")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # --- 1. Load Metadata & Init Assembler ---
        log(f"Loading main input: {input_path}")
        if input_path.endswith('.csv'):
            main_df = pd.read_csv(input_path)
        else:
            main_df = pd.read_pickle(input_path)
            
        # Robust ID handling
        if 'sample_id' not in main_df.columns:
            if main_df.index.name == 'sample_id':
                main_df = main_df.reset_index()
            else:
                log("Warning: No 'sample_id' found. Generating sequential IDs.")
                main_df['sample_id'] = [f"sample_{i}" for i in range(len(main_df))]
        
        main_df['sample_id'] = main_df['sample_id'].astype(str)
        
        # Save lightweight meta for stages (only IDs needed for alignment)
        meta_path = os.path.join(temp_dir, "meta.pkl")
        main_df[['sample_id']].to_pickle(meta_path) 
        
        # Load Feature Definition from Stage 1 Artifacts
        # We need to know the exact feature order of the trained model
        s1_artifact = Stage1Artifact(artifact_dir)
        all_features = s1_artifact.load("stage1_features")
        if isinstance(all_features, pd.Index): all_features = all_features.tolist()
        
        # Assemble Matrix (Beta -> X.dat)
        assembler = InferenceDataAssembler(temp_dir, all_features)
        mmap_path = assembler.assemble(main_df, input_chalm, input_camda, input_pc)
        
        # --- 2. Stage 1 Inference ---
        s1_out = os.path.join(temp_dir, "stage1_pred.csv")
        # Call strict signature: (artifact_dir, mmap_path, meta_path, output_path, all_features)
        predict_stage1(artifact_dir, mmap_path, meta_path, s1_out, all_features)
        
        # --- 3. Stage 2 Inference ---
        s2_out = os.path.join(temp_dir, "stage2_pred.csv")
        # Call strict signature: (stage1_dir, stage2_dir, mmap_path, meta_path, output_path, all_features)
        # Note: We assume all artifacts (S1 and S2) are in the same 'artifact_dir' folder
        predict_stage2(artifact_dir, artifact_dir, mmap_path, meta_path, s2_out, all_features)
        
        # --- 4. Stage 3 Inference ---
        # Stage 3 needs a merged input (S1 Predictions + S2 Residuals + PCs)
        
        df_s1 = pd.read_csv(s1_out)
        df_s2 = pd.read_csv(s2_out)
        
        # Merge S1 and S2
        df_merged = pd.merge(df_s1, df_s2, on="sample_id")
        
        # Merge PCs if available (Stage 3 might use them)
        if input_pc and os.path.exists(input_pc):
            df_pc = pd.read_csv(input_pc)
            pc_cols = [c for c in df_pc.columns if c.startswith('RF_PC')]
            
            if 'sample_id' in df_pc.columns:
                df_pc['sample_id'] = df_pc['sample_id'].astype(str)
                df_merged = pd.merge(df_merged, df_pc[['sample_id'] + pc_cols], on="sample_id", how="left")
            else:
                log("Warning: PC file missing 'sample_id'. Skipping PC merge for Stage 3.")

        # Save merged input for Stage 3
        s3_input = os.path.join(temp_dir, "stage3_input.csv")
        df_merged.to_csv(s3_input, index=False)
        
        # Run S3
        predict_stage3(artifact_dir, s3_input, output_path)
        
        log(f"Pipeline Finished. Results saved to: {output_path}")
        
    finally:
        # Cleanup Temp Workspace
        if os.path.exists(temp_dir):
            try:
                gc.collect() # Ensure memmap handles are closed
                shutil.rmtree(temp_dir)
                log("Cleaned up temporary inference files.")
            except Exception as e:
                log(f"Warning: Failed to clean temp dir: {e}")