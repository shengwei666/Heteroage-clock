import os
import shutil
import argparse
import pandas as pd
from heteroage_clock.stages.stage1 import predict_stage1
from heteroage_clock.stages.stage2 import predict_stage2
from heteroage_clock.stages.stage3 import predict_stage3
from heteroage_clock.utils.logging import log

def main():
    parser = argparse.ArgumentParser(description="HeteroAge Full Inference Pipeline")
    
    # 路径参数
    parser.add_argument("--artifact-dir", required=True, help="Root directory containing stage1/stage2/stage3 subfolders")
    parser.add_argument("--beta", required=True, help="Path to Beta matrix pickle")
    parser.add_argument("--chalm", default=None, help="Path to Chalm matrix pickle")
    parser.add_argument("--camda", default=None, help="Path to Camda matrix pickle")
    parser.add_argument("--pc", default=None, help="Path to PC csv file")
    parser.add_argument("--output", required=True, help="Final output CSV path")
    parser.add_argument("--temp-dir", default="./temp_inference", help="Temporary directory for intermediate files")
    
    args = parser.parse_args()
    
    # 0. 准备工作
    s1_dir = os.path.join(args.artifact_dir, "stage1")
    s2_dir = os.path.join(args.artifact_dir, "stage2")
    s3_dir = os.path.join(args.artifact_dir, "stage3")
    
    os.makedirs(args.temp_dir, exist_ok=True)
    project_id = os.path.basename(args.output).replace("_Prediction.csv", "")
    log(f"🚀 Starting Inference for: {project_id}")

    try:
        # ======================================================================
        # 1. Stage 1 Inference (Global Clock)
        # ======================================================================
        log("--- Stage 1: Global Prediction ---")
        s1_out = os.path.join(args.temp_dir, f"{project_id}_s1.csv")
        s1_mmap = os.path.join(args.temp_dir, f"{project_id}_mmap.dat")
        s1_meta = os.path.join(args.temp_dir, f"{project_id}_meta.pkl")
        
        # [UPDATED] Using 'input_beta' to match the updated stage1.py
        predict_stage1(
            artifact_dir=s1_dir,
            input_beta=args.beta,       # 对应 stage1.py 中的 input_beta
            input_chalm=args.chalm,
            input_camda=args.camda,
            output_path=s1_out,
            keep_mmap_path=s1_mmap, 
            keep_meta_path=s1_meta
        )
        
        # ======================================================================
        # 2. Stage 2 Inference (Hallmark Experts)
        # ======================================================================
        log("--- Stage 2: Hallmark Experts ---")
        s2_out = os.path.join(args.temp_dir, f"{project_id}_s2.csv")
        
        # Stage 2 reuses the memory map from Stage 1 for speed
        predict_stage2(
            artifact_dir=s2_dir,
            stage1_dir=s1_dir,
            mmap_path=s1_mmap,
            meta_path=s1_meta,
            output_path=s2_out
        )
        
        # ======================================================================
        # 3. Stage 3 Inference (Context-Aware Fusion)
        # ======================================================================
        log("--- Stage 3: Context-Aware Fusion ---")
        
        df_s1 = pd.read_csv(s1_out)
        df_s2 = pd.read_csv(s2_out)
        
        # Merge S1 and S2
        df_merged = pd.merge(df_s1, df_s2, on="sample_id", how="inner")
        
        # Merge PCs (if available)
        if args.pc and os.path.exists(args.pc):
            log(f"Loading PCs from {args.pc}")
            df_pc = pd.read_csv(args.pc)
            # Find PC columns
            pc_cols = [c for c in df_pc.columns if c.startswith('RF_PC')]
            # Ensure Sample ID type matches
            df_pc['sample_id'] = df_pc['sample_id'].astype(str)
            df_merged['sample_id'] = df_merged['sample_id'].astype(str)
            
            df_merged = pd.merge(df_merged, df_pc[['sample_id'] + pc_cols], on="sample_id", how="left")
            df_merged[pc_cols] = df_merged[pc_cols].fillna(0)
        
        s3_input_temp = os.path.join(args.temp_dir, f"{project_id}_s3_in.csv")
        df_merged.to_csv(s3_input_temp, index=False)
        
        # Run Stage 3
        predict_stage3(
            artifact_dir=s3_dir,
            input_path=s3_input_temp,
            output_path=args.output
        )
        
        log(f"✅ Prediction Successful: {args.output}")

    except Exception as e:
        log(f"❌ Pipeline Failed: {str(e)}")
        raise e
        
    finally:
        # ======================================================================
        # 4. Cleanup
        # ======================================================================
        if os.path.exists(s1_mmap): os.remove(s1_mmap)
        if os.path.exists(s1_meta): os.remove(s1_meta)
        shutil.rmtree(args.temp_dir)

if __name__ == "__main__":
    main()