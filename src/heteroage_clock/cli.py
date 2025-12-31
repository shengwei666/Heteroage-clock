"""
heteroage_clock.cli

Command Line Interface.
Exposes all file paths and hyperparameters as arguments.
"""

import argparse
import sys
import os
from heteroage_clock.stages.stage1 import train_stage1, predict_stage1
from heteroage_clock.stages.stage2 import train_stage2, predict_stage2
from heteroage_clock.stages.stage3 import train_stage3, predict_stage3
from heteroage_clock.utils.logging import log

def resolve_path(arg_path, project_root, default_rel_path):
    if arg_path: return arg_path
    if project_root:
        potential_path = os.path.join(project_root, default_rel_path)
        if os.path.exists(potential_path): return potential_path
    return None

def main():
    parser = argparse.ArgumentParser(description="HeteroAge-Clock CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Stage 1 Train ---
    p_s1 = subparsers.add_parser("stage1-train", help="Train Stage 1 Global Anchor")
    p_s1.add_argument("--project-root", type=str, help="Root directory for default path inference")
    p_s1.add_argument("--output-dir", type=str, required=True, help="Directory to save artifacts")
    
    # Input Files
    p_s1.add_argument("--pc-path", type=str, help="Path to PC covariates CSV")
    p_s1.add_argument("--dict-path", type=str, help="Path to Hallmark JSON Dictionary")
    p_s1.add_argument("--beta-path", type=str, help="Path to Beta Value Pickle")
    p_s1.add_argument("--chalm-path", type=str, help="Path to Chalm Value Pickle")
    p_s1.add_argument("--camda-path", type=str, help="Path to Camda Value Pickle")
    p_s1.add_argument("--sweep-file", type=str, help="Optional sweep file")
    
    # Hyperparameters
    p_s1.add_argument("--alpha-start", type=float, default=-4.0, help="Log10 start of alpha range")
    p_s1.add_argument("--alpha-end", type=float, default=-0.5, help="Log10 end of alpha range")
    p_s1.add_argument("--n-alphas", type=int, default=30, help="Number of alphas")
    p_s1.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet mixing parameter")
    p_s1.add_argument("--n-splits", type=int, default=5, help="Number of CV splits")
    p_s1.add_argument("--seed", type=int, default=42, help="Random seed")
    p_s1.add_argument("--max-iter", type=int, default=2000, help="Max iterations for solver")

    # --- Stage 2 Train ---
    p_s2 = subparsers.add_parser("stage2-train", help="Train Stage 2 Hallmark Experts")
    p_s2.add_argument("--project-root", type=str, help="Root directory for default path inference")
    p_s2.add_argument("--output-dir", type=str, required=True)
    
    # Input Files
    p_s2.add_argument("--stage1-oof", type=str, help="Path to Stage 1 OOF CSV")
    p_s2.add_argument("--stage1-dict", type=str, help="Path to Stage 1 Orthogonalized Dict")
    p_s2.add_argument("--pc-path", type=str, help="Path to PC covariates")
    p_s2.add_argument("--beta-path", type=str, help="Path to Beta Value Pickle")
    p_s2.add_argument("--chalm-path", type=str, help="Path to Chalm Value Pickle")
    p_s2.add_argument("--camda-path", type=str, help="Path to Camda Value Pickle")
    
    # Hyperparameters (Same as Stage 1)
    p_s2.add_argument("--alpha-start", type=float, default=-4.0)
    p_s2.add_argument("--alpha-end", type=float, default=-0.5)
    p_s2.add_argument("--n-alphas", type=int, default=30)
    p_s2.add_argument("--l1-ratio", type=float, default=0.5)
    p_s2.add_argument("--n-splits", type=int, default=5)
    p_s2.add_argument("--seed", type=int, default=42)
    p_s2.add_argument("--max-iter", type=int, default=2000)

    # --- Stage 3 Train ---
    p_s3 = subparsers.add_parser("stage3-train", help="Train Stage 3 Context Fusion")
    p_s3.add_argument("--project-root", type=str, help="Root directory for default path inference")
    p_s3.add_argument("--output-dir", type=str, required=True)
    
    # Input Files
    p_s3.add_argument("--stage1-oof", type=str, help="Path to Stage 1 OOF CSV")
    p_s3.add_argument("--stage2-oof", type=str, help="Path to Stage 2 OOF CSV")
    p_s3.add_argument("--pc-path", type=str, help="Path to PC covariates")
    
    # Hyperparameters (LightGBM)
    p_s3.add_argument("--n-estimators", type=int, default=2000)
    p_s3.add_argument("--learning-rate", type=float, default=0.01)
    p_s3.add_argument("--num-leaves", type=int, default=31)
    p_s3.add_argument("--max-depth", type=int, default=-1)
    p_s3.add_argument("--n-splits", type=int, default=5)
    p_s3.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "stage1-train":
        pc = resolve_path(args.pc_path, args.project_root, "4.Data_assembly/Feature_Sets/RefGuided_PCs.csv")
        dic = resolve_path(args.dict_path, args.project_root, "4.Data_assembly/Feature_Sets/Hallmark_CpG_Dict_Final.json")
        beta = resolve_path(args.beta_path, args.project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Beta_Train_RefGuided_NoCpH.pkl")
        chalm = resolve_path(args.chalm_path, args.project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Chalm_Train_RefGuided_NoCpH.pkl")
        camda = resolve_path(args.camda_path, args.project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Camda_Train_RefGuided_NoCpH.pkl")

        if not all([pc, dic, beta, chalm, camda]):
            log("Error: Missing input files. Provide explicit paths or valid --project-root.")
            sys.exit(1)

        train_stage1(
            output_dir=args.output_dir,
            pc_path=pc,
            dict_path=dic,
            beta_path=beta,
            chalm_path=chalm,
            camda_path=camda,
            sweep_file=args.sweep_file,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
            n_alphas=args.n_alphas,
            l1_ratio=args.l1_ratio,
            n_splits=args.n_splits,
            seed=args.seed,
            max_iter=args.max_iter
        )

    elif args.command == "stage2-train":
        s1_oof = args.stage1_oof
        s1_dict = args.stage1_dict
        pc = resolve_path(args.pc_path, args.project_root, "4.Data_assembly/Feature_Sets/RefGuided_PCs.csv")
        beta = resolve_path(args.beta_path, args.project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Beta_Train_RefGuided_NoCpH.pkl")
        chalm = resolve_path(args.chalm_path, args.project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Chalm_Train_RefGuided_NoCpH.pkl")
        camda = resolve_path(args.camda_path, args.project_root, "4.Data_assembly/Raw_Aligned_RefGuided/Camda_Train_RefGuided_NoCpH.pkl")

        if not all([s1_oof, s1_dict, pc, beta, chalm, camda]):
             log("Error: Missing input files for Stage 2.")
             sys.exit(1)

        train_stage2(
            output_dir=args.output_dir,
            stage1_oof_path=s1_oof,
            stage1_dict_path=s1_dict,
            pc_path=pc,
            beta_path=beta,
            chalm_path=chalm,
            camda_path=camda,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
            n_alphas=args.n_alphas,
            l1_ratio=args.l1_ratio,
            n_splits=args.n_splits,
            seed=args.seed,
            max_iter=args.max_iter
        )

    elif args.command == "stage3-train":
        s1_oof = args.stage1_oof
        s2_oof = args.stage2_oof
        pc = resolve_path(args.pc_path, args.project_root, "4.Data_assembly/Feature_Sets/RefGuided_PCs.csv")
        
        if not all([s1_oof, s2_oof, pc]):
             log("Error: Missing input files for Stage 3.")
             sys.exit(1)
             
        train_stage3(
            output_dir=args.output_dir,
            stage1_oof_path=s1_oof,
            stage2_oof_path=s2_oof,
            pc_path=pc,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            n_splits=args.n_splits,
            seed=args.seed
        )

    else:
        parser.print_help()

if __name__ == "__main__":
    main()