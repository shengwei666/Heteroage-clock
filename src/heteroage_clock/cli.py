"""
heteroage_clock.cli

Command Line Interface.
Exposes all file paths and hyperparameters as arguments.
Updates: Added support for n_jobs, hyperparameter lists, and intelligent sampling params for BOTH Stage 1 and Stage 2.
"""

import argparse
import sys
import os
# Import from pipeline to maintain architecture
from heteroage_clock.pipeline import (
    train_stage1, train_stage2, train_stage3, 
    predict_pipeline, predict_stage1, predict_stage2, predict_stage3
)
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

    # ==========================
    # --- Stage 1 Train ---
    # ==========================
    p_s1 = subparsers.add_parser("stage1-train", help="Train Stage 1 Global Anchor")
    p_s1.add_argument("--project-root", type=str, help="Root directory for default path inference")
    p_s1.add_argument("--output-dir", type=str, required=True, help="Directory to save artifacts")
    
    p_s1.add_argument("--pc-path", type=str, help="Path to PC covariates CSV")
    p_s1.add_argument("--dict-path", type=str, help="Path to Hallmark JSON Dictionary")
    p_s1.add_argument("--beta-path", type=str, help="Path to Beta Value Pickle")
    p_s1.add_argument("--chalm-path", type=str, help="Path to Chalm Value Pickle")
    p_s1.add_argument("--camda-path", type=str, help="Path to Camda Value Pickle")
    p_s1.add_argument("--sweep-file", type=str, help="Optional sweep file")
    
    p_s1.add_argument("--alphas", type=float, nargs='+', help="List of alpha values to search")
    p_s1.add_argument("--l1-ratios", type=float, nargs='+', help="List of L1 ratio values to search")
    p_s1.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs for grid search")
    
    p_s1.add_argument("--alpha-start", type=float, default=-4.0)
    p_s1.add_argument("--alpha-end", type=float, default=-0.5)
    p_s1.add_argument("--n-alphas", type=int, default=30)
    p_s1.add_argument("--l1-ratio", type=float, default=0.5)
    p_s1.add_argument("--n-splits", type=int, default=5)
    p_s1.add_argument("--seed", type=int, default=42)
    p_s1.add_argument("--max-iter", type=int, default=2000)

    # Intelligent Sampling Params (Stage 1)
    p_s1.add_argument("--min-cohorts", type=int, default=1)
    p_s1.add_argument("--min-cap", type=int, default=30)
    p_s1.add_argument("--max-cap", type=int, default=500)
    p_s1.add_argument("--median-mult", type=float, default=1.0)

    # ==========================
    # --- Stage 2 Train ---
    # ==========================
    p_s2 = subparsers.add_parser("stage2-train", help="Train Stage 2 Hallmark Experts")
    p_s2.add_argument("--project-root", type=str, help="Root directory for default path inference")
    p_s2.add_argument("--output-dir", type=str, required=True)
    
    p_s2.add_argument("--stage1-oof", type=str, help="Path to Stage 1 OOF CSV")
    p_s2.add_argument("--stage1-dict", type=str, help="Path to Stage 1 Orthogonalized Dict")
    p_s2.add_argument("--pc-path", type=str, help="Path to PC covariates")
    p_s2.add_argument("--beta-path", type=str, help="Path to Beta Value Pickle")
    p_s2.add_argument("--chalm-path", type=str, help="Path to Chalm Value Pickle")
    p_s2.add_argument("--camda-path", type=str, help="Path to Camda Value Pickle")
    
    p_s2.add_argument("--alphas", type=float, nargs='+', help="List of alpha values")
    p_s2.add_argument("--l1-ratios", type=float, nargs='+', help="List of L1 ratios")
    p_s2.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    
    p_s2.add_argument("--alpha-start", type=float, default=-4.0)
    p_s2.add_argument("--alpha-end", type=float, default=-0.5)
    p_s2.add_argument("--n-alphas", type=int, default=30)
    p_s2.add_argument("--l1-ratio", type=float, default=0.5)
    p_s2.add_argument("--n-splits", type=int, default=5)
    p_s2.add_argument("--seed", type=int, default=42)
    p_s2.add_argument("--max-iter", type=int, default=2000)

    # Intelligent Sampling Params (Stage 2 - NEW)
    p_s2.add_argument("--min-cohorts", type=int, default=1)
    p_s2.add_argument("--min-cap", type=int, default=30)
    p_s2.add_argument("--max-cap", type=int, default=500)
    p_s2.add_argument("--median-mult", type=float, default=1.0)

    # ==========================
    # --- Stage 3 Train ---
    # ==========================
    p_s3 = subparsers.add_parser("stage3-train", help="Train Stage 3 Context Fusion")
    p_s3.add_argument("--project-root", type=str, help="Root directory for default path inference")
    p_s3.add_argument("--output-dir", type=str, required=True)
    
    p_s3.add_argument("--stage1-oof", type=str, help="Path to Stage 1 OOF CSV")
    p_s3.add_argument("--stage2-oof", type=str, help="Path to Stage 2 OOF CSV")
    p_s3.add_argument("--pc-path", type=str, help="Path to PC covariates")
    
    p_s3.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel threads for LightGBM")
    p_s3.add_argument("--n-estimators", type=int, default=2000)
    p_s3.add_argument("--learning-rate", type=float, default=0.01)
    p_s3.add_argument("--num-leaves", type=int, default=31)
    p_s3.add_argument("--max-depth", type=int, default=-1)
    p_s3.add_argument("--n-splits", type=int, default=5)
    p_s3.add_argument("--seed", type=int, default=42)

    # --- Inference Commands ---
    p_pipe = subparsers.add_parser("pipeline-predict", help="Full Pipeline Inference")
    p_pipe.add_argument("--artifact-dir", required=True)
    p_pipe.add_argument("--input", required=True)
    p_pipe.add_argument("--out", required=True)

    p_p1 = subparsers.add_parser("stage1-predict", help="Stage 1 Inference")
    p_p1.add_argument("--artifact-dir", required=True)
    p_p1.add_argument("--input", required=True)
    p_p1.add_argument("--out", required=True)

    p_p2 = subparsers.add_parser("stage2-predict", help="Stage 2 Inference")
    p_p2.add_argument("--artifact-dir", required=True)
    p_p2.add_argument("--input", required=True)
    p_p2.add_argument("--out", required=True)

    p_p3 = subparsers.add_parser("stage3-predict", help="Stage 3 Inference")
    p_p3.add_argument("--artifact-dir", required=True)
    p_p3.add_argument("--input", required=True)
    p_p3.add_argument("--out", required=True)

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
            alphas=args.alphas,
            l1_ratios=args.l1_ratios,
            n_jobs=args.n_jobs,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
            n_alphas=args.n_alphas,
            l1_ratio=args.l1_ratio,
            n_splits=args.n_splits,
            seed=args.seed,
            max_iter=args.max_iter,
            # Stage 1 Sampling
            min_cohorts=args.min_cohorts,
            min_cap=args.min_cap,
            max_cap=args.max_cap,
            median_mult=args.median_mult,
            project_root=args.project_root
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
            stage1_oof=s1_oof,
            stage1_dict=s1_dict,
            pc_path=pc,
            beta_path=beta,
            chalm_path=chalm,
            camda_path=camda,
            alphas=args.alphas,
            l1_ratios=args.l1_ratios,
            n_jobs=args.n_jobs,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
            n_alphas=args.n_alphas,
            l1_ratio=args.l1_ratio,
            n_splits=args.n_splits,
            seed=args.seed,
            max_iter=args.max_iter,
            # Stage 2 Sampling (Pass through)
            min_cohorts=args.min_cohorts,
            min_cap=args.min_cap,
            max_cap=args.max_cap,
            median_mult=args.median_mult,
            project_root=args.project_root
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
            stage1_oof=s1_oof,
            stage2_oof=s2_oof,
            pc_path=pc,
            n_jobs=args.n_jobs,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            n_splits=args.n_splits,
            seed=args.seed,
            project_root=args.project_root
        )

    elif args.command == "pipeline-predict":
        predict_pipeline(args.artifact_dir, args.input, args.out)
    elif args.command == "stage1-predict":
        predict_stage1(args.artifact_dir, args.input, args.out)
    elif args.command == "stage2-predict":
        predict_stage2(args.artifact_dir, args.input, args.out)
    elif args.command == "stage3-predict":
        predict_stage3(args.artifact_dir, args.input, args.out)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()