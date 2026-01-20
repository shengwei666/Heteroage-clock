"""
heteroage_clock.cli

Command Line Interface.
Exposes all file paths and hyperparameters as arguments.
Updates: 
1. [Critical] Stage 2 now requires explicit '--mmap-path'.
2. [Major] Added Optuna search range arguments for Stage 1 & 2.
3. [Distributed] Added --storage and --study-name for all training stages.
"""

import argparse
import sys
import os

from heteroage_clock.stages.stage1 import train_stage1, predict_stage1
from heteroage_clock.stages.stage2 import train_stage2, predict_stage2
from heteroage_clock.stages.stage3 import train_stage3, predict_stage3
try:
    from heteroage_clock.pipeline import predict_pipeline
except ImportError:
    predict_pipeline = None

from heteroage_clock.utils.logging import log

def resolve_path(arg_path, project_root, default_rel_path):
    if arg_path: return arg_path
    if project_root:
        potential_path = os.path.join(project_root, default_rel_path)
        if os.path.exists(potential_path): return potential_path
    return None

def main():
    parser = argparse.ArgumentParser(description="HeteroAge-Clock CLI (Optuna Distributed)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ==========================
    # --- Stage 1 Train ---
    # ==========================
    p_s1 = subparsers.add_parser("stage1-train", help="Train Stage 1")
    p_s1.add_argument("--project-root", type=str)
    p_s1.add_argument("--output-dir", type=str, required=True)
    p_s1.add_argument("--pc-path", type=str)
    p_s1.add_argument("--dict-path", type=str)
    p_s1.add_argument("--beta-path", type=str)
    p_s1.add_argument("--chalm-path", type=str)
    p_s1.add_argument("--camda-path", type=str)
    
    p_s1.add_argument("--n-trials", type=int, default=50)
    p_s1.add_argument("--n-jobs", type=int, default=1)
    p_s1.add_argument("--n-splits", type=int, default=5)
    p_s1.add_argument("--seed", type=int, default=42)
    p_s1.add_argument("--max-iter", type=int, default=2000)
    p_s1.add_argument("--min-cohorts", type=int, default=2)

    # Search Space
    p_s1.add_argument("--min-cap-low", type=int, default=10)
    p_s1.add_argument("--min-cap-high", type=int, default=60)
    p_s1.add_argument("--max-cap-low", type=int, default=200)
    p_s1.add_argument("--max-cap-high", type=int, default=1000)
    p_s1.add_argument("--median-mult-low", type=float, default=0.5)
    p_s1.add_argument("--median-mult-high", type=float, default=2.5)
    p_s1.add_argument("--alpha-low", type=float, default=1e-4)
    p_s1.add_argument("--alpha-high", type=float, default=1.0)
    p_s1.add_argument("--l1-low", type=float, default=0.0)
    p_s1.add_argument("--l1-high", type=float, default=1.0)
    
    # [Distributed Args]
    p_s1.add_argument("--storage", type=str)
    p_s1.add_argument("--study-name", type=str)

    # ==========================
    # --- Stage 2 Train ---
    # ==========================
    p_s2 = subparsers.add_parser("stage2-train", help="Train Stage 2")
    p_s2.add_argument("--project-root", type=str)
    p_s2.add_argument("--output-dir", type=str, required=True)
    p_s2.add_argument("--stage1-dir", type=str, required=True)
    p_s2.add_argument("--mmap-path", type=str, required=True)
    
    p_s2.add_argument("--n-trials", type=int, default=30)
    p_s2.add_argument("--n-jobs", type=int, default=1)
    p_s2.add_argument("--n-splits", type=int, default=5)
    p_s2.add_argument("--seed", type=int, default=42)
    p_s2.add_argument("--max-iter", type=int, default=2000)
    p_s2.add_argument("--min-cohorts", type=int, default=2)

    # Search Space
    p_s2.add_argument("--min-cap-low", type=int, default=10)
    p_s2.add_argument("--min-cap-high", type=int, default=60)
    p_s2.add_argument("--max-cap-low", type=int, default=200)
    p_s2.add_argument("--max-cap-high", type=int, default=1000)
    p_s2.add_argument("--median-mult-low", type=float, default=0.5)
    p_s2.add_argument("--median-mult-high", type=float, default=2.5)
    p_s2.add_argument("--alpha-low", type=float, default=1e-4)
    p_s2.add_argument("--alpha-high", type=float, default=1.0)
    p_s2.add_argument("--l1-low", type=float, default=0.0)
    p_s2.add_argument("--l1-high", type=float, default=1.0)
    
    # [Distributed Args]
    p_s2.add_argument("--storage", type=str)
    p_s2.add_argument("--study-name", type=str)

    # ==========================
    # --- Stage 3 Train ---
    # ==========================
    p_s3 = subparsers.add_parser("stage3-train", help="Train Stage 3")
    p_s3.add_argument("--project-root", type=str)
    p_s3.add_argument("--output-dir", type=str, required=True)
    p_s3.add_argument("--stage1-oof", type=str, required=True)
    p_s3.add_argument("--stage2-oof", type=str, required=True)
    p_s3.add_argument("--pc-path", type=str)
    
    p_s3.add_argument("--n-trials", type=int, default=0)
    p_s3.add_argument("--alpha", type=float, default=1.0)
    p_s3.add_argument("--min-samples-for-tissue", type=int, default=50)
    p_s3.add_argument("--n-splits", type=int, default=5)
    p_s3.add_argument("--seed", type=int, default=42)
    
    # [Distributed Args]
    p_s3.add_argument("--storage", type=str)
    p_s3.add_argument("--study-name", type=str)

    # ==========================
    # --- Inference ---
    # ==========================
    p_pipe = subparsers.add_parser("pipeline-predict", help="Full Pipeline Inference")
    p_pipe.add_argument("--artifact-dir", required=True)
    p_pipe.add_argument("--input", required=True)
    p_pipe.add_argument("--out", required=True)
    p_pipe.add_argument("--input-chalm")
    p_pipe.add_argument("--input-camda")
    p_pipe.add_argument("--input-pc")

    p_p1 = subparsers.add_parser("stage1-predict", help="Stage 1 Only (Debug)")
    p_p1.add_argument("--artifact-dir", required=True)
    p_p1.add_argument("--input", required=True)
    p_p1.add_argument("--out", required=True)

    p_p2 = subparsers.add_parser("stage2-predict", help="Stage 2 Only (Debug)")
    p_p2.add_argument("--artifact-dir", required=True)
    p_p2.add_argument("--input", required=True)
    p_p2.add_argument("--out", required=True)

    p_p3 = subparsers.add_parser("stage3-predict", help="Stage 3 Only (Debug)")
    p_p3.add_argument("--artifact-dir", required=True)
    p_p3.add_argument("--input", required=True)
    p_p3.add_argument("--out", required=True)

    args = parser.parse_args()

    # --- Execution ---

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
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            n_splits=args.n_splits,
            seed=args.seed,
            max_iter=args.max_iter,
            min_cohorts=args.min_cohorts,
            min_cap_low=args.min_cap_low, min_cap_high=args.min_cap_high,
            max_cap_low=args.max_cap_low, max_cap_high=args.max_cap_high,
            median_mult_low=args.median_mult_low, median_mult_high=args.median_mult_high,
            alpha_low=args.alpha_low, alpha_high=args.alpha_high,
            l1_low=args.l1_low, l1_high=args.l1_high,
            # [Distributed]
            storage=args.storage,
            study_name=args.study_name
        )

    elif args.command == "stage2-train":
        train_stage2(
            stage1_dir=args.stage1_dir,
            output_dir=args.output_dir,
            mmap_path=args.mmap_path,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            n_splits=args.n_splits,
            seed=args.seed,
            max_iter=args.max_iter,
            min_cohorts=args.min_cohorts,
            min_cap_low=args.min_cap_low, min_cap_high=args.min_cap_high,
            max_cap_low=args.max_cap_low, max_cap_high=args.max_cap_high,
            median_mult_low=args.median_mult_low, median_mult_high=args.median_mult_high,
            alpha_low=args.alpha_low, alpha_high=args.alpha_high,
            l1_low=args.l1_low, l1_high=args.l1_high,
            # [Distributed]
            storage=args.storage,
            study_name=args.study_name
        )

    elif args.command == "stage3-train":
        pc = resolve_path(args.pc_path, args.project_root, "4.Data_assembly/Feature_Sets/RefGuided_PCs.csv")
        train_stage3(
            output_dir=args.output_dir,
            stage1_oof=args.stage1_oof,
            stage2_oof=args.stage2_oof,
            pc_path=pc,
            n_trials=args.n_trials,
            alpha=args.alpha,
            min_samples_for_tissue=args.min_samples_for_tissue,
            n_splits=args.n_splits,
            seed=args.seed,
            # [Distributed]
            storage=args.storage,
            study_name=args.study_name
        )

    elif args.command == "pipeline-predict":
        if predict_pipeline:
            predict_pipeline(
                args.artifact_dir, 
                args.input, 
                args.out,
                input_chalm=args.input_chalm,
                input_camda=args.input_camda,
                input_pc=args.input_pc
            )
        else:
            log("Error: predict_pipeline import failed.")
            sys.exit(1)

    elif args.command == "stage1-predict":
        predict_stage1(args.artifact_dir, args.input, args.input, args.out, [])

    elif args.command == "stage2-predict":
        predict_stage2(args.artifact_dir, args.artifact_dir, args.input, args.input, args.out, [])

    elif args.command == "stage3-predict":
        predict_stage3(args.artifact_dir, args.input, args.out)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()