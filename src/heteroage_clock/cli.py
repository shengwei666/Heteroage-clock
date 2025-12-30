"""
heteroage_clock.cli

This module provides the command-line interface (CLI) for running the heteroage-clock pipeline.
It allows users to execute different stages of the pipeline, including training and inference, from the terminal.
"""

import argparse
from .pipeline import (
    train_stage1, train_stage2, train_stage3, 
    predict_pipeline, predict_stage1, predict_stage2, predict_stage3
)

def create_parser():
    """
    Create and return the argument parser for the CLI.
    """
    parser = argparse.ArgumentParser(description="heteroage-clock CLI")
    subparsers = parser.add_subparsers(dest="command")

    # ==========================================
    # Training Commands
    # ==========================================

    # Stage 1 Train Command
    stage1_train = subparsers.add_parser("stage1-train", help="Train Stage 1 (Global Anchor)")
    stage1_train.add_argument("--project-root", required=True, help="Root directory for project data")
    stage1_train.add_argument("--output-dir", required=True, help="Directory to save Stage 1 outputs")
    stage1_train.add_argument("--pc-path", required=True, help="Path to the PCs CSV file")
    stage1_train.add_argument("--dict-name", required=True, help="Filename of the Hallmark CpG dictionary (e.g. Hallmark_CpG_Dict.json)")
    stage1_train.add_argument("--sweep-file", help="Path to existing sweep report to skip the sweep step")

    # Stage 2 Train Command
    stage2_train = subparsers.add_parser("stage2-train", help="Train Stage 2 (Hallmark Experts)")
    stage2_train.add_argument("--project-root", required=True, help="Root directory for project data")
    stage2_train.add_argument("--output-dir", required=True, help="Directory to save Stage 2 outputs")
    stage2_train.add_argument("--pc-path", required=True, help="Path to the PCs CSV file")
    stage2_train.add_argument("--stage1-oof", required=True, help="Path to Stage 1 OOF predictions CSV")
    stage2_train.add_argument("--stage1-dict", required=True, help="Filename of the Stage 1 Hallmark dictionary (e.g. Hallmark_CpG_Dict.json)")

    # Stage 3 Train Command
    stage3_train = subparsers.add_parser("stage3-train", help="Train Stage 3 (Context-Aware Fusion)")
    stage3_train.add_argument("--project-root", required=False, help="Root directory for project data (optional, for logging)")
    stage3_train.add_argument("--output-dir", required=True, help="Directory to save Stage 3 outputs")
    stage3_train.add_argument("--stage1-oof", required=True, help="Path to Stage 1 OOF predictions CSV")
    stage3_train.add_argument("--stage2-oof", required=True, help="Path to Stage 2 OOF predictions CSV")
    stage3_train.add_argument("--pc-path", required=True, help="Path to the PCs CSV file")

    # ==========================================
    # Inference Commands
    # ==========================================

    # Pipeline Predict Command (Full)
    pipeline_predict = subparsers.add_parser("pipeline-predict", help="Run full pipeline inference (Stage 1 -> 2 -> 3)")
    pipeline_predict.add_argument("--artifact-dir", required=True, help="Root directory containing stage1/stage2/stage3 artifacts")
    pipeline_predict.add_argument("--input", required=True, help="Path to the master input table (CSV or Pickle)")
    pipeline_predict.add_argument("--out", required=True, help="Path to save final predictions")

    # Stage 1 Predict Command
    stage1_predict = subparsers.add_parser("stage1-predict", help="Run Stage 1 inference only")
    stage1_predict.add_argument("--artifact-dir", required=True, help="Directory containing Stage 1 artifacts")
    stage1_predict.add_argument("--input", required=True, help="Path to input table")
    stage1_predict.add_argument("--out", required=True, help="Path to save predictions")

    # Stage 2 Predict Command
    stage2_predict = subparsers.add_parser("stage2-predict", help="Run Stage 2 inference only")
    stage2_predict.add_argument("--artifact-dir", required=True, help="Directory containing Stage 2 artifacts")
    stage2_predict.add_argument("--input", required=True, help="Path to input table")
    stage2_predict.add_argument("--out", required=True, help="Path to save predictions")

    # Stage 3 Predict Command
    stage3_predict = subparsers.add_parser("stage3-predict", help="Run Stage 3 inference only")
    stage3_predict.add_argument("--artifact-dir", required=True, help="Directory containing Stage 3 artifacts")
    stage3_predict.add_argument("--input", required=True, help="Path to merged input table")
    stage3_predict.add_argument("--out", required=True, help="Path to save predictions")

    return parser

def main():
    """
    Main function to parse arguments and run the appropriate pipeline function.
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "stage1-train":
        train_stage1(
            project_root=args.project_root,
            output_dir=args.output_dir,
            pc_path=args.pc_path,
            dict_name=args.dict_name,
            sweep_file=args.sweep_file
        )
    elif args.command == "stage2-train":
        train_stage2(
            project_root=args.project_root,
            output_dir=args.output_dir,
            pc_path=args.pc_path,
            stage1_oof=args.stage1_oof,
            stage1_dict=args.stage1_dict
        )
    elif args.command == "stage3-train":
        train_stage3(
            stage1_oof=args.stage1_oof,
            stage2_oof=args.stage2_oof,
            pc_path=args.pc_path,
            output_dir=args.output_dir,
            project_root=args.project_root
        )
    elif args.command == "pipeline-predict":
        predict_pipeline(
            artifact_dir=args.artifact_dir,
            input_path=args.input,
            output_path=args.out
        )
    elif args.command == "stage1-predict":
        predict_stage1(
            artifact_dir=args.artifact_dir,
            input_path=args.input,
            output_path=args.out
        )
    elif args.command == "stage2-predict":
        predict_stage2(
            artifact_dir=args.artifact_dir,
            input_path=args.input,
            output_path=args.out
        )
    elif args.command == "stage3-predict":
        predict_stage3(
            artifact_dir=args.artifact_dir,
            input_path=args.input,
            output_path=args.out
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()