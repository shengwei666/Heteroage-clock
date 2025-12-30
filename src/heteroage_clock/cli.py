"""
heteroage_clock.cli

This module provides the command-line interface (CLI) for running the heteroage-clock pipeline.
It allows users to execute different stages of the pipeline, including training and inference, from the terminal.
"""

import argparse
from .pipeline import train_stage1, train_stage2, train_stage3, predict_pipeline, predict_stage1, predict_stage2, predict_stage3

def create_parser():
    """
    Create and return the argument parser for the CLI.
    """
    parser = argparse.ArgumentParser(description="heteroage-clock CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Stage 1 Train Command
    stage1_train = subparsers.add_parser("stage1-train", help="Train Stage 1")
    stage1_train.add_argument("--project-root", required=True, help="Root directory for project data")
    stage1_train.add_argument("--output-dir", required=True, help="Directory to save Stage 1 outputs")
    stage1_train.add_argument("--pc-path", required=True, help="Path to the PCs CSV file")
    stage1_train.add_argument("--dict-name", required=True, help="Path to the Hallmark CpG dictionary")
    stage1_train.add_argument("--sweep-file", help="Path to existing sweep report to skip the sweep step")

    # Stage 2 Train Command
    stage2_train = subparsers.add_parser("stage2-train", help="Train Stage 2")
    stage2_train.add_argument("--project-root", required=True, help="Root directory for project data")
    stage2_train.add_argument("--output-dir", required=True, help="Directory to save Stage 2 outputs")
    stage2_train.add_argument("--pc-path", required=True, help="Path to the PCs CSV file")
    stage2_train.add_argument("--stage1-oof", required=True, help="Path to Stage 1 OOF predictions")
    stage2_train.add_argument("--stage1-dict", required=True, help="Path to Stage 1 Hallmark dictionary")

    # Stage 3 Train Command
    stage3_train = subparsers.add_parser("stage3-train", help="Train Stage 3")
    stage3_train.add_argument("--output-dir", required=True, help="Directory to save Stage 3 outputs")
    stage3_train.add_argument("--stage1-oof", required=True, help="Path to Stage 1 OOF predictions")
    stage3_train.add_argument("--stage2-oof", required=True, help="Path to Stage 2 OOF corrections")
    stage3_train.add_argument("--pc-path", required=True, help="Path to the PCs CSV file")

    # Pipeline Predict Command
    pipeline_predict = subparsers.add_parser("pipeline-predict", help="Run full pipeline inference (Stage 1 + Stage 2 + Stage 3)")
    pipeline_predict.add_argument("--artifact-dir", required=True, help="Path to the artifact directory")
    pipeline_predict.add_argument("--input", required=True, help="Path to the master input table (CSV or Parquet)")
    pipeline_predict.add_argument("--out", required=True, help="Path to save final predictions")

    # Stage 1 Predict Command
    stage1_predict = subparsers.add_parser("stage1-predict", help="Stage 1 inference")
    stage1_predict.add_argument("--artifact-dir", required=True, help="Path to the Stage 1 artifacts")
    stage1_predict.add_argument("--input", required=True, help="Path to the master input table (CSV or Parquet)")
    stage1_predict.add_argument("--out", required=True, help="Path to save Stage 1 predictions")

    # Stage 2 Predict Command
    stage2_predict = subparsers.add_parser("stage2-predict", help="Stage 2 inference")
    stage2_predict.add_argument("--artifact-dir", required=True, help="Path to the Stage 2 artifacts")
    stage2_predict.add_argument("--input", required=True, help="Path to the master input table (CSV or Parquet)")
    stage2_predict.add_argument("--out", required=True, help="Path to save Stage 2 corrections")

    # Stage 3 Predict Command
    stage3_predict = subparsers.add_parser("stage3-predict", help="Stage 3 inference")
    stage3_predict.add_argument("--artifact-dir", required=True, help="Path to the Stage 3 artifacts")
    stage3_predict.add_argument("--input", required=True, help="Path to the merged input table (CSV or Parquet)")
    stage3_predict.add_argument("--out", required=True, help="Path to save Stage 3 predictions")

    return parser

def main():
    """
    Main function to parse arguments and run the appropriate pipeline function.
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "stage1-train":
        train_stage1(args.project_root, args.output_dir, args.pc_path, args.dict_name, args.sweep_file)
    elif args.command == "stage2-train":
        train_stage2(args.project_root, args.output_dir, args.pc_path, args.stage1_oof, args.stage1_dict)
    elif args.command == "stage3-train":
        train_stage3(args.stage1_oof, args.stage2_oof, args.pc_path, args.output_dir)
    elif args.command == "pipeline-predict":
        predict_pipeline(args.artifact_dir, args.input, args.out)
    elif args.command == "stage1-predict":
        predict_stage1(args.artifact_dir, args.input, args.out)
    elif args.command == "stage2-predict":
        predict_stage2(args.artifact_dir, args.input, args.out)
    elif args.command == "stage3-predict":
        predict_stage3(args.artifact_dir, args.input, args.out)

if __name__ == "__main__":
    main()
