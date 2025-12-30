# Changelog for heteroage-clock

All notable changes to this project will be documented in this file. Dates are in YYYY-MM-DD format.

## [Unreleased]

### Added
- Initial release of the `heteroage-clock` pipeline.
- Added Stage 1 (Global Anchor) for baseline age prediction using ElasticNet with heterogeneity-aware sampling and leakage-free cross-validation.
- Added Stage 2 (Hallmark Experts) for training hallmark-specific residual models (ElasticNet) per biological domain.
- Added Stage 3 (Context-Aware Fusion) that uses LightGBM for meta-learning and combines Stage 2 corrections with context features (Tissue, Sex, PCs).
- Implemented a CLI tool (`heteroage`) for running all stages of the pipeline.
- Documentation for pipeline architecture and usage.

---

## [v1.0.0] - 2025-12-30

### Added
- Full implementation of Stage 1, Stage 2, and Stage 3 for biological age prediction.
- Command-line interface (CLI) support for training and inference (`stage1-train`, `stage2-train`, `stage3-train`).
- Complete data preprocessing steps, including handling missing values, zero-filling NaN columns, and feature scaling.
- Reproducibility via serialized artifacts for all stages of the pipeline.
- Inference support using pre-trained models and a master table for Stage 1, Stage 2, and Stage 3 predictions.
- Integration of principal component features and biological data (CpG methylation) across all stages.
- Full documentation, including installation instructions, data requirements, and usage guides.

### Fixed
- Various bug fixes and optimizations to ensure deterministic behavior during training and inference.

### Changed
- Updated README with clear step-by-step instructions for training and inference.
- Refinement of data contract and example files for consistent use across training and inference.

---

## [v0.1.0] - 2025-12-15

### Added
- Initial prototype version of the `heteroage-clock` pipeline.
- Basic structure for the three stages of age prediction.
- Preliminary data loading and preprocessing functions.
- First pass at CLI support for running the pipeline.

### Fixed
- Initial bug fixes related to data alignment and imputation.
