# stage3.py
"""
Stage 3 Artifact class for heteroage-clock.

This class handles the saving and loading of artifacts generated during Stage 3 of the pipeline,
which includes the final meta-learner model, attention importance scores, and Out-of-Fold (OOF) predictions.
"""

from .base import BaseArtifact
import joblib
import pandas as pd
import os
import json

class Stage3Artifact(BaseArtifact):
    """
    Stage 3 Artifact class responsible for managing the outputs of Stage 3 in the heteroage-clock pipeline.

    This includes:
    - The final meta-learner model (LightGBM).
    - The OOF predictions from the meta-learner.
    - The feature importance scores from the meta-learner.
    """

    def __init__(self, artifact_dir: str):
        """
        Initialize Stage3Artifact class.

        Args:
            artifact_dir (str): Directory where the Stage 3 artifacts are saved.
        """
        super().__init__(artifact_dir)

    def save_meta_learner_model(self, model):
        """Save the LightGBM meta-learner model to the artifact directory.

        Args:
            model: The trained LightGBM model for meta-learner.
        """
        model_path = "stage3_meta_learner_model.joblib"
        self.save(model_path, model)

    def load_meta_learner_model(self):
        """Load the LightGBM meta-learner model from the artifact directory.

        Returns:
            The loaded LightGBM model.
        """
        model_path = "stage3_meta_learner_model.joblib"
        return self.load(model_path)

    def save_oof_predictions(self, oof_df: pd.DataFrame):
        """Save Out-of-Fold (OOF) predictions for Stage 3.

        Args:
            oof_df (pd.DataFrame): DataFrame containing OOF predictions from the meta-learner.
        """
        self.save_dataframe('stage3_oof_predictions', oof_df)

    def load_oof_predictions(self):
        """Load Out-of-Fold (OOF) predictions for Stage 3.

        Returns:
            pd.DataFrame: DataFrame containing OOF predictions.
        """
        return self.load_dataframe('stage3_oof_predictions')

    def save_attention_importance(self, importance_df: pd.DataFrame):
        """Save attention importance scores (gain) from the meta-learner.

        Args:
            importance_df (pd.DataFrame): DataFrame containing the feature importance (gain) from the LightGBM model.
        """
        self.save_dataframe('stage3_attention_importance', importance_df)

    def load_attention_importance(self):
        """Load attention importance scores (gain) for Stage 3.

        Returns:
            pd.DataFrame: DataFrame containing feature importance (gain).
        """
        return self.load_dataframe('stage3_attention_importance')

    def save_stage3_manifest(self, manifest: dict):
        """Save the Stage 3 manifest containing the configurations and paths for inference.

        Args:
            manifest (dict): Dictionary containing the configuration and path for the final meta-learner.
        """
        path = os.path.join(self.artifact_dir, 'stage3_manifest.json')
        with open(path, 'w') as f:
            json.dump(manifest, f)
        print(f"Stage 3 Manifest saved to {path}")

    def load_stage3_manifest(self):
        """Load the Stage 3 manifest containing the configurations and paths for inference.

        Returns:
            dict: Dictionary containing the configuration and path for the final meta-learner.
        """
        path = os.path.join(self.artifact_dir, 'stage3_manifest.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Stage 3 Manifest not found in {self.artifact_dir}")
