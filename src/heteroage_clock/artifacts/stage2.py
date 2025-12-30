# stage2.py
"""
Stage 2 Artifact class for heteroage-clock.

This class handles the saving and loading of artifacts generated during Stage 2 of the pipeline,
including the hallmark-specific expert models, OOF corrections, and other intermediate outputs.
"""

from .base import BaseArtifact
import joblib
import pandas as pd

class Stage2Artifact(BaseArtifact):
    """
    Stage 2 Artifact class responsible for managing the outputs of Stage 2 in the heteroage-clock pipeline.

    This includes:
    - The hallmark expert models.
    - The OOF corrections for each hallmark.
    - Any intermediate results related to Stage 2.
    """

    def __init__(self, artifact_dir: str):
        """
        Initialize Stage2Artifact class.

        Args:
            artifact_dir (str): Directory where the Stage 2 artifacts are saved.
        """
        super().__init__(artifact_dir)

    def save_expert_model(self, hallmark_name: str, model):
        """Save the hallmark-specific expert model to the artifact directory.

        Args:
            hallmark_name (str): The name of the hallmark (e.g., Inflammation, Metabolism).
            model: The trained ElasticNet model for the given hallmark.
        """
        model_path = f"stage2_{hallmark_name}_expert_model.joblib"
        self.save(model_path, model)

    def load_expert_model(self, hallmark_name: str):
        """Load the hallmark-specific expert model from the artifact directory.

        Args:
            hallmark_name (str): The name of the hallmark (e.g., Inflammation, Metabolism).

        Returns:
            The loaded ElasticNet model for the given hallmark.
        """
        model_path = f"stage2_{hallmark_name}_expert_model.joblib"
        return self.load(model_path)

    def save_oof_corrections(self, oof_df: pd.DataFrame):
        """Save Out-of-Fold (OOF) corrections for Stage 2.

        Args:
            oof_df (pd.DataFrame): DataFrame containing OOF corrections for Stage 2.
        """
        self.save_dataframe('stage2_oof_corrections', oof_df)

    def load_oof_corrections(self):
        """Load Out-of-Fold (OOF) corrections for Stage 2.

        Returns:
            pd.DataFrame: DataFrame containing OOF corrections.
        """
        return self.load_dataframe('stage2_oof_corrections')

    def save_expert_weights(self, weights_df: pd.DataFrame):
        """Save the weights of the best models for all hallmarks in Stage 2.

        Args:
            weights_df (pd.DataFrame): DataFrame containing the weights of the best models for each hallmark.
        """
        self.save_dataframe('stage2_expert_weights', weights_df)

    def load_expert_weights(self):
        """Load the weights of the best models for all hallmarks from Stage 2.

        Returns:
            pd.DataFrame: DataFrame containing the weights of the best models.
        """
        return self.load_dataframe('stage2_expert_weights')

    def save_stage2_manifest(self, manifest: dict):
        """Save the Stage 2 manifest containing the models and configurations for inference.

        Args:
            manifest (dict): Dictionary containing the model configurations and paths for each hallmark.
        """
        path = os.path.join(self.artifact_dir, 'stage2_manifest.json')
        with open(path, 'w') as f:
            json.dump(manifest, f)
        print(f"Stage 2 Manifest saved to {path}")

    def load_stage2_manifest(self):
        """Load the Stage 2 manifest containing the models and configurations for inference.

        Returns:
            dict: Dictionary containing the model configurations and paths for each hallmark.
        """
        path = os.path.join(self.artifact_dir, 'stage2_manifest.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Stage 2 Manifest not found in {self.artifact_dir}")
