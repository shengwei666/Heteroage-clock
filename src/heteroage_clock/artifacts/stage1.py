# stage1.py
"""
Stage 1 Artifact class for heteroage-clock.

This class handles the saving and loading of artifacts generated during Stage 1 of the pipeline, 
including the global anchor model, OOF predictions, and other intermediate outputs.
"""

from .base import BaseArtifact
import joblib
import pandas as pd

class Stage1Artifact(BaseArtifact):
    """
    Stage 1 Artifact class responsible for managing the outputs of Stage 1 in the heteroage-clock pipeline.

    This includes:
    - The global anchor model.
    - The OOF predictions (Out-of-Fold).
    - The best model weights.
    - Any intermediate results related to Stage 1.
    """

    def __init__(self, artifact_dir: str):
        """
        Initialize Stage1Artifact class.

        Args:
            artifact_dir (str): Directory where the Stage 1 artifacts are saved.
        """
        super().__init__(artifact_dir)

    def save_global_model(self, model):
        """Save the global anchor model to the artifact directory.

        Args:
            model: The trained ElasticNet model from Stage 1.
        """
        self.save('stage1_global_model', model)

    def load_global_model(self):
        """Load the global anchor model from the artifact directory.

        Returns:
            The loaded ElasticNet model.
        """
        return self.load('stage1_global_model')

    def save_oof_predictions(self, oof_df: pd.DataFrame):
        """Save Out-of-Fold (OOF) predictions for Stage 1.

        Args:
            oof_df (pd.DataFrame): DataFrame containing OOF predictions for Stage 1.
        """
        self.save_dataframe('stage1_oof_predictions', oof_df)

    def load_oof_predictions(self):
        """Load Out-of-Fold (OOF) predictions for Stage 1.

        Returns:
            pd.DataFrame: DataFrame containing OOF predictions.
        """
        return self.load_dataframe('stage1_oof_predictions')

    def save_best_model_weights(self, weights_df: pd.DataFrame):
        """Save the weights of the best model from Stage 1.

        Args:
            weights_df (pd.DataFrame): DataFrame containing the weights of the best model.
        """
        self.save_dataframe('stage1_best_model_weights', weights_df)

    def load_best_model_weights(self):
        """Load the weights of the best model from Stage 1.

        Returns:
            pd.DataFrame: DataFrame containing the weights of the best model.
        """
        return self.load_dataframe('stage1_best_model_weights')

    def save_orthogonalized_dict(self, ortho_dict: dict):
        """Save the orthogonalized hallmark dictionary from Stage 1.

        Args:
            ortho_dict (dict): The orthogonalized hallmark dictionary.
        """
        path = os.path.join(self.artifact_dir, 'stage1_orthogonalized_hallmark_dict.json')
        with open(path, 'w') as f:
            json.dump(ortho_dict, f)
        print(f"Orthogonalized Hallmark Dict saved to {path}")

    def load_orthogonalized_dict(self):
        """Load the orthogonalized hallmark dictionary from Stage 1.

        Returns:
            dict: The loaded orthogonalized hallmark dictionary.
        """
        path = os.path.join(self.artifact_dir, 'stage1_orthogonalized_hallmark_dict.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Orthogonalized Hallmark Dict not found in {self.artifact_dir}")
