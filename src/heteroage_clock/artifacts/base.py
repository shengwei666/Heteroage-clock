# base.py
"""
Base Artifact class for heteroage-clock.

This class provides a base structure for saving and loading artifacts for each training stage.
Artifacts include trained models, weights, residual corrections, and other intermediate outputs.
"""

import os
import joblib
import pandas as pd

class BaseArtifact:
    """
    Base class for handling artifacts in the heteroage-clock pipeline.

    This class defines common methods for saving and loading artifacts (such as models and weights)
    that are used across different stages (Stage1, Stage2, Stage3).

    Attributes:
        artifact_dir (str): The directory where the artifact is saved.
    """

    def __init__(self, artifact_dir: str):
        """
        Initialize the BaseArtifact.

        Args:
            artifact_dir (str): Path to the directory where the artifacts will be saved or loaded from.
        """
        self.artifact_dir = artifact_dir
        self.ensure_artifact_dir()

    def ensure_artifact_dir(self):
        """Ensure that the artifact directory exists, create it if necessary."""
        if not os.path.exists(self.artifact_dir):
            os.makedirs(self.artifact_dir)
    
    def save(self, artifact_name: str, obj):
        """Save an artifact (e.g., model, weights, etc.) to the artifact directory.

        Args:
            artifact_name (str): The name of the artifact.
            obj: The object to be saved (can be a model, dataframe, or other serializable objects).
        """
        path = os.path.join(self.artifact_dir, f"{artifact_name}.joblib")
        joblib.dump(obj, path)
        print(f"Artifact saved to {path}")

    def load(self, artifact_name: str):
        """Load an artifact from the artifact directory.

        Args:
            artifact_name (str): The name of the artifact to be loaded.

        Returns:
            The loaded object.
        """
        path = os.path.join(self.artifact_dir, f"{artifact_name}.joblib")
        if os.path.exists(path):
            obj = joblib.load(path)
            print(f"Artifact loaded from {path}")
            return obj
        else:
            raise FileNotFoundError(f"Artifact {artifact_name} not found in {self.artifact_dir}")

    def save_dataframe(self, artifact_name: str, df: pd.DataFrame):
        """Save a dataframe artifact to the artifact directory.

        Args:
            artifact_name (str): The name of the artifact.
            df (pd.DataFrame): The DataFrame to be saved.
        """
        path = os.path.join(self.artifact_dir, f"{artifact_name}.csv")
        df.to_csv(path, index=False)
        print(f"DataFrame artifact saved to {path}")

    def load_dataframe(self, artifact_name: str):
        """Load a dataframe artifact from the artifact directory.

        Args:
            artifact_name (str): The name of the artifact to be loaded.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        path = os.path.join(self.artifact_dir, f"{artifact_name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"DataFrame artifact loaded from {path}")
            return df
        else:
            raise FileNotFoundError(f"DataFrame artifact {artifact_name} not found in {self.artifact_dir}")
