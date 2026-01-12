"""
heteroage_clock.artifacts.stage3

Artifact handler for Stage 3 (Context Fusion).
"""

import os
import joblib
import pandas as pd
from typing import Any, List

class Stage3Artifact:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, name: str, data: Any) -> None:
        """Generic save for lists, dicts, etc."""
        path = os.path.join(self.output_dir, f"{name}.joblib")
        joblib.dump(data, path)

    def load(self, name: str) -> Any:
        """Generic load."""
        path = os.path.join(self.output_dir, f"{name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact {name} not found at {path}")
        return joblib.load(path)

    def save_fusion_model(self, model: Any) -> None:
        """Saves the trained LightGBM fusion model."""
        path = os.path.join(self.output_dir, "Stage3_Fusion_Model.joblib")
        joblib.dump(model, path)

    def load_fusion_model(self) -> Any:
        """Loads the trained LightGBM fusion model."""
        path = os.path.join(self.output_dir, "Stage3_Fusion_Model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fusion model not found at {path}")
        return joblib.load(path)

    def save_final_predictions(self, df: pd.DataFrame) -> None:
        """Saves the final HeteroAge predictions."""
        path = os.path.join(self.output_dir, "Stage3_HeteroAge_Predictions.csv")
        df.to_csv(path, index=False)

    def load_final_predictions(self) -> pd.DataFrame:
        path = os.path.join(self.output_dir, "Stage3_HeteroAge_Predictions.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Predictions not found at {path}")
        return pd.read_csv(path)