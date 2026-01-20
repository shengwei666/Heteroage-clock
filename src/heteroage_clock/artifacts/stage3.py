import os
import joblib
import json
from typing import Any, Dict, List

class Stage3Artifact:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_fusion_model(self, models: Dict[str, Any]):
        """
        Save the dictionary of fusion models (Ridge objects).
        Structure: {'Global': model, 'TissueA': model, ...}
        """
        joblib.dump(models, os.path.join(self.output_dir, "stage3_fusion_models.pkl"))

    def load_fusion_model(self) -> Dict[str, Any]:
        """Load the fusion models."""
        path = os.path.join(self.output_dir, "stage3_fusion_models.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stage 3 models not found at {path}")
        return joblib.load(path)

    def save_feature_names(self, features: List[str]):
        """Save list of features used in Stage 3."""
        joblib.dump(features, os.path.join(self.output_dir, "stage3_features.pkl"))
        
    def load_feature_names(self) -> List[str]:
        """Load list of features used in Stage 3."""
        return self.load("stage3_features")

    def save(self, name: str, data: Any):
        """
        Generic save method for arbitrary artifacts (config, lists, etc).
        If data is a config dict, also saves a JSON copy for human inspection.
        """
        # Primary storage: Pickle (maintains Python types perfectly)
        pkl_path = os.path.join(self.output_dir, f"{name}.pkl")
        joblib.dump(data, pkl_path)
        
        # Secondary storage: JSON (if dict, for readability)
        if isinstance(data, dict):
            try:
                json_path = os.path.join(self.output_dir, f"{name}.json")
                with open(json_path, 'w') as f:
                    # default=str handles non-serializable types like numpy ints
                    json.dump(data, f, indent=4, default=str)
            except Exception:
                # Ignore JSON errors, pickle is what matters
                pass

    def load(self, name: str) -> Any:
        """Generic load method."""
        path = os.path.join(self.output_dir, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact '{name}' not found at {path}")
        return joblib.load(path)