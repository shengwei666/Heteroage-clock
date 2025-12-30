"""
heteroage_clock.artifacts.stage1

Artifact handler for Stage 1 (Global Anchor).
"""

from .base import BaseArtifact
import pandas as pd
from typing import Any, Dict, List

class Stage1Artifact(BaseArtifact):
    
    def save_global_model(self, model: Any) -> None:
        self.save("stage1_model", model, file_format="pkl")  # Or joblib
        
    def load_global_model(self) -> Any:
        return self.load("stage1_model")

    def save_orthogonalized_dict(self, data: Dict[str, List[str]]) -> None:
        # Saved as JSON or Pickle. Pickle is safer for Dicts.
        self.save("Stage1_Orthogonalized_Hallmark_Dict", data, file_format="pkl")
        # Optional: Also save as JSON for readability if needed, but Pipeline uses Pickle/JSON logic
        # For simplicity in pipeline.py integration (which expects JSON sometimes), 
        # let's ensure consistency.
        # Updated: pipeline typically reads JSON for dicts. Let's support that.
        import json
        json_path = self._get_path("Stage1_Orthogonalized_Hallmark_Dict.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def save_oof_predictions(self, df: pd.DataFrame) -> None:
        self.save("Stage1_Global_Anchor_OOF", df, file_format="csv")