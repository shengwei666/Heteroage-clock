"""
heteroage_clock.artifacts.stage3

Artifact handler for Stage 3 (Context Fusion).
"""

from .base import BaseArtifact
import pandas as pd
from typing import Any

class Stage3Artifact(BaseArtifact):
    
    def save_meta_learner_model(self, model: Any) -> None:
        self.save("stage3_model", model, file_format="pkl")
        
    def load_meta_learner_model(self) -> Any:
        return self.load("stage3_model")

    def save_attention_importance(self, df: pd.DataFrame) -> None:
        self.save("Stage3_Feature_Importance", df, file_format="csv")
        
    def save_oof_predictions(self, df: pd.DataFrame) -> None:
        self.save("Stage3_Final_Predictions_Train", df, file_format="csv")