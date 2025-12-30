"""
heteroage_clock.artifacts.stage2

Artifact handler for Stage 2 (Hallmark Experts).
"""

from .base import BaseArtifact
import pandas as pd
from typing import Any

class Stage2Artifact(BaseArtifact):
    
    def save_expert_model(self, hallmark: str, model: Any) -> None:
        filename = f"stage2_{hallmark}_expert_model"
        self.save(filename, model, file_format="joblib")
        
    def load_expert_model(self, hallmark: str) -> Any:
        filename = f"stage2_{hallmark}_expert_model"
        return self.load(filename)

    def save_oof_corrections(self, df: pd.DataFrame) -> None:
        self.save("Stage2_Hallmark_OOF", df, file_format="csv")