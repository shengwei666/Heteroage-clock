import os
import joblib
import pandas as pd
from typing import Any, Dict

class Stage2Artifact:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_expert_models(self, models: Dict[str, Any]):
        """Save the dictionary of Hallmark Expert Models."""
        joblib.dump(models, os.path.join(self.output_dir, "stage2_expert_models.pkl"))

    def load_expert_models(self) -> Dict[str, Any]:
        path = os.path.join(self.output_dir, "stage2_expert_models.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert models not found at {path}")
        return joblib.load(path)

    def save_oof_predictions(self, df: pd.DataFrame):
        """Save OOF predictions CSV."""
        df.to_csv(os.path.join(self.output_dir, "Stage2_Hallmark_Experts_OOF.csv"), index=False)
    
    def load_oof_predictions(self) -> pd.DataFrame:
        path = os.path.join(self.output_dir, "Stage2_Hallmark_Experts_OOF.csv")
        if not os.path.exists(path):
            # 修复了这里的缩进问题
            raise FileNotFoundError(f"OOF predictions not found at {path}")
        return pd.read_csv(path)

    def save(self, name: str, data: Any):
        """Generic save method."""
        path = os.path.join(self.output_dir, f"{name}.pkl")
        if isinstance(data, pd.DataFrame):
            data.to_pickle(path)
        else:
            joblib.dump(data, path)

    def load(self, name: str) -> Any:
        """Generic load method."""
        path = os.path.join(self.output_dir, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact '{name}' not found at {path}")
        return joblib.load(path)