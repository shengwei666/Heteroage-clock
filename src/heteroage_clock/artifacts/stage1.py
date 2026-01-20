import os
import joblib
import json
import pandas as pd
from typing import Any, List, Dict

class Stage1Artifact:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_global_model(self, model: Any):
        """Save the trained Stage 1 Pipeline."""
        joblib.dump(model, os.path.join(self.output_dir, "stage1_model.pkl"))

    def load_global_model(self) -> Any:
        path = os.path.join(self.output_dir, "stage1_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Global model not found at {path}")
        return joblib.load(path)

    def save_orthogonalized_dict(self, data: Dict[str, List[str]]):
        """Save the Hallmark -> Features dictionary (JSON)."""
        with open(os.path.join(self.output_dir, "Stage1_Orthogonalized_Hallmark_Dict.json"), 'w') as f:
            json.dump(data, f, indent=4)

    def load_orthogonalized_dict(self) -> Dict[str, List[str]]:
        path = os.path.join(self.output_dir, "Stage1_Orthogonalized_Hallmark_Dict.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dictionary not found at {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def save_oof_predictions(self, df: pd.DataFrame):
        """Save Out-of-Fold predictions CSV."""
        df.to_csv(os.path.join(self.output_dir, "Stage1_Global_Anchor_OOF.csv"), index=False)

    def load_oof_predictions(self) -> pd.DataFrame:
        path = os.path.join(self.output_dir, "Stage1_Global_Anchor_OOF.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"OOF predictions not found at {path}")
        return pd.read_csv(path)

    def save(self, name: str, data: Any):
        """
        Generic save method.
        Handles DataFrames (as Pickle) and generic objects (Pickle).
        """
        if isinstance(data, pd.DataFrame):
            data.to_pickle(os.path.join(self.output_dir, f"{name}.pkl"))
        else:
            joblib.dump(data, os.path.join(self.output_dir, f"{name}.pkl"))

    def load(self, name: str) -> Any:
        """Generic load method."""
        path = os.path.join(self.output_dir, f"{name}.pkl")
        if not os.path.exists(path):
            # 修复了这里的缩进问题
            raise FileNotFoundError(f"Artifact '{name}' not found at {path}")
        return joblib.load(path)