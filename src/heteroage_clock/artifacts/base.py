"""
heteroage_clock.artifacts.base

Base class for artifact management. 
Handles low-level IO operations (save/load) with automatic format detection.
"""

import os
import joblib
import pandas as pd
import pickle
from typing import Any, Optional
from heteroage_clock.utils.logging import log

class BaseArtifact:
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def _get_path(self, filename: str) -> str:
        return os.path.join(self.artifact_dir, filename)

    def save(self, name: str, data: Any, file_format: str = None) -> None:
        """
        Generic save method.
        
        Args:
            name (str): Identifier name (e.g., 'stage1_features').
            data (Any): The object to save.
            file_format (str, optional): 'csv', 'pkl', 'joblib'. If None, inferred.
        """
        # Infer format if not provided
        if file_format is None:
            if isinstance(data, pd.DataFrame):
                file_format = 'csv'
            elif hasattr(data, 'fit') and hasattr(data, 'predict'): # Scikit-learn models
                file_format = 'joblib'
            else:
                file_format = 'pkl'

        # Construct filename
        if not name.endswith(f".{file_format}"):
            filename = f"{name}.{file_format}"
        else:
            filename = name
            
        path = self._get_path(filename)
        
        log(f"Saving artifact '{name}' to {path}...")
        
        if file_format == 'csv':
            data.to_csv(path, index=False)
        elif file_format == 'joblib':
            joblib.dump(data, path)
        elif file_format == 'pkl':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    def load(self, name: str, file_format: str = None) -> Any:
        """
        Generic load method.
        """
        # Try to find file if format is not strictly given
        path = self._get_path(name)
        
        if not os.path.exists(path):
            # Try appending common extensions
            for ext in ['pkl', 'joblib', 'csv']:
                test_path = f"{path}.{ext}"
                if os.path.exists(test_path):
                    path = test_path
                    file_format = ext
                    break
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Artifact not found: {name} in {self.artifact_dir}")

        # Infer format from extension if found
        if file_format is None:
            if path.endswith('.csv'): file_format = 'csv'
            elif path.endswith('.joblib'): file_format = 'joblib'
            elif path.endswith('.pkl'): file_format = 'pkl'

        log(f"Loading artifact '{name}' from {path}...")

        if file_format == 'csv':
            return pd.read_csv(path)
        elif file_format == 'joblib':
            return joblib.load(path)
        elif file_format == 'pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown format for file: {path}")