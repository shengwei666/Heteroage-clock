"""
heteroage_clock.data.io

Input/Output utilities for loading data and saving predictions.
"""

import os
import pandas as pd
from typing import Optional
from heteroage_clock.utils.logging import log

def load_data(file_path: str, file_type: str = None) -> pd.DataFrame:
    """
    Load data from a file (CSV or Pickle).
    
    Args:
        file_path (str): Path to the file.
        file_type (str, optional): 'csv' or 'pickle'. If None, inferred from extension.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if file_type is None:
        if file_path.endswith('.csv'):
            file_type = 'csv'
        elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            file_type = 'pickle'
        else:
            # Default fallback
            file_type = 'csv'
            
    log(f"Loading data from {file_path} ({file_type})...")
    
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'pickle':
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    """
    Save predictions to a CSV file, ensuring the directory exists.
    
    Args:
        df (pd.DataFrame): Dataframe containing predictions.
        output_path (str): Destination path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"Predictions saved to {output_path}")