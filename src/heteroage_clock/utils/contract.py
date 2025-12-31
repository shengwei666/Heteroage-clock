"""
heteroage_clock.utils.contract

Data contract validation utilities.
"""

from typing import List, Dict, Optional
import pandas as pd

def create_contract(columns: List[str]) -> Dict[str, str]:
    contract = {col: "float32" for col in columns}
    return contract

def validate_contract(input_columns: List[str], contract: Dict[str, str]) -> bool:
    contract_columns = set(contract.keys())
    input_columns_set = set(input_columns)

    if not contract_columns.issubset(input_columns_set):
        missing = contract_columns - input_columns_set
        print(f"Validation Failed. Missing required columns: {missing}")
        return False
    return True

# [FIX] Added missing function to satisfy tests
def validate_master_table(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> None:
    """
    Validates the inference master table structure.
    
    Args:
        df: Input DataFrame.
        required_cols: Optional list of specific columns that must exist.
    
    Raises:
        ValueError: If validation fails.
    """
    # 1. Check Metadata
    if 'sample_id' not in df.columns:
        raise ValueError("Master table missing metadata columns: ['sample_id']")

    # 2. Check Required Features (if provided)
    if required_cols:
        missing_feats = [c for c in required_cols if c not in df.columns]
        if missing_feats:
            raise ValueError(f"Master table missing {len(missing_feats)} required features: {missing_feats[:5]}...")