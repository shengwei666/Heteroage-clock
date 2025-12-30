"""
heteroage_clock.utils.contract

This module defines the contract management utility, which is used for handling the schema and column
contract in the pipeline. It ensures consistency between training and inference inputs and outputs,
preventing issues related to missing or misaligned columns.
"""

from typing import List, Dict


def create_contract(columns: List[str]) -> Dict[str, str]:
    """
    Creates a contract that maps column names to their expected types.

    Args:
        columns (List[str]): List of expected column names.

    Returns:
        Dict[str, str]: A dictionary mapping each column to its expected type.
    """
    contract = {col: "float32" for col in columns}
    return contract


def validate_contract(input_columns: List[str], contract: Dict[str, str]) -> bool:
    """
    Validates that the input columns match the expected contract.

    Args:
        input_columns (List[str]): List of input columns to check.
        contract (Dict[str, str]): The expected contract.

    Returns:
        bool: True if input columns match the contract, otherwise False.
    """
    contract_columns = set(contract.keys())
    input_columns_set = set(input_columns)

    if input_columns_set != contract_columns:
        missing = contract_columns - input_columns_set
        extra = input_columns_set - contract_columns
        print(f"Missing columns: {missing}")
        print(f"Extra columns: {extra}")
        return False
    return True
