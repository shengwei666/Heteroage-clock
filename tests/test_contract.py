import pytest
import pandas as pd
from heteroage_clock.utils.contract import validate_master_table

def test_validate_master_table_valid():
    """Test that a valid master table passes the validation."""
    data = {
        'sample_id': [1, 2, 3],
        'Tissue': ['Blood', 'Brain', 'Liver'],
        'Sex': ['M', 'F', 'M'],
        'RF_PC1': [0.1, 0.2, 0.3],
        'RF_PC2': [0.4, 0.5, 0.6],
        'cg00000029_beta': [0.7, 0.8, 0.9]
    }
    df = pd.DataFrame(data)
    
    # Assuming validate_master_table is the function to check input table
    assert validate_master_table(df) is None  # Should pass without errors

def test_validate_master_table_missing_column():
    """Test that a missing required column raises an error."""
    data = {
        'sample_id': [1, 2, 3],
        'Tissue': ['Blood', 'Brain', 'Liver'],
        'Sex': ['M', 'F', 'M'],
        'RF_PC1': [0.1, 0.2, 0.3]
    }
    df = pd.DataFrame(data)

    # Check that it raises an error for missing columns like 'cg00000029_beta'
    with pytest.raises(ValueError):
        validate_master_table(df)
