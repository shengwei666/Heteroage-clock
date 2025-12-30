import pytest
import numpy as np
import pandas as pd
from heteroage_clock.core.splits import make_stratified_group_folds

def test_make_stratified_group_folds():
    """Test the stratified group fold function."""
    # Create dummy data for groups and tissues
    groups = np.array([1, 1, 2, 2, 3, 3])
    tissues = np.array(['Blood', 'Brain', 'Liver', 'Heart', 'Kidney', 'Lung'])
    
    # Test with 2 folds
    folds = make_stratified_group_folds(groups, tissues, 2, seed=42)
    
    assert len(folds) == 2  # Should produce 2 folds
    assert len(folds[0][0]) + len(folds[0][1]) == len(groups)  # Split should cover all samples
    assert len(folds[1][0]) + len(folds[1][1]) == len(groups)
    
    # Check if samples are correctly stratified by group
    group_1_samples = np.isin(groups[folds[0][0]], [1])
    assert group_1_samples.sum() == 1  # At least one group 1 sample should be in fold 0
    group_2_samples = np.isin(groups[folds[1][0]], [2])
    assert group_2_samples.sum() == 1  # At least one group 2 sample should be in fold 1

def test_make_stratified_group_folds_with_small_groups():
    """Test with a small number of groups and samples."""
    groups = np.array([1, 2, 3])
    tissues = np.array(['Blood', 'Brain', 'Liver'])
    
    # Test with 3 folds
    folds = make_stratified_group_folds(groups, tissues, 3, seed=42)
    
    assert len(folds) == 3  # Should produce 3 folds
    assert len(folds[0][0]) + len(folds[0][1]) == len(groups)
    assert len(folds[1][0]) + len(folds[1][1]) == len(groups)
    assert len(folds[2][0]) + len(folds[2][1]) == len(groups)
