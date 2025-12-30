import pytest
import heteroage_clock

def test_import():
    """Test that the heteroage_clock package imports without issues."""
    assert 'heteroage_clock' in globals()

def test_import_core():
    """Test the import of core submodules."""
    from heteroage_clock.core import metrics, sampling, splits
    assert metrics is not None
    assert sampling is not None
    assert splits is not None

def test_import_data():
    """Test the import of data submodules."""
    from heteroage_clock.data import io, assemble
    assert io is not None
    assert assemble is not None
