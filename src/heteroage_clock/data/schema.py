"""
heteroage_clock.data.schema

This module defines the expected schema of input data.
Updates: Aligned column naming conventions with the new suffix logic (_beta, _chalm, _camda).
"""

from typing import List

class DataSchema:
    """
    Defines the expected columns for each modality.
    """

    @staticmethod
    def get_cpg_columns() -> List[str]:
        """
        Returns the list of expected CpG columns (Base ID).
        Example: cg00000029
        """
        return [f"cg{str(i).zfill(8)}" for i in range(1, 1001)]

    @staticmethod
    def get_beta_columns() -> List[str]:
        """
        Returns the list of expected Beta columns (Suffix: _beta).
        """
        return [f"cg{str(i).zfill(8)}_beta" for i in range(1, 1001)]

    @staticmethod
    def get_chalm_columns() -> List[str]:
        """
        Returns the list of expected CHALM columns (Suffix: _chalm).
        """
        return [f"cg{str(i).zfill(8)}_chalm" for i in range(1, 1001)]

    @staticmethod
    def get_camda_columns() -> List[str]:
        """
        Returns the list of expected CAMDA columns (Suffix: _camda).
        """
        return [f"cg{str(i).zfill(8)}_camda" for i in range(1, 1001)]

    @staticmethod
    def get_pc_columns() -> List[str]:
        """
        Returns the list of expected RF_PC columns.
        """
        return [f"RF_PC{i}" for i in range(1, 21)]

    @staticmethod
    def get_metadata_columns() -> List[str]:
        """
        Returns the expected metadata columns.
        """
        return ['sample_id', 'project_id', 'Tissue', 'Age', 'Sex', 'Is_Healthy']