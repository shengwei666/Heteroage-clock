"""
heteroage_clock.data.schema

This module defines the expected schema of input data (e.g., columns and feature types).
It ensures that data is structured consistently across different stages of the pipeline.
"""

from typing import List


# Define the expected feature set for each modality
class DataSchema:
    """
    Defines the expected columns for each modality (CpG beta, CHALM, CAMDA, PCs).
    These schemas help ensure that the incoming data is structured correctly.
    """

    @staticmethod
    def get_cpg_columns() -> List[str]:
        """
        Returns the list of expected CpG columns (e.g., cg00000029).

        Returns:
            List[str]: List of CpG column names.
        """
        return [f"cg{str(i).zfill(8)}" for i in range(1, 1001)]  # Example columns

    @staticmethod
    def get_chalm_columns() -> List[str]:
        """
        Returns the list of expected CHALM columns.

        Returns:
            List[str]: List of CHALM feature columns.
        """
        return [f"chalm_{i}" for i in range(1, 51)]  # Example columns

    @staticmethod
    def get_camda_columns() -> List[str]:
        """
        Returns the list of expected CAMDA columns.

        Returns:
            List[str]: List of CAMDA feature columns.
        """
        return [f"camda_{i}" for i in range(1, 51)]  # Example columns

    @staticmethod
    def get_pc_columns() -> List[str]:
        """
        Returns the list of expected RF_PC columns.

        Returns:
            List[str]: List of Principal Component columns.
        """
        return [f"RF_PC{i}" for i in range(1, 21)]  # Example columns

    @staticmethod
    def get_metadata_columns() -> List[str]:
        """
        Returns the expected metadata columns.

        Returns:
            List[str]: List of metadata columns (e.g., sample_id, tissue, age).
        """
        return ['sample_id', 'project_id', 'tissue', 'age', 'sex']
