"""
heteroage_clock.utils.hashing

This module provides utility functions for generating hashes of data, ensuring reproducibility
and unique identification of inputs, models, and outputs.
"""

import hashlib
import json


def hash_object(obj: object) -> str:
    """
    Generate a hash for an arbitrary Python object by serializing it to JSON.

    Args:
        obj (object): The object to hash.

    Returns:
        str: A unique hash of the object.
    """
    obj_json = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(obj_json.encode("utf-8")).hexdigest()


def hash_file(file_path: str) -> str:
    """
    Generate a hash for a file based on its content.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: A unique hash of the file.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
