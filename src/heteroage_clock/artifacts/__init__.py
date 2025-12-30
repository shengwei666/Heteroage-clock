"""
heteroage_clock.artifacts

Artifact management classes for handling model serialization and IO.
"""

from .base import BaseArtifact
from .stage1 import Stage1Artifact
from .stage2 import Stage2Artifact
from .stage3 import Stage3Artifact

__all__ = [
    "BaseArtifact",
    "Stage1Artifact",
    "Stage2Artifact",
    "Stage3Artifact",
]