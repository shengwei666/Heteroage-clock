# __init__.py
"""
Artifacts module for heteroage-clock.

This module handles the loading, saving, and management of artifacts (models, weights, results)
produced during the training process in the biological age prediction pipeline.

Artifacts include:
    - Models: Serialized trained models.
    - Weights: Feature importance weights.
    - Residuals: Stage-wise residual corrections.
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
