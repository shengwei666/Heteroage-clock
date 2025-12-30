"""
heteroage_clock.stages

This module exposes the core training and inference functions for each stage
of the biological age prediction pipeline.
"""

from .stage1 import train_stage1, predict_stage1
from .stage2 import train_stage2, predict_stage2
from .stage3 import train_stage3, predict_stage3

__all__ = [
    "train_stage1", 
    "predict_stage1",
    "train_stage2", 
    "predict_stage2",
    "train_stage3", 
    "predict_stage3",
]