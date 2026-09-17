"""
Training loops, optimization routines, and MLP baseline trainer.
"""

from src.training.train_mlp import MLPTrainer
from src.training.contrastive import (
    NTXentLoss,
    GraphContrastiveModel,
    ContrastiveTrainer,
)

__all__ = ["MLPTrainer", "NTXentLoss", "GraphContrastiveModel", "ContrastiveTrainer"]
