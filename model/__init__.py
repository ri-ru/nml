"""Shared trajectory-prediction model — single source of truth for the architecture."""
from .architecture import (
    NBADataset,
    NBASampler,
    MoGNLLLoss,
    MultiStepMSE,
    GRUCell,
    TypeAwareDistanceBiasedAttention,
    NBAv46Model,
    Trainer,
    K_MIX,
)
from .variants import BallWeightedMoGNLLLoss, build_member

__all__ = [
    "NBADataset", "NBASampler", "MoGNLLLoss", "MultiStepMSE", "GRUCell",
    "TypeAwareDistanceBiasedAttention", "NBAv46Model", "Trainer", "K_MIX",
    "BallWeightedMoGNLLLoss", "build_member",
]
