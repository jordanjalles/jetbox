"""Training objectives for JEPA."""

from core.objectives.jepa_objectives import (
    ContrastiveLoss,
    JEPAObjective,
    LatentPredictionLoss,
    create_mask_block,
    create_mask_random,
)

__all__ = [
    "ContrastiveLoss",
    "JEPAObjective",
    "LatentPredictionLoss",
    "create_mask_block",
    "create_mask_random",
]
