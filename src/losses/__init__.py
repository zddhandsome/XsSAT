# GeoSATformer v2 Losses Module

from .multitask_loss import (
    MultiTaskLoss,
    SATClassificationLoss,
    UNSATCoreLoss,
)

__all__ = [
    'MultiTaskLoss',
    'SATClassificationLoss',
    'UNSATCoreLoss',
]
