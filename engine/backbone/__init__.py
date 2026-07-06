"""Backbone registrations used by DRQ-DETR."""

from .common import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .hgnetv2 import HGNetv2

__all__ = [
    "get_activation",
    "FrozenBatchNorm2d",
    "freeze_batch_norm2d",
    "HGNetv2",
]
