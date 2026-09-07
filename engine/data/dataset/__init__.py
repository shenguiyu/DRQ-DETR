"""COCO-style detection datasets and evaluators."""

from .coco_dataset import (
    CocoDetection,
    mscoco_category2label,
    mscoco_category2name,
    mscoco_label2category,
)
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset

__all__ = [
    "CocoDetection",
    "CocoEvaluator",
    "get_coco_api_from_dataset",
    "mscoco_category2label",
    "mscoco_category2name",
    "mscoco_label2category",
]
