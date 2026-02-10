from dataclasses import dataclass
import numpy as np
from typing import List
from enum import Enum
from types.entities import BoundingBox, PolygonalMask

@dataclass
class AnnotationCell:

    id: str
    img: np.ndarray
    original_img: np.ndarray
    classes_boxes: List[BoundingBox]
    classes_masks: List[PolygonalMask]
    excluded_classes_boxes: List[BoundingBox]
    excluded_classes_masks: List[PolygonalMask]
    valid: bool

