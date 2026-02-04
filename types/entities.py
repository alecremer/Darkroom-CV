from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class BoundingBox:

    label: str
    box: Tuple[int, int, int, int]
    confidence: float

@dataclass
class Point:
    x: int
    y: int
    
@dataclass
class PolygonalMask:
    label: str
    points: List[Point]
