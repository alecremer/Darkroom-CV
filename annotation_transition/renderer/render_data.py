from dataclasses import dataclass, field
from types.entities import BoundingBox, PolygonalMask, Rectangle, Point
from typing import List

@dataclass
class RenderData:
    construct_box: Rectangle = Rectangle()
    construct_poly: List[Point] = field(default_factory=list)