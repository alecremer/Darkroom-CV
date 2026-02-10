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

    def __iter__(self):
        yield self.x
        yield self.y

@dataclass
class Rectangle:
    p0: Point
    p: Point

    def __iter__(self):
        yield self.p0
        yield self.p

    def to_coords(self):
        return self.p0.x, self.p0.y, self.p.x, self.p.y
    
@dataclass
class PolygonalMask:
    label: str
    points: List[Point]
