from dataclasses import dataclass, field
from annotation_transition.annotation_cell import AnnotationCell
from annotation_transition.renderer.draw_state import DrawState
from types.entities import BoundingBox, PolygonalMask, Rectangle, Point
from typing import Any, List

@dataclass
class RenderData:
    construct_box: Rectangle = Rectangle()
    construct_poly: List[Point] = field(default_factory=list)
    annotations: List[AnnotationCell]
    current_annotation: AnnotationCell
    annotation_index: int = 0
    label: str
    show_ui: bool
    img: Any
    mouse_xy: Point
    draw_state: DrawState = DrawState.IDLE