from dataclasses import dataclass, field, fields
from annotation_app.annotation_cell import AnnotationCell
from annotation_app.renderer.draw_state import DrawState
from entities.entities import BoundingBox, PolygonalMask, Rectangle, Point
from typing import Any, List

@dataclass
class RenderData:
    construct_box: Rectangle | None = None
    construct_poly: List[Point] = field(default_factory=list)
    # annotations: List[AnnotationCell] = field(default_factory=list)
    current_annotation: AnnotationCell | None = None
    file_index: int = 0
    label: str = ""
    show_ui: bool = False
    select_label: bool = False
    img: Any | None = None
    mouse_xy: Point | None = field(default_factory= lambda: Point(0, 0))
    draw_state: DrawState = DrawState.IDLE
    pixel_lasso_dist = 10
    num_imgs_total: int = 0
    num_imgs_annotated: int = 0

    def update_from(self, other: "RenderData") -> None:
        # if not isinstance(other, type(self)):
            # raise TypeError("update_from requires same type, type: " + str(type(other)))
        other_fields = {f.name for f in fields(other)}

        for f in fields(self):
            if f.name in other_fields:
                setattr(self, f.name, getattr(other, f.name))
