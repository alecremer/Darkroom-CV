from typing import Any
from annotation_transition.renderer.annotation_action import AnnotationAction
from annotation_transition.renderer.render_data import RenderData
from entities.entities import PolygonalMask, Rectangle

class PayloadHandler:

    @staticmethod
    def handle(action: AnnotationAction, data: RenderData, payload: Any = None, ):
        
        if action is AnnotationAction.ANNOTATE_BBOX:
            new_payload = Rectangle(data.construct_box.p0, payload)
            return new_payload
        
        elif action is AnnotationAction.ANNOTATE_MASK:
            mask = PolygonalMask(
                    label=data.label,
                    points=data.construct_poly
                )
            new_payload = mask
            return new_payload
        
        return payload