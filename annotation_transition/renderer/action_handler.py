from render_data import RenderData
from annotation_action import AnnotationAction
from typing import Any

class ActionHandler:
    def __init__(self, render_data: RenderData):
        self.render_data = render_data

    def can_handle(self, action: AnnotationAction) -> bool:
        return action in {AnnotationAction.START_CONSTRUCT_RECTANGLE,
                          AnnotationAction.DRAW_CONSTRUCT_RECTANGLE,
                          AnnotationAction.CANCEL_CONSTRUCT_MASK,
                          AnnotationAction.TOGGLE_SHOW_UI,
                          AnnotationAction.ANNOTATE_BBOX}

    def handle(self, action: AnnotationAction, payload: Any):

        if action is AnnotationAction.START_CONSTRUCT_RECTANGLE:
            self.render_data.construct_box.p0=payload

        elif action is AnnotationAction.ANNOTATE_BBOX:
            self.render_data.construct_box = None

        elif action is AnnotationAction.DRAW_CONSTRUCT_RECTANGLE:
            self.render_data.construct_box.p=payload

        elif action is AnnotationAction.CANCEL_CONSTRUCT_MASK:
            self.render_data.construct_poly = []

        elif action is AnnotationAction.TOGGLE_SHOW_UI:
            self.render_data.show_ui = not self.render_data.show_ui 
