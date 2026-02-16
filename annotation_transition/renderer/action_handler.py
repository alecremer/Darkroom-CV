from typing import Any

from annotation_transition.renderer.annotation_action import AnnotationAction
from annotation_transition.renderer.draw_state import DrawState
from annotation_transition.renderer.render_data import RenderData
from entities.entities import Point, Rectangle

class ActionHandler:
    def __init__(self, render_data: RenderData, on_quit_requested: callable):
        self.render_data = render_data
        self.on_quit_requested = on_quit_requested

    def can_handle(self, action: AnnotationAction) -> bool:
        return action in {AnnotationAction.START_CONSTRUCT_RECTANGLE,
                          AnnotationAction.DRAW_CONSTRUCT_RECTANGLE,
                          AnnotationAction.START_CONSTRUCT_MASK,
                          AnnotationAction.DRAW_CONSTRUCT_MASK,
                          AnnotationAction.ANNOTATE_MASK,
                          AnnotationAction.CANCEL_CONSTRUCT_MASK,
                          AnnotationAction.CANCEL_CONSTRUCT_BOX,
                          AnnotationAction.TOGGLE_SHOW_UI,
                          AnnotationAction.SELECT_LABEL,
                          AnnotationAction.QUIT,
                          AnnotationAction.UNDO_MASK_POINT,
                          AnnotationAction.ANNOTATE_BBOX}

    def handle(self, action: AnnotationAction, payload: Any):

        if action is AnnotationAction.START_CONSTRUCT_RECTANGLE:
            p = self.render_data.mouse_xy
            self.render_data.construct_box = Rectangle(p, p)
            self.render_data.draw_state = DrawState.DRAWING_RECTANGLE

        elif action is AnnotationAction.ANNOTATE_BBOX:
            self.render_data.construct_box = None
            self.render_data.draw_state = DrawState.IDLE

        elif action is AnnotationAction.DRAW_CONSTRUCT_RECTANGLE:
            self.render_data.construct_box.p=payload

        elif action is AnnotationAction.CANCEL_CONSTRUCT_MASK:
            self.render_data.construct_poly = []
            self.render_data.draw_state = DrawState.IDLE

        elif action is AnnotationAction.CANCEL_CONSTRUCT_BOX:
            self.render_data.construct_box = None

        elif action is AnnotationAction.TOGGLE_SHOW_UI:
            self.render_data.show_ui = not self.render_data.show_ui 

        elif action is AnnotationAction.START_CONSTRUCT_MASK:
            self.render_data.draw_state = DrawState.DRAWING_MASK
        
        elif action is AnnotationAction.ANNOTATE_MASK:
            self.render_data.draw_state = DrawState.IDLE
            self.render_data.construct_poly = []

        elif action is AnnotationAction.DRAW_CONSTRUCT_MASK:
            self.render_data.construct_poly.append(payload)

        elif action is AnnotationAction.UNDO_MASK_POINT:
            self.render_data.construct_poly.pop()

        elif action is AnnotationAction.SELECT_LABEL:
            self.render_data.select_label = True

        elif action is AnnotationAction.QUIT:
            self.on_quit_requested()
