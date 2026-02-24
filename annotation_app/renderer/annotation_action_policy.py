from annotation_app.renderer.annotation_action import AnnotationAction
from annotation_app.renderer.draw_state import DrawState
from annotation_app.renderer.render_data import RenderData


class ActionPolicy:

    @staticmethod
    def evaluate(action: AnnotationAction, data: RenderData):
        
        if action is AnnotationAction.DONE_OR_CREATE_MASK:
            if data.draw_state is DrawState.IDLE:
                action = AnnotationAction.START_CONSTRUCT_MASK

            if data.draw_state is DrawState.DRAWING_MASK or data.draw_state is DrawState.DRAWING_MASK_LASSO:
                action = AnnotationAction.ANNOTATE_MASK

        if action is AnnotationAction.START_CONSTRUCT_MASK_LASSO:
            if data.draw_state is DrawState.DRAWING_MASK_LASSO:
                action = AnnotationAction.ANNOTATE_MASK

        if action is AnnotationAction.UNDO_MASK_POINT:
            if len(data.construct_poly) < 1:
                action = AnnotationAction.IDLE

        if action is AnnotationAction.START_CONSTRUCT_RECTANGLE:
            if data.draw_state is DrawState.DRAWING_MASK:
                action = AnnotationAction.IDLE

        return action