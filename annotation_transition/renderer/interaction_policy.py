from annotation_transition.annotation_renderer.annotation_action import AnnotationAction
from draw_state import DrawState
from annotation_transition.annotation_renderer.annotation_action import AnnotationAction
from annotation_transition.annotation_renderer.input_intent import InputIntent
from dataclasses import dataclass

@dataclass(frozen=True)
class PolicyResult:
    action: AnnotationAction
    next_state: DrawState | None

class InteractionPolicy:
    
    @staticmethod
    def decide(state: DrawState, intent: InputIntent) -> PolicyResult:

        action: AnnotationAction = AnnotationAction.IDLE
        next_state: DrawState | None = None

        if intent is InputIntent.MOVE:
                
            if state is DrawState.DRAWING_RECTANGLE:
                action = AnnotationAction.DRAW_CONSTRUCT_RECTANGLE

        elif intent is InputIntent.LMB_UP:

            if state is DrawState.DRAWING_RECTANGLE:
                action = AnnotationAction.ANNOTATE_BBOX

        elif intent is InputIntent.RMB_UP:

            if state is DrawState.DRAWING_MASK:
                action = AnnotationAction.CANCEL_CONSTRUCT_POLY

        elif intent is InputIntent.LMB_DOWN:

            if state is DrawState.STARTING_RECTANGLE:
                action = AnnotationAction.START_CONSTRUCT_RECTANGLE
                next_state = DrawState.DRAWING_RECTANGLE

            elif state is DrawState.STARTING_MASK:
                action = AnnotationAction.START_CONSTRUCT_MASK
                next_state = DrawState.DRAWING_MASK

            elif state is DrawState.IDLE:
                action = AnnotationAction.EXCLUDE_CLICKED_ENTITY

        

        return PolicyResult(action, next_state)