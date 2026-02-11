from annotation_overlay import AnnotationOverlay
from annotation_view import AnnotationView
from draw_state import DrawState
from input_intent import InputIntent
from interaction_policy import InteractionPolicy
from keyboard_handler import KeyboardHandler
from mouse_handler import MouseHandler
from typing import List
from annotation_transition.render_command import RenderCommand
from annotation_transition.render_intent import RenderIntent
from action_handler import ActionHandler
from render_data import RenderData

class AnnotationController:

    def __init__(self, labels: List[str]):
        
        self.draw_state = DrawState.IDLE
        self.policy = InteractionPolicy()
        self.view = AnnotationView(labels)
        self.overlay = AnnotationOverlay()
        self.data = RenderData()
        self.action_handler = ActionHandler(self.data)

    def routine(self, cmd: RenderCommand):

        self.action_handler.handle(cmd.intent, cmd.payload)