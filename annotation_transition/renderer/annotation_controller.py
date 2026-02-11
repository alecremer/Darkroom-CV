from annotation_transition.renderer.action_dispatcher import ActionDispatcher
from annotation_transition.renderer.command_adapter import CommandAdapter
from view.annotation_overlay import AnnotationOverlay
from view.annotation_view import AnnotationView
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
from view.opencv_pipeline import OpencvPipeline

class AnnotationController:

    def __init__(self, labels: List[str]):
        
        self.view = AnnotationView(labels)
        self.overlay = AnnotationOverlay()
        self.data = RenderData()

        self.action_handler = ActionHandler(self.data)
        self.command_adapter = CommandAdapter()
        self.action_dispatcher = ActionDispatcher(self.action_handler, self.command_adapter)
        self.policy = InteractionPolicy()
        self.mouse_handler = MouseHandler(self.data, self.action_dispatcher, self.policy)
        self.keyboard_handler = KeyboardHandler(self.action_dispatcher)
        self.view_pipeline = OpencvPipeline(self.data, self.mouse_handler, self.keyboard_handler, self.view, self.overlay, labels)


    def run(self):
        while True:
            self.view_pipeline.routine(self.data.img)