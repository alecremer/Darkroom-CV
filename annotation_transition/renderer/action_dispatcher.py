from typing import Any
from annotation_transition.renderer.action_handler import ActionHandler
from annotation_transition.renderer.annotation_action import AnnotationAction
from annotation_transition.renderer.command_adapter import CommandAdapter


class ActionDispatcher:
    def __init__(self, handler: ActionHandler, command_adapter: CommandAdapter):
        self.handler = handler
        self.command_adapter = command_adapter

    def dispatch(self, action: AnnotationAction, payload: Any):
        if self.handler.can_handle(action):
            self.handler.handle(action, payload)
        else:
            self.command_adapter.send(action, payload)
