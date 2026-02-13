from typing import Any
from annotation_transition.renderer.action_handler import ActionHandler
from annotation_transition.renderer.action_mapper import ActionMapper
from annotation_transition.renderer.annotation_action import AnnotationAction
from annotation_transition.renderer.annotation_action_policy import ActionPolicy
from annotation_transition.renderer.command_adapter import CommandAdapter
from annotation_transition.renderer.payload_handler import PayloadHandler
from annotation_transition.renderer.render_data import RenderData


class ActionDispatcher:
    def __init__(self, handler: ActionHandler, command_adapter: CommandAdapter, data: RenderData):
        self.handler = handler
        self.command_adapter = command_adapter
        self.data: RenderData = data

    def dispatch(self, action: AnnotationAction, payload: Any = None):
        
        action = ActionPolicy.evaluate(action, self.data)
        payload = PayloadHandler.handle(action, self.data, payload)
        
        if ActionMapper.can_map(action):
            print(f"action {action}")
            print(f"show ui before: {self.data.show_ui}")
            d = self.command_adapter.send(action, payload)
            self.data.update_from(d)
            print(f"show ui after: {self.data.show_ui}")

        if self.handler.can_handle(action):
            self.handler.handle(action, payload)
        