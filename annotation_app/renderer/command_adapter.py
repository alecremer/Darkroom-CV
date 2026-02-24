from typing import Any
from annotation_app.renderer.action_mapper import ActionMapper
from annotation_app.renderer.annotation_action import AnnotationAction
from annotation_app.renderer.render_data import RenderData

class CommandAdapter:
    def __init__(self, comm: callable):
        self.comm = comm

    def send(self, action: AnnotationAction, payload: Any = None):

        action_mapped = ActionMapper.map(action)        
        response = self.comm(action_mapped, payload)

        return response
