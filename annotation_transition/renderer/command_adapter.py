from typing import Any
from annotation_transition.renderer.action_mapper import ActionMapper
from annotation_transition.renderer.annotation_action import AnnotationAction
from annotation_transition.renderer.render_data import RenderData

class CommandAdapter:
    def __init__(self, comm: callable):
        self.comm = comm

    def send(self, action: AnnotationAction, payload: Any = None):

        action_mapped = ActionMapper.map(action)        
        response = self.comm(action_mapped, payload)

        data: RenderData = RenderData()
        data.annotations = response.annotations
        data.current_annotation = response.current_annotation
        data.img = response.img
        data.label = response.label

        return data
