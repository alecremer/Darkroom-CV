from typing import Any
from annotation_transition.renderer.annotation_action import AnnotationAction

class CommandAdapter:
    def send(self, action: AnnotationAction, payload: Any):
        raise NotImplementedError
