from typing import Any, List
from annotation_transition.annotation_pipeline import AnnotationPipeline
from annotation_transition.engine_action import AnnotationEngineAction
from annotation_transition.renderer.annotation_action import AnnotationAction
from annotation_transition.renderer.annotation_controller import AnnotationController
from configs.annotate_model_config import AnnotateModelConfig
from entities.model_types import Model


class AnnotationOpencvFactory:

    def annotate(self, img_path: str, annotate_model_config: List[AnnotateModelConfig], models_trained: List[Model]):
        self.pipeline = AnnotationPipeline(img_path, annotate_model_config, models_trained)
        comm = lambda action, payload:  self.comm_handler(action, payload)
        self.interface = AnnotationController(self.pipeline.labels, comm)

        self.interface.run()

    def comm_handler(self, action: str, payload: Any):
        action_mapped = AnnotationEngineAction(action)
        self.pipeline.run(action_mapped)
        return self.pipeline.data
