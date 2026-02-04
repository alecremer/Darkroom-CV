from inference_pipeline import InferencePipeline
from model_tasks import Task
from rendering.drawer import Drawer
from inference_result import InferenceResult

class YOLO_InferencePipeline(InferencePipeline):

    def inference(self, frame) -> InferenceResult:
        inference_results = []
        for m in self.models:
            inf_result = InferenceResult()

            inf_result.results = m.model(frame, stream=True, conf=m.confidence)
            inf_result.task = m.task

            inference_results.append(inf_result)
        
        return inference_results