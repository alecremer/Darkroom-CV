from typing import List
from configs.detect_model_config import DetectModelConfig
from models_loader import ModelsLoader
from inference_result import InferenceResult

class InferenceRunner():
    def __init__(self, models_config: List[DetectModelConfig]):

        self.models = ModelsLoader.load_models(models_config)

    def inference(self, frame) -> InferenceResult:
        inference_results = []
        for m in self.models:
            inf_result = InferenceResult()

            inf_result.results = m.model(frame, stream=True, conf=m.confidence)
            inf_result.task = m.task

            inference_results.append(inf_result)
        
        return inference_results