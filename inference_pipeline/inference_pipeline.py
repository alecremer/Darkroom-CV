from abc import ABC, abstractmethod
from typing import List
from configs.detect_model_config import DetectModelConfig
from models_loader import ModelsLoader
from inference_result import InferenceResult

class InferencePipeline(ABC):
    def __init__(self, models_config: List[DetectModelConfig]):

        self.models = ModelsLoader.load_models(models_config)

    def inference(self, frame) -> List[InferenceResult]:
        pass
