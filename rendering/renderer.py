from abc import ABC, abstractmethod
from inference_runners.inference_result import InferenceResult
from typing import List

class Renderer(ABC):

    @abstractmethod
    def create_masks_in_frame(self, result, frame, label):
        pass

    @abstractmethod
    def create_bounding_box_in_frame(self, detection_result, frame, label):
        pass

    @abstractmethod
    def render_results(self, inference_results: List[InferenceResult], frame, label):
        pass