from abc import ABC, abstractmethod
from configs.video_inference_config import VideoInferenceConfig

class VideoExecutor(ABC):

    def __init__(self, config: VideoInferenceConfig):
        self.config = config

    @abstractmethod
    def execute(self, frame_count: int):
        pass
    
    @abstractmethod
    def should_stop(self) -> bool:
        pass

    @abstractmethod
    def shutdown(self):
        pass

