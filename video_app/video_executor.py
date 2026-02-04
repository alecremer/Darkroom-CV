from abc import ABC, abstractmethod
from configs.video_inference_config import VideoInferenceConfig
from rendering.drawer import Drawer

class VideoExecutor(ABC):

    def __init__(self, config: VideoInferenceConfig, renderer: Drawer):
        self.config = config
        self.renderer = renderer

    @abstractmethod
    def execute(self, frame_count: int):
        pass
    
    @abstractmethod
    def should_stop(self) -> bool:
        pass

    @abstractmethod
    def shutdown(self):
        pass

