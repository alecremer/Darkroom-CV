from video_app.video_loop import VideoLoop
from configs.video_inference_config import VideoInferenceConfig
from video_executors.opencv_video_executor import OpenCV_VideoExecutor
from inference_runners.inference_runner import InferenceRunner
from video_app.source.video_source_config import VideoSourceConfig
from video_app.video_config_converter import VideoConfigConverter
from rendering.opencv_drawer import OpenCV_Drawer

class VideoApp:
    
    def __init__(self, config: VideoInferenceConfig):
        
        video_source_config = VideoConfigConverter.video_inference_to_source_config(config)

        inference_pipeline = InferenceRunner()
        renderer = OpenCV_Drawer()
        executor = OpenCV_VideoExecutor(config, inference_pipeline, video_source_config, renderer)
        self.looper = VideoLoop(config, executor)

    def run(self):
        self.looper.loop()
