from configs.video_inference_config import VideoInferenceConfig
from video_app.source.video_source_config import VideoSourceConfig

class VideoConfigConverter:
    @classmethod
    def video_inference_to_source_config(self, video_inference_config: VideoInferenceConfig) -> VideoSourceConfig:
        args: dict = {}
        args['file'] = video_inference_config.file
        args['ip'] = video_inference_config.ip
        args['fps'] = video_inference_config.fps

        video_source_config = VideoSourceConfig()
        video_source_config.source = video_inference_config.source
        video_source_config.args = args

        return video_source_config

