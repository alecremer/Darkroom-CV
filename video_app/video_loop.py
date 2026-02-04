from configs.video_inference_config import VideoInferenceConfig
from video_app.video_executor import VideoExecutor

class VideoLoop:
    def __init__(self, config: VideoInferenceConfig, video_executor: VideoExecutor):
        self.config: VideoInferenceConfig = config
        self.video_executor: VideoExecutor = video_executor
    
    def _handle_callback(self, callback: callable):
        if callback is not None:
            callback()

    def _should_process(self, frame_count: int, frames_to_skip: int):
        if frames_to_skip <= 0:
            return True
        return frame_count % frames_to_skip == 0
    
    def loop(self):
        frame_count: int = 0

        while True:

            frame_count += 1

            if self._should_process(frame_count, self.config.skip_frames):

                self._handle_callback(self.config.loop_start_callback)
                self.video_executor.execute(frame_count)
                self._handle_callback(self.config.loop_end_callback)

            if self.video_executor.should_stop():
                self.video_executor.shutdown()
                break
                