from video_app.video_executor import VideoExecutor
from inference_runners.inference_runner import InferenceRunner
from video_app.source.opencv_video_source import OpenCV_VideoSourceTools
from video_app.source.video_source_config import VideoSourceConfig
from configs.video_inference_config import VideoInferenceConfig
import cv2
from rendering.renderer import Renderer
from inference_runners.inference_result import InferenceResult

class OpenCV_VideoExecutor(VideoExecutor):

    def __init__(self, config: VideoInferenceConfig, inference_pipeline: InferenceRunner, video_source_config: VideoSourceConfig, renderer: Renderer):
        super().__init__(config, renderer)
        self.inference_pipeline: InferenceRunner = inference_pipeline
        self.cam = OpenCV_VideoSourceTools._select_video_source_(video_source_config)
        self.video_source_config = video_source_config
        self.frame_stream = []
        
    def execute(self, frame_count):

        check, frame = self.cam.read()
        if not check:
            return
        
        result: InferenceResult = self.inference_pipeline.inference(frame)
        frame_inference = self.renderer.render_results(result)

        if self.config.show_video:
            self.frame_stream.append(frame_inference)
            cv2.imshow('video', frame_inference)

    def _save_recorded_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_file = self.config.record_file_name + ".mkv"  
        fps = self.video_source_config.args.get('fps', 30)
        frame_size = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)  
        for f in self.frame_stream:
            out.write(f)

        out.release()
    
    def shutdown(self):
        if(self.config.record):
            self._save_recorded_video()
        self.cam.release()
        cv2.destroyAllWindows()
    
    def should_stop(self):
        return cv2.waitKey(1) == 27 # ESC