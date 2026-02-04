from runner import Runner
from metrics import Metrics
from typing import List, Tuple
from config_resolver import ConfigResolver
from video_app.source.video_source import VideoSource
from configs.video_inference_config import VideoInferenceConfig
from configs.detect_model_config import DetectModelConfig

class LiveConfigFactory:

    def _select_source(self, args) -> VideoSource: 
        
        source = VideoSource.SELF
        if args.rtsp:
            source = VideoSource.RTSP
        elif args.file:
            source = VideoSource.FILE
        return source
    
    def set_metrics_active_ais(self, detect_config_list: List[DetectModelConfig], metrics: Metrics) -> Metrics:

        for d in detect_config_list:
            metrics.active_ias = metrics.active_ias + " - " + d.label
        return metrics
    
    def _build_performance_log(self, detect_cfg) -> Metrics:
        metrics = Metrics()
        metrics = self.set_metrics_active_ais(detect_cfg, metrics)
        return metrics

    def build_config(self, args) -> VideoInferenceConfig:

        show_video = not self.args.no_video
        performance_log = self.args.performance_log
        skip_frames = self.args.skip_frames
        capture_objects = self.args.capture_objects
        file_name = self.args.file
        record_name = self.args.record
        record = record_name != ""
        test = self.args.test

        source = self._select_source(args)

        video_config = VideoInferenceConfig(show_video, capture_objects, performance_log, source, file_name, skip_frames, record, record_name)

        return video_config
    
    def _build_runner(self, args) -> Tuple[Runner, VideoInferenceConfig]:

        runner = Runner()

        video_config = self.build_config(args)
        
        
        if video_config.performance_log:
            metrics = self._build_performance_log(video_config)
            self.runner.metrics = metrics
            video_config.loop_start_callback = lambda: metrics.log_performance()

        return (runner, video_config)