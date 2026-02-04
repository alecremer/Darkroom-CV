from dataclasses import dataclass
from video_app.source.video_source import VideoSource

@dataclass
class VideoInferenceConfig:

    show_video: bool = True
    capture_objects: bool = False
    performance_log: bool = False
    source: VideoSource = VideoSource.SELF
    file: str = None
    skip_frames: int = 0
    record: bool = False
    record_file_name: str = None
    ip: str = None
    loop_start_callback: callable = None
    loop_end_callback: callable = None
    fps: int = 24