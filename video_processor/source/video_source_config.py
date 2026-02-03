from abc import ABC
from video_source import VideoSource

class VideoSourceConfig(ABC):
    source: VideoSource
    args: dict