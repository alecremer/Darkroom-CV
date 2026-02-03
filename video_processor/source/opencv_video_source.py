import cv2
from video_source import VideoSource
from video_source_config import VideoSourceConfig


class OpenCV_VideoSourceTools:

    @classmethod
    def _select_video_source_(self: str, source_config: VideoSourceConfig):
        
        source = source_config.source
        args = source_config.args

        if source.value is VideoSource.SELF.value:
            
            source = cv2.VideoCapture(0)
        
        elif source.value == VideoSource.RTSP:
            
            try:
                source = cv2.VideoCapture("rtsp://" + args['ip'])
            except KeyError as e:
                raise(f"can't find rtsp source or rtsp ip: {e}")
            
            source.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            source.set(cv2.CAP_PROP_FPS, args.get('fps', 30))

        elif source.value == VideoSource.FILE:
            try:
                source = cv2.VideoCapture(args['file'])
            except KeyError as e:
                raise(f"cant find file: {e}") 

        return source