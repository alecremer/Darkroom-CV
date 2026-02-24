import threading
import torch
from ultralytics import YOLO
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from annotation_app.annotation_opencv_factory import AnnotationOpencvFactory
from annotation_app.annotation_pipeline import AnnotationPipeline, AnnotateModelConfig
from configs.detect_model_config import DetectModelConfig
from entities.model_types import ModelType, Model
from configs.train_model_config import TrainModelConfig
from configs.video_inference_config import VideoInferenceConfig
from models_loader import ModelsLoader
from train_engine.train_engine import TrainEngine
from video_app.video_app import VideoApp


class Engine:

    _train = TrainEngine()

    def train(self, train_cfg_list: List[TrainModelConfig]): 
        self._train.train(train_cfg_list)

    def test(self, weight_path, test_path, show_image = True):
        model_trained = YOLO(weight_path)
        result = model_trained.predict(test_path, show=show_image)[0] 

    def annotate(self, img_path: str, detect_model_config: List[DetectModelConfig]):
        annotation_tool = AnnotationOpencvFactory()
        annotation_tool.annotate(img_path, detect_model_config)

    def live(self, video_config: VideoInferenceConfig, config_path: str):
        video_app = VideoApp(video_config)
        video_app.run()

    

    
    
    