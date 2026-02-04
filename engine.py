import threading
import torch
from ultralytics import YOLO
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from annotation_tool import AnnotationTool, AnnotateModelConfig
from model_types import ModelType, Model
from configs.train_model_config import TrainModelConfig
from configs.video_inference_config import VideoInferenceConfig
from video_app.video_app import VideoApp



class Train:

    @classmethod
    def train_one_model(self, train_cfg: TrainModelConfig):
        
        # parse
        epochs = train_cfg.epochs
        path = train_cfg.dataset_path
        device = train_cfg.device
        model = train_cfg.model
        results_folder_name = train_cfg.results_folder_name

        model = YOLO("models/" + model)

        if train_cfg.model_par_config:

            results = model.train(data=(path + "/data.yaml"), device=device, 
                            project=path + "/runs", name=results_folder_name, **train_cfg.model_par_config)
        else:
            results = model.train(data=(path + "/data.yaml"), epochs=epochs, device=device, 
                            project=path + "/runs", name=results_folder_name, patience=50)
        model.val()

    @classmethod
    def train(self, train_cfg_list: List[TrainModelConfig]):

        if train_cfg_list and len(train_cfg_list) > 0:
            
            for cfg in train_cfg_list:

                self.train_one_model(cfg)

        else:
            raise("train configuration could not be empty")

class Engine:

    
    _train = Train()

    # def __init__(self):
    #     # logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
    #     self.create_rectangle = False
    #     self.drawing_rectangle = False
    #     self.show_ui = True
    #     self.x_y_mouse = 0,0

    def train(self, train_cfg_list: List[TrainModelConfig]): 
        self._train.train(train_cfg_list)
    

    def test(self, weight_path, test_path, show_image = True):
        model_trained = YOLO(weight_path)
        result = model_trained.predict(test_path, show=show_image)[0] 

    def annotate(self, img_path: str, annotate_model_config: List[AnnotateModelConfig]):
        annotation_tool = AnnotationTool()
        weight_path = [config.weights_path for config in annotate_model_config]
        models_trained = self._set_trained_models(weight_path)
        annotation_tool.annotate(img_path, annotate_model_config, models_trained)

    def live(self, video_config: VideoInferenceConfig, config_path: str):
        video_app = VideoApp(video_config)
        video_app.run()

    

    
    
    