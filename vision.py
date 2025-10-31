import threading
import torch
from ultralytics import YOLO
import cv2
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from annotation_tool import AnnotationTool, AnnotateModelConfig
@dataclass
class DetectConfig:

    show_video: bool = True
    capture_objects: bool = False
    performance_log: bool = False
    source: str = "self"
    file: str = None
    skip_frames: int = 0
    record: bool = False
    record_file_name: str = None
    ip: str = None
    loop_start_callback: callable = None
    loop_end_callback: callable = None

@dataclass
class DetectModelConfig:

    weights_path: str
    label: str
    confidence: float
    device: str
    test_path: str | None
    segmentation: bool = False



@dataclass
class TrainModelConfig:

    dataset_path: str
    epochs: int
    label: str
    device: str
    model: str
    results_folder_name: str
    model_par_config: dict

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

class Vision:

    
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
        print(weight_path)
        print(test_path)
        print(show_image)
        model_trained = YOLO(weight_path)
        result = model_trained.predict(test_path, show=show_image)[0] 

    def annotate(self, img_path: str, annotate_model_config: List[AnnotateModelConfig]):
        annotation_tool = AnnotationTool()
        weight_path = [config.weights_path for config in annotate_model_config]
        models_trained = self._set_trained_models(weight_path)
        annotation_tool.annotate(img_path, annotate_model_config, models_trained)

    def _select_cam_source_(self, source, ip=None, file=None):

        if source == "self":
            
            cam = cv2.VideoCapture(0)
        
        elif source == "rtsp":
            
            print("run rtsp ip: " + ip)
            cam = cv2.VideoCapture("rtsp://" + ip)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cam.set(cv2.CAP_PROP_FPS, 30)

        elif source == "file":
            cam = cv2.VideoCapture(file)

        return cam
    
    def _set_trained_models(self, weight_paths):

        print(f"weights_path: {weight_paths}")
        models_trained = []

        
        
        for p in weight_paths:
            
            if p:
                print(p)
                model_trained = YOLO(p)
                model_trained.verbose = False
                models_trained.append(model_trained)
        print("----------------------------------------")
        
        
        # if not models_trained:
            # raise Exception("weights paths are empty")

        return models_trained


    def _live_detection_loop(self, cam, confidence, labels, segmentation, models_trained, config: DetectConfig):

        result = None
        
        frame_count = 0

        process_frame = False
        frame_stream = []
        
        while True:
            

            if config.skip_frames != 0:
                
                if frame_count % config.skip_frames == 0:

                    process_frame = True
                    
                    frame_count = 0
                else:
                    process_frame = False

                frame_count += 1
            else:
                
                process_frame = True
              
            index = 0

                    
            check, frame = cam.read()
            if process_frame:


                if config.loop_start_callback:
                    config.loop_start_callback()
                
                for m in models_trained:
                    # result = m(frame, stream=True, conf=0.65)
                    result = m(frame, stream=True, conf=confidence[index])
                    
                    if config.capture_objects:
                        objects = []
                        for r in result:
                            boxes = r.boxes
                            # for box in boxes:
                            #     objects.append(box.xyxy[0]) 
                    if config.show_video:

                        if segmentation[index]:
                            frame = self.create_masks_in_frames(result, frame, labels[index])
                        else:
                            self.create_bounding_box(result, frame, labels[index])
                        
                            

                    index = index + 1

                if config.show_video:
                    frame_stream.append(frame)
                    cv2.imshow('video', frame)

                if config.loop_end_callback:
                    config.loop_end_callback()



            key = cv2.waitKey(1)
            if key == 27: # esc 
                
                if(config.record):
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec de vídeo
                    output_file = config.record_file_name + ".mkv"  # Nome do arquivo de saída
                    fps = 10.0  # Quadros por segundo
                    frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)  # Cria o objeto de gravação de vídeo
                    for f in frame_stream:
                        out.write(f)

                    out.release()
                break

    def live_detection(self, detect_model_config: List[DetectModelConfig], config: DetectConfig):
        


        weight_paths = []
        labels = []
        confidence = []
        segmentation = []

        print("detection for classe: ")
        for model_cfg in detect_model_config:

            weight_paths.append(model_cfg.weights_path)
            labels.append(model_cfg.label)
            confidence.append(model_cfg.confidence)
            segmentation.append(model_cfg.segmentation)

            print(model_cfg.label)

        models_trained = self._set_trained_models(weight_paths)
        cam = self._select_cam_source_(config.source, config.ip, config.file)

        self._live_detection_loop(cam, confidence, labels, segmentation, models_trained, config)
        


        cam.release()

        cv2.destroyAllWindows()

    
    