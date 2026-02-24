import cv2
from dataclasses import dataclass
from typing import Any, Tuple, List
import os

from matplotlib import pyplot as plt
from annotation_app.action_handler import ActionHandler
from annotation_app.annotation_cell import AnnotationCell
from annotation_app.annotation_data import AnnotationData
from annotation_app.annotation_engine import AnnotationEngine
from annotation_app.annotation_repository import AnnotationRepository
from annotation_app.dataset_navigator import DatasetNavigator
from annotation_app.engine_action import AnnotationEngineAction
from configs.detect_model_config import DetectModelConfig
from dataset_analysis.ceav import CEAV
from dataset_analysis.ceav_exploration import CEAV_Exploration
from entities.model_types import ModelType, Model
from configs.annotate_model_config import AnnotateModelConfig
from entities.entities import BoundingBox, PolygonalMask, Point
from inference_runners.inference_result import InferenceResult
from inference_runners.inference_result_mapper import InferenceResultMapper
from inference_runners.inference_runner import InferenceRunner
from models_loader import ModelsLoader
class AnnotationPipeline:

    def __init__(self, img_path: str, detect_model_config: List[DetectModelConfig]):

        self.poly: List[Point] = []
        self.engine = AnnotationEngine()


        print("Start annotation")
        labels_to_annotate = []
        annotate_confidence = []
        self.labels : List[str] = []
        self.data = AnnotationData()
        self.img_path = img_path

        self.inference_runner = InferenceRunner(detect_model_config)

        for annotate_cfg in detect_model_config:

            # weight_paths.append(annotate_cfg.weights_path)
            for label in annotate_cfg.labels:
                labels_to_annotate.append(label)
            annotate_confidence.append(annotate_cfg.confidence)

        print("annotate for classes: ")
        print(labels_to_annotate)
        self.labels = labels_to_annotate    
        self.data.label = labels_to_annotate[0]
        self.data.labels = labels_to_annotate

        self.repo = AnnotationRepository(labels_to_annotate, img_path)

        self.repo.create_work_dir()
        self.data.num_imgs_annotated = self.repo.get_num_annotations()
        self.folder_list = self.repo.filter_workdir()
        self.data.num_imgs_total = len(self.folder_list)
        self.navigator = DatasetNavigator(self.folder_list)

        self.action_handler = ActionHandler(self.engine, self.navigator, self.repo)

        # dataset metric experiment (bypassed)
        # CEAV_Exploration.explore(self.img_path, self.data.num_imgs_total, self.folder_list, self.repo)

    def run(self, action: AnnotationEngineAction, payload: Any = None):

        self.action_handler.handle_navigation(action, self.data)

        # if annotation not exist
        if len(self.data.annotations) < self.data.file_index + 1:
            img, id = self.repo.load_img(self.img_path, self.folder_list, self.data.file_index)
            img_boxes = []
            classes_masks = []

            img_boxes, classes_masks = self.repo.load_annotation(img, id)

            # inference_in_all_frames = False
            inference_in_all_frames = True
            merge_masks = False
            
            if (len(img_boxes) == 0 and len(classes_masks) == 0) or inference_in_all_frames:
                result: List[InferenceResult] = self.inference_runner.inference(img)
                masks = []
                
                for r in result:
                    masks.append(r.results)
                    if merge_masks:
                        classes_masks.append(r.results)
                
                if not merge_masks:
                    classes_masks = masks

            annotation = AnnotationCell(id, img, img, img_boxes, classes_masks, [], [], True)
            self.data.annotations.append(annotation)
            self.data.current_annotation = annotation
            if self.data.file_index + 1 > self.data.num_imgs_annotated:
                self.data.num_imgs_annotated = self.data.file_index + 1


        else:
            self.data.current_annotation = self.data.annotations[self.data.file_index]

        self.action_handler.handle(action, self.data, payload)

    
    def label_to_index(self, labels_to_annotate):
        for label_list in labels_to_annotate:
            for label in label_list:
                if label not in self.labels:
                    self.labels.append(label)
  
