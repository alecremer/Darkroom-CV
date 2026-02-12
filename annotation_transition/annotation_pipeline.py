import cv2
from dataclasses import dataclass
from typing import Any, Tuple, List
import os
from annotation_transition.action_handler import ActionHandler
from annotation_transition.annotation_cell import AnnotationCell
from annotation_transition.annotation_data import AnnotationData
from annotation_transition.annotation_engine import AnnotationEngine
from annotation_transition.annotation_repository import AnnotationRepository
from annotation_transition.dataset_navigator import DatasetNavigator
from annotation_transition.engine_action import AnnotationEngineAction
from entities.model_types import ModelType, Model
from configs.annotate_model_config import AnnotateModelConfig
from entities.entities import BoundingBox, PolygonalMask, Point
class AnnotationPipeline:

    def __init__(self, img_path: str, annotate_model_config: List[AnnotateModelConfig], models_trained: List[Model]):

        self.poly: List[Point] = []
        self.engine = AnnotationEngine()


        print("Start annotation")
        weight_paths = []
        labels_to_annotate = []
        annotate_confidence = []
        segmentation = []
        self.labels : List[str] = []
        self.data = AnnotationData()
        self.img_path = img_path

        print("annotate for classes: ")
        for annotate_cfg in annotate_model_config:

            # weight_paths.append(annotate_cfg.weights_path)
            labels_to_annotate.append(annotate_cfg.labels_to_annotate)
            annotate_confidence.append(annotate_cfg.annotate_confidence)

            print(annotate_cfg.labels_to_annotate)
            
        self.label_to_index(labels_to_annotate)

        self.repo = AnnotationRepository(labels_to_annotate, img_path)

        self.repo.create_work_dir()
        self.folder_list = self.repo.filter_workdir()
        self.navigator = DatasetNavigator(self.folder_list)

        self.action_handler = ActionHandler(self.engine, self.navigator)

    def run(self, action: AnnotationEngineAction, payload: Any = None):

        self.action_handler.handle_navigation(action, self.data)

        # if annotation not exist
        if len(self.data.annotations) < self.data.file_index + 1:
            img, id = self.repo.load_img(self.img_path, self.folder_list, self.data.file_index)
            img_boxes = []
            classes_masks = [[]]
            annotation = AnnotationCell(id, img, img, img_boxes, classes_masks, [], [], True)
            self.data.annotations.append(annotation)
            self.data.current_annotation = annotation
        else:
            self.data.current_annotation = self.data.annotations[self.data.file_index]

        self.action_handler.handle(action, self.data, payload)
        return self.data



    # def reset_annotation_cell(self):
    #     self.annotation[self.file_index].classes_boxes = [[]]
    #     self.annotation[self.file_index].classes_masks = [[]]
    #     self.poly = []
    #     self.draw_state = DrawState.IDLE
    #     self.annotation[self.file_index].excluded_classes_boxes = []
    #     self.annotation[self.file_index].excluded_classes_masks = []
    
    # def toggle_show_ui(self):
    #     self.annotation_view.toggle_show_ui()

    # def set_create_rectangle(self):
    #     self.draw_state = DrawState.STARTING_RECTANGLE

    # def undo_polygon_point(self):
    #     if len(self.poly) > 1:
    #         self.poly.pop()
    #     else:
    #         self.draw_state = DrawState.IDLE
    #         self.poly = []

    # def done_or_create_polygon(self):
    #     if self.create_poly:
    #         self.draw_state = DrawState.IDLE
    #         mask = PolygonalMask(
    #             label=self.current_label,
    #             points=self.poly
    #         )
    #         self.annotation[self.file_index].classes_masks.append([mask])
    #         self.poly = []
    #     else:
    #         self.create_poly = True

    # def quit(self):
    #     self.has_files = False

    # def handle_key(self):

    #     key = cv2.waitKey(10) & 0xFF  
    #     self.keyboard_handler.routine(key)
    
    def label_to_index(self, labels_to_annotate):
        for label_list in labels_to_annotate:
            for label in label_list:
                if label not in self.labels:
                    self.labels.append(label)
    
    # def log_cmds(self):
    #     print("----------------------------------------")
    #     print("g or right: next")
    #     print("d or left: previous")
    #     print("t: reset current image")
    #     print("e: show/hide UI")
    #     print("s: save")
    #     print("r: create rectangle")
    #     print("f: create/save polygon")
    #     print("w: delete last polygon point")
    #     print("q: quit")

    # def annotate(self, img_path: str, annotate_model_config: List[AnnotateModelConfig], models_trained: List[Model]):

    #     self.create_work_dir(img_path)        

    #     print("Start annotation")
    #     weight_paths = []
    #     labels_to_annotate = []
    #     annotate_confidence = []
    #     segmentation = []
    #     self.labels : List[str] = []

    #     print("annotate for classes: ")
    #     for annotate_cfg in annotate_model_config:

    #         # weight_paths.append(annotate_cfg.weights_path)
    #         labels_to_annotate.append(annotate_cfg.labels_to_annotate)
    #         annotate_confidence.append(annotate_cfg.annotate_confidence)

    #         print(annotate_cfg.labels_to_annotate)

    #     print("----------------------------------------")
        
    #     self.label_to_index(labels_to_annotate)

    #     folder_list = self.filter_workdir(img_path)

    #     self.folder_list = sorted(folder_list, key=self.natural_sort)
    #     print(f"{len(self.folder_list)} files to annotate")

    #     self.log_cmds()
        
    #     self.has_files = len(self.folder_list) > 0
    #     self.file_index = 0

    #     self.current_label = self.labels[0]

    #     #TODO: move from here
        
    #     self.annotation: List[AnnotationCell] = []

    #     if self.load_annotation:
    #         self.load_annotation()
            
    #     #TODO: Separate init flow and annotation loop
    #     # annotation loop
    #     while self.has_files:
            
    #         self.current_annotation : AnnotationCell = None
            
    #         # if img not exists
    #         if len(self.annotation) < self.file_index +1:
                
    #             file = self.folder_list[self.file_index]
    #             id = file.split(".")[0]
    #             img_original = cv2.imread(os.path.join(img_path, file))
    #             img = img_original.copy()

    #             img_boxes = []
    #             classes_masks = [[]]

                
    #             # img = img_original.copy()

    #             # load annotations
                
    #             if self.repo.check_if_annotated(id):
    #                 img_boxes, classes_masks = self.repo.load_annotations(img)
    #             # else:
    #             # IA assistance
    #             #TODO: Try to use existent inference structures here

    #         else:
    #             self.current_annotation = self.annotation[self.file_index]

    #         if self.current_annotation is not None:
                
    #             if not self.has_files:
    #                 print("empty folder")
    #                 break
                
    #             self.render_annotation()

    #             self.handle_key()
    #         if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
    #             print("killing...")
    #             break
    #     cv2.destroyAllWindows()
    #     exit(0)