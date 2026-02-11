import cv2
from dataclasses import dataclass
from typing import Tuple, List
import os
from types.model_types import ModelType, Model
from configs.annotate_model_config import AnnotateModelConfig
from types.entities import BoundingBox, PolygonalMask, Point
from annotation_transition.keyboard_handler import KeyboardHandler
from annotation_transition.opencv.annotation_view import AnnotationView
from annotation_transition.interaction_policy import InteractionPolicy, PolicyResult
from draw_state import DrawState
from annotation_engine import AnnotationEngine

class AnnotationPipeline:

    def __init__(self):

        self.draw_state: DrawState
        self.show_ui = True
        self.x_y_mouse = 0,0
        self.poly: List[Point] = []
        self.excluded_color = (150, 150, 150) #TODO move to config file
        self.keyboard_handler = KeyboardHandler(self)
        self.keyboard_handler.build()
        self.annotation_view = AnnotationView()
        self.rectangle_start_point: Point
        self.rectangle_end_point: Point
        self.policy = InteractionPolicy()
        self.engine = AnnotationEngine()

    def reset_annotation_cell(self):
        self.annotation[self.file_index].classes_boxes = [[]]
        self.annotation[self.file_index].classes_masks = [[]]
        self.poly = []
        self.draw_state = DrawState.IDLE
        self.annotation[self.file_index].excluded_classes_boxes = []
        self.annotation[self.file_index].excluded_classes_masks = []
    
    def toggle_show_ui(self):
        self.annotation_view.toggle_show_ui()

    def set_create_rectangle(self):
        self.draw_state = DrawState.STARTING_RECTANGLE

    def undo_polygon_point(self):
        if len(self.poly) > 1:
            self.poly.pop()
        else:
            self.draw_state = DrawState.IDLE
            self.poly = []

    def done_or_create_polygon(self):
        if self.create_poly:
            self.draw_state = DrawState.IDLE
            mask = PolygonalMask(
                label=self.current_label,
                points=self.poly
            )
            self.annotation[self.file_index].classes_masks.append([mask])
            self.poly = []
        else:
            self.create_poly = True

    def quit(self):
        self.has_files = False

    def handle_key(self):

        key = cv2.waitKey(10) & 0xFF  
        self.keyboard_handler.routine(key)
    
    def label_to_index(self, labels_to_annotate):
        for label_list in labels_to_annotate:
            for label in label_list:
                if label not in self.labels:
                    self.labels.append(label)
    
    def log_cmds(self):
        print("----------------------------------------")
        print("g or right: next")
        print("d or left: previous")
        print("t: reset current image")
        print("e: show/hide UI")
        print("s: save")
        print("r: create rectangle")
        print("f: create/save polygon")
        print("w: delete last polygon point")
        print("q: quit")

    def annotate(self, img_path: str, annotate_model_config: List[AnnotateModelConfig], models_trained: List[Model]):

        self.create_work_dir(img_path)        

        print("Start annotation")
        weight_paths = []
        labels_to_annotate = []
        annotate_confidence = []
        segmentation = []
        self.labels : List[str] = []

        print("annotate for classes: ")
        for annotate_cfg in annotate_model_config:

            # weight_paths.append(annotate_cfg.weights_path)
            labels_to_annotate.append(annotate_cfg.labels_to_annotate)
            annotate_confidence.append(annotate_cfg.annotate_confidence)

            print(annotate_cfg.labels_to_annotate)

        print("----------------------------------------")
        
        self.label_to_index(labels_to_annotate)

        folder_list = self.filter_workdir(img_path)

        self.folder_list = sorted(folder_list, key=self.natural_sort)
        print(f"{len(self.folder_list)} files to annotate")

        self.log_cmds()
        
        self.has_files = len(self.folder_list) > 0
        self.file_index = 0

        self.current_label = self.labels[0]

        #TODO: move from here
        # set cv2
        window_name = "Annotation"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_click_annotate_callback)
        
        self.annotation: List[AnnotationCell] = []

        if self.load_annotation:
            self.load_annotation()
            
        #TODO: Separate init flow and annotation loop
        # annotation loop
        while self.has_files:
            
            self.current_annotation : AnnotationCell = None
            
            # if img not exists
            if len(self.annotation) < self.file_index +1:
                
                file = self.folder_list[self.file_index]
                id = file.split(".")[0]
                img_original = cv2.imread(os.path.join(img_path, file))
                img = img_original.copy()

                img_boxes = []
                classes_masks = [[]]

                
                # img = img_original.copy()

                # load annotations
                
                if self.repo.check_if_annotated(id):
                    img_boxes, classes_masks = self.repo.load_annotations(img)
                # else:
                # IA assistance
                #TODO: Try to use existent inference structures here

            else:
                self.current_annotation = self.annotation[self.file_index]

            if self.current_annotation is not None:
                
                if not self.has_files:
                    print("empty folder")
                    break
                
                self.render_annotation()

                self.handle_key()
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("killing...")
                break
        cv2.destroyAllWindows()
        exit(0)