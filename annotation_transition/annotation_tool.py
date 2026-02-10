import cv2
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import torch
import os
import math
import re
from types.model_types import ModelType, Model
from matplotlib.path import Path
from configs.annotate_model_config import AnnotateModelConfig
from types.entities import BoundingBox, PolygonalMask, Point
from enum import Enum
from annotation_transition.keyboard_handler import KeyboardHandler
from annotation_transition.annotation_view import AnnotationView
from command_policy import CommandPolicy, PolicyResult
from draw_state import DrawState
from command_intent import CommandIntent
from annotation_engine import AnnotationEngine
from annotation_action import AnnotationAction

class AnnotationTool:

    def __init__(self):
        # logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
        
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
        self.policy = CommandPolicy()
        self.engine = AnnotationEngine()

    def render_annotation(self):
        self.current_annotation.img = self.current_annotation.original_img.copy()
        for bb in self.current_annotation.classes_boxes:
            self.bounding_box_to_image_box(self.current_annotation.img, bb)
        
        self.bounding_box_to_image_box(self.current_annotation.img, self.current_annotation.excluded_classes_boxes, self.excluded_color)
        self.mark_validation_img(self.current_annotation.img, self.current_annotation.valid)

        if self.show_ui:
            self.draw_label_buttons(self.current_annotation.img)

            # put name of image
            image_name = self.current_annotation.id
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(image_name, font, font_scale, thickness)
            x = self.current_annotation.img.shape[1] - text_width - 10  # 10px de margem
            y = self.current_annotation.img.shape[0] - 10  # 10px acima da borda inferior
            cv2.putText(self.current_annotation.img, image_name, (x, y), font, font_scale, color, thickness)
        
        img_copy = self.current_annotation.img

        # render annotation active polygon
        if len(self.poly)>0:
            img_copy = self.render_poly(self.poly, img_copy, (128, 128, 255))
        
        classes_masks = self.annotation[self.file_index].classes_masks
        for masks in classes_masks:
            if len(masks) > 0:
                for mask in masks:
                    if mask:
                        poly = mask.points
                        img_copy = self.render_poly(poly, img_copy)
        
        excluded_masks = self.annotation[self.file_index].excluded_classes_masks
        for mask in excluded_masks:
            if mask:
                poly = mask.points
                img_copy = self.render_poly(poly, img_copy, self.excluded_color)
            # img_copy = self.current_annotation.img.copy()
        
        self.draw_guide_lines(img_copy, self.x_y_mouse[0], self.x_y_mouse[1])
        self.resize_and_show(img_copy)

    def next_img(self):
        if self.file_index +1 < len(self.folder_list):
            self.file_index = self.file_index + 1

    def previous_img(self):
        self.file_index = self.file_index - 1

        if self.file_index < 0:
            self.file_index = 0

    def reset_annotation_cell(self):
        self.annotation[self.file_index].classes_boxes = [[]]
        self.annotation[self.file_index].classes_masks = [[]]
        self.poly = []
        self.draw_state = DrawState.IDLE
        self.annotation[self.file_index].excluded_classes_boxes = []
        self.annotation[self.file_index].excluded_classes_masks = []
    
    def toggle_show_ui(self):
        self.show_ui = not self.show_ui
            # self.file_index = len(self.folder_list)-1 future maybe

    def set_create_rectangle(self):
        self.draw_state = DrawState.STARTING_RECTANGLE

    def undo_polygon_point(self):
        if len(self.poly) > 1:
            self.poly.pop()
        else:
            # self.create_poly = False
            # self.drawing_poly = False
            self.draw_state = DrawState.IDLE
            self.poly = []

    def done_or_create_polygon(self):
        if self.create_poly:
            self.draw_state = DrawState.IDLE
            # self.create_poly = False
            # self.drawing_poly = False
            mask = PolygonalMask(
                label=self.current_label,
                points=self.poly
            )
            self.annotation[self.file_index].classes_masks.append([mask])
            self.poly = []
        else:
            self.create_poly = True

    def quit(self):
        print("quit")
        self.has_files = False

    def handle_key(self):

        key = cv2.waitKey(10) & 0xFF  
        self.keyboard_handler.routine(key)
        
    def on_label_change(self, event):
        print("Label selecionada:", self.current_label.get())
    
    def save_annotations(self):
        
        # print("id: ", self.annotation[0].id)
        # print("classes boxes: ", self.annotation[0].classes_boxes)

        # create labels file
        with open(self.labels_path + "/classes.txt", "w") as f:
            for label in self.labels:
                f.write(f"{label}\n")

        files: List[str] = []
        for annotation in self.annotation:
            if annotation.valid:
                with open(self.labels_path + "/" + annotation.id + ".txt", "w") as f:
                    
                    h, w = annotation.original_img.shape[:2]

                    # save boxes
                    for class_box in annotation.classes_boxes:
                        for box in class_box:
                            excluded_box = False
                            for bb in annotation.excluded_classes_boxes:
                                excluded_box =  torch.allclose(bb.box.to(torch.float32).cpu(), box.box.to(torch.float32).cpu(), atol=1e-3)
                                if excluded_box:
                                    break
                    
                            if not excluded_box:
                                x1, y1, x2, y2 = box.box
                                
                                x1_norm = min(x1, x2)/w
                                x2_norm = max(x1, x2)/w
                                y1_norm = min(y1, y2)/h
                                y2_norm = max(y1, y2)/h

                                box_w_norm = x2_norm - x1_norm
                                box_h_norm = y2_norm - y1_norm

                                x_center = x1_norm + box_w_norm/2
                                y_center = y1_norm + box_h_norm/2
                                
                                label_num = self.labels.index(box.label)
                                txt_line = f"{label_num} {x_center:.6f} {y_center:.6f} {box_w_norm:.6f} {box_h_norm:.6f}\n"
                                f.write(txt_line)
                                print(txt_line)

                    # save masks
                    for class_mask in annotation.classes_masks:
                        for mask in class_mask:
                            for excluded_mask in self.annotation[self.file_index].excluded_classes_masks:
                                excluded_points_list = [(p.x, p.y) for p in excluded_mask.points]
                                excluded_mask_array = np.array(excluded_points_list, dtype=np.float32)

                                mask_points_list = [(p.x, p.y) for p in mask.points]
                                mask_array = np.array(mask_points_list, dtype=np.float32)

                                if mask_array.shape == excluded_mask_array.shape:
                                    break

                            label_num = self.labels.index(mask.label)
                            points_str_proto = (f"{p.x/w} {p.y/h}" for p in mask.points)
                            points_str = " ".join(points_str_proto)
                            txt_line = f"{label_num} {points_str}\n"
                            f.write(txt_line)
                            print(txt_line)
    
    def natural_sort(self, l):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', l)]

    def create_work_dir(self, img_path):
        self.labels_path = img_path + "/labels"
        os.makedirs(self.labels_path, exist_ok=True)
        self.load_annotation = True
        self.autosave = True

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
        # labels to index
        for label_list in labels_to_annotate:
            for label in label_list:
                if label not in self.labels:
                    self.labels.append(label)


        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        folder_list = [
            f for f in os.listdir(img_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        self.folder_list = sorted(folder_list, key=self.natural_sort)
        print(f"{len(self.folder_list)} files to annotate")

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


        
        self.has_files = len(self.folder_list) > 0
        self.file_index = 0

        

        self.current_label = self.labels[0]
        window_name = "Annotation"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_click_annotate_callback)
        
        self.annotation: List[AnnotationCell] = []

        if self.load_annotation:
            label_extensions = {".txt"}
            labels_list = [
                f for f in os.listdir(self.labels_path)
                if os.path.splitext(f)[1].lower() in label_extensions
            ]
            self.label_list_sorted = sorted(labels_list, key=self.natural_sort)
            
        
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
                if id in [l.split(".")[0] for l in self.label_list_sorted]:

                    with open(os.path.join(self.labels_path, id + ".txt"), "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            values = line.split(" ")
                            cls = int(values[0])

                            h, w = img.shape[:2]
                            if len(values) == 5: # rectangle


                                xc = float(values[1])
                                yc = float(values[2])
                                wb = float(values[3])
                                hb = float(values[4])

                                x1 = xc-wb/2
                                x2 = xc+wb/2
                                y1 = yc+hb/2
                                y2 = yc-hb/2
                                
                                x1_scaled = x1*w
                                x2_scaled = x2*w
                                y1_scaled = y1*h
                                y2_scaled = y2*h

                                bb = BoundingBox(
                                    label=self.labels[cls],
                                    box = torch.tensor([min(x1_scaled, x2_scaled), min(y1_scaled, y2_scaled), max(x1_scaled, x2_scaled), max(y1_scaled, y2_scaled)], dtype=torch.float32),
                                    confidence=1.0
                                )
                                img_boxes.append([bb])
                            elif len(values) > 5: # mask
                                poly = [Point(int(float(x)*w), int(float(y)*h)) for x, y in zip(values[1::2], values[2::2])]
                                mask = PolygonalMask(
                                    label=self.labels[cls],
                                    points=poly
                                )
                                classes_masks.append([mask])

                # IA assistance
                else:
                    for i, m in enumerate(models_trained):
                        # if m and m.strip():
                        if m.model_type == ModelType.YOLO.value:
                            result = m.model(img, conf=annotate_confidence[i])
                            if m.model.task == "detect":
                                bounding_boxes = self.result_to_bounding_box(result, labels_to_annotate[i])
                                img_boxes.append(bounding_boxes)
                            elif m.model.task == "segment":
                                masks = self.get_masks_from_result(result, img)
                                classes_masks.append(masks)
                                # boxes = self.create_bounding_box_to_annotate(result, img, labels_to_annotate[index])
                        elif m.model_type == ModelType.VITMAE_SEG.value:
                            predict_mask = m.model.predict_from_image(img)
                            predict_mask_8bit = (predict_mask * 255).astype(np.uint8)
                            contours_list = cv2.findContours(predict_mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            masks = []
                            for contours_obj in contours_list:
                                contours_obj = [c[0] for c in contours_obj[0]]
                                
                                mask = []
                                for contours in contours_obj:
                                    if isinstance(contours, np.ndarray) and  len(contours) == 2:
                                        # print(contours)
                                        point = Point(contours[0], contours[1])
                                        mask.append(point)
                                    # mask = [Point(p[0], p[1]) for p in contours_obj]
                                    # masks.append(PolygonalMask("none_label", mask))
                                #     # print(contours)
                                #     # contour = contours.squeeze(axis=1)
                                #     print(contours)
                                #     contour = contours[0]
                                    
                                #     mask = [Point(p[0], p[1]) for p in contour]
                                masks.append(PolygonalMask(self.current_label, mask))
                            result = masks
                            # print(result)
                            classes_masks.append(result)
                
                self.annotation.append(AnnotationCell(id, img, img_original, img_boxes, classes_masks, [], [], True))

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
    
    def result_to_bounding_box(self, detection_result, labels_to_annotate = None):
        detected = []
        for r in detection_result:
            for annotate_index, annotate_label in enumerate(labels_to_annotate):
                boxes = r.boxes
                class_indices = boxes.cls

                for i, class_index in enumerate(class_indices):
                    class_id = int(class_index)


                    if annotate_index == class_id:
                        label = annotate_label
                        box = boxes.xyxy[i]
                        
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # confidence
                        confidence = math.ceil((boxes.conf[i]*100))/100
                        bounding_box = BoundingBox(label, box, confidence)
                        detected.append(bounding_box)
        return detected

    
    def mark_validation_img(self, img, valid: bool):
        valid_text = "To Save" if valid else "Discard"
        color_valid = (0, 255, 0)
        color_not_valid = (0, 0, 255)
        color = color_valid if valid else color_not_valid
        org = [10, 900]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        cv2.putText(img, valid_text, org, font, fontScale, color, thickness)

    

    def get_masks_from_result(self, result, frame_ref):
        masks = []
        for r in result:
            if r.masks is not None and len(r.masks.data) > 0:
                
                mask = r.masks.data[0].cpu().numpy()
                if mask is not None and mask.size > 0:
                    mask_resized = cv2.resize(mask, (frame_ref.shape[1], frame_ref.shape[0]))
                    masks.append(mask_resized)

        return masks