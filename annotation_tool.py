import cv2
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import torch
import os
import math
import re
from model_types import ModelType, Model
from matplotlib.path import Path
from configs.annotate_model_config import AnnotateModelConfig

@dataclass
class BoundingBox:

    label: str
    box: Tuple[int, int, int, int]
    confidence: float

@dataclass
class Point:
    x: int
    y: int

@dataclass
class PolygonalMask:
    label: str
    points: List[Point]

@dataclass
class AnnotationCell:

    id: str
    img: np.ndarray
    original_img: np.ndarray
    classes_boxes: List[BoundingBox]
    classes_masks: List[PolygonalMask]
    excluded_classes_boxes: List[BoundingBox]
    excluded_classes_masks: List[PolygonalMask]
    valid: bool

class AnnotationTool:

    def __init__(self):
        # logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
        self.create_rectangle = False
        self.create_poly = False
        self.drawing_poly = False
        self.drawing_rectangle = False
        self.show_ui = True
        self.x_y_mouse = 0,0
        self.poly: List[Point] = []
        self.excluded_color = (150, 150, 150)

    def resize_and_show(self, img):
        screen_res = 1080, 720
        scale_width = screen_res[0] / img.shape[1]
        scale_height = screen_res[1] / img.shape[0]
        scale = min(scale_width, scale_height, 1.0) 
        self.resize_scale = scale

        if scale < 1.0:
            display_img = cv2.resize(img, None,
                                    fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA)
        else:
            display_img = img
        cv2.imshow('Annotation', display_img)

    def _mouse_click_annotate_callback(self, event, x, y, flags, param):
        x = int(x/self.resize_scale)
        y = int(y/self.resize_scale)
        self.x_y_mouse = x, y

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_rectangle:
                img_copy = self.current_annotation.img.copy()
                cv2.rectangle(img_copy, self.rectangle_start_point, (x, y), (0, 255, 0), 2)
                self.resize_and_show(img_copy)
            

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing_rectangle:
                self.drawing_rectangle = False
                self.create_rectangle = False
                self.end_point = (x, y)

                x1, y1 = self.rectangle_start_point
                x2, y2 = self.end_point

                h, w = self.current_annotation.img.shape[:2]
                h_original, w_original = self.current_annotation.original_img.shape[:2]
                x1 = (x1/w)*w_original
                x2 = (x2/w)*w_original
                y1 = (y1/h)*h_original
                y2 = (y2/h)*h_original
                bb = BoundingBox(
                    label=self.current_label,
                    box=torch.tensor([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=torch.float32),
                    confidence=1.0
                )
                self.annotation[self.file_index].classes_boxes.append([bb])

                self.render_annotation()

        if event == cv2.EVENT_RBUTTONUP:
            if self.drawing_poly:
                self.drawing_poly = False
                self.create_poly = False
                self.poly = []
                self.render_annotation()



        if event == cv2.EVENT_LBUTTONDOWN:
            if self.show_ui:
                for i, label in enumerate(self.labels):
                    if 10 <= x <= 200 and 10 + i*40 <= y <= 40 + i*40:
                        self.current_label = label
                        self.render_annotation()
            if self.create_rectangle:
                self.rectangle_start_point = (x, y)
                self.drawing_rectangle = True

            if self.create_poly:
                self.poly.append(Point(x, y))
                self.drawing_poly = True

            else:
                # exclude clicked box from annotations
                for classes_boxes in self.annotation[self.file_index].classes_boxes:
                    for i, bounding_boxes in enumerate(classes_boxes): 
                        x1, y1, x2, y2 = bounding_boxes.box
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            
                            excluded_box = False
                            for bb in self.annotation[self.file_index].excluded_classes_boxes:
                                excluded_box =  torch.allclose(bb.box.to(torch.float32).cpu(), bounding_boxes.box.to(torch.float32).cpu(), atol=1e-3)
                                if excluded_box:
                                    break
                                
                            # error when users excludes all annotations
                            if excluded_box:
                                self.annotation[self.file_index].excluded_classes_boxes.remove(bounding_boxes)
                            else:
                                self.annotation[self.file_index].excluded_classes_boxes.append(bounding_boxes)
                            self.render_annotation()
                                # cv2.rectangle(self.annotation[self.file_index].img, (int(x1), int(y1)), (int(x2), int(y2)), self.excluded_color, 3)
                                # cv2.imshow('Annotation', self.annotation[self.file_index].img)
                
                #exclude polygon from masks
                for classes_masks in self.annotation[self.file_index].classes_masks:
                    for i, masks in enumerate(classes_masks):
                        masks: PolygonalMask
                        points = [(p.x, p.y) for p in masks.points]

                        if len(points) < 3:
                            continue

                        mask_path = Path(points)
                        click_point = (x, y)

                        if mask_path.contains_point(click_point):
                            
                            clicked_array = np.array(points, dtype=np.float32)
                            
                            excluded_mask = False
                            # if already excluded, remove from blacklist
                            for excluded_mask in self.annotation[self.file_index].excluded_classes_masks:
                                excluded_points_list = [(p.x, p.y) for p in excluded_mask.points]
                                excluded_mask_array = np.array(excluded_points_list, dtype=np.float32)

                                if clicked_array.shape == excluded_mask_array.shape:
                                    excluded_mask = True
                                
                                if excluded_mask:
                                    break

                            if excluded_mask:
                                self.annotation[self.file_index].excluded_classes_masks.remove(masks)
                            else:
                                self.annotation[self.file_index].excluded_classes_masks.append(masks)
                            self.render_annotation()

    def draw_guide_lines(self, img, x, y):
        h, w = img.shape[:2]
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)   # vertical
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)   # horizontal

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

    def render_poly(self, poly, img, color = (128, 255, 0)):

        if len(poly) > 2:
            point_list = [(p.x, p.y) for p in poly]
            point_list = np.array(point_list, np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [point_list], (color[0], color[1], color[2], 0.5))
            cv2.polylines(overlay, [point_list], True, (0, 0, 0), 2)
            img_overlay = cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0)
            img = img_overlay
        for p in poly:
            cv2.circle(img, (p.x, p.y), 10, color, -1)
        return img

    def handle_key(self):

        key = cv2.waitKey(10) & 0xFF  
        
        # right
        if key == 83 or key == ord('g') or key == ord('G'):
            if self.file_index +1 < len(self.folder_list):
                self.file_index = self.file_index + 1

        # left
        elif key == 81 or key == ord('d') or key == ord('D'):
            self.file_index = self.file_index - 1

            if self.file_index < 0:
                self.file_index = 0

        elif key == ord('t') or key == ord('T'):
            self.annotation[self.file_index].classes_boxes = [[]]
            self.annotation[self.file_index].classes_masks = [[]]
            self.poly = []
            self.create_poly = False
            self.annotation[self.file_index].excluded_classes_boxes = []
            self.annotation[self.file_index].excluded_classes_masks = []
        
        elif key == ord('e') or key == ord('E'):
            self.show_ui = not self.show_ui
                # self.file_index = len(self.folder_list)-1 future maybe
        
        elif key == ord('s') or key == ord('S'):
            print("save")
            self.save_annotations()
            # self.file_index = self.file_index + 1
        
        elif key == ord('v') or key == ord('V'):
            print("validate")
            if self.current_annotation.valid:
                self.current_annotation.valid = False
            else:
                self.current_annotation.valid = True
            self.render_annotation()
            # self.file_index = self.file_index + 1
        elif key == ord('r') or key == ord('R'):
            self.create_rectangle = True

        elif key == ord('w') or key == ord('W'):
            if len(self.poly) > 1:
                self.poly.pop()
            else:
                self.create_poly = False
                self.drawing_poly = False
                self.poly = []

        elif key == ord('f') or key == ord('F'):
            if self.create_poly:
                self.create_poly = False
                self.drawing_poly = False
                mask = PolygonalMask(
                    label=self.current_label,
                    points=self.poly
                )
                self.annotation[self.file_index].classes_masks.append([mask])
                self.poly = []
            else:
                self.create_poly = True
        
        elif key == ord('q') or key == ord('Q'):
            print("quit")
            self.has_files = False

    def on_label_change(self, event):
        print("Label selecionada:", self.current_label.get())

    def draw_label_buttons(self, img):
        for i, label in enumerate(self.labels):
            cv2.rectangle(img, (10, 10 + i*40), (200, 40 + i*40), (50,50,50), -1)
            color = (0,255,0) if label == self.current_label else (255,255,255)
            cv2.putText(img, label, (20, 35 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def create_bounding_box_to_annotate(self, detection_result, frame, labels_to_annotate = None):
        boxes_detected = []
        for r in detection_result:
            for annotate_index, annotate_label in enumerate(labels_to_annotate):
                boxes = r.boxes
                class_indices = boxes.cls
                boxes_img = []

                for i, class_index in enumerate(class_indices):
                    class_id = int(class_index)


                    if annotate_index == class_id:
                        label = annotate_label
                        box = boxes.xyxy[i]
                        boxes_img.append(box)
                        
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # put box in cam
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # confidence
                        confidence = math.ceil((boxes.conf[i]*100))/100

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(frame, label + " " + str(confidence), org, font, fontScale, color, thickness)
                        boxes_detected.append(box)
        return boxes_detected
    
    def bounding_box_to_image_box(self, img, bounding_boxes: List[BoundingBox], color = (255, 0, 255)):
        if bounding_boxes == [[]]: return
        
        for bb in bounding_boxes:
            x1, y1, x2, y2 = bb.box
            
            # put box in cam
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)


            # object details
            org = [int(x1), int(y1)]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            text_color = (255, 0, 0)


            cv2.putText(img, bb.label + " " + str(bb.confidence), org, font, fontScale, text_color, thickness)


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


    def annotate(self, img_path: str, annotate_model_config: List[AnnotateModelConfig], models_trained: List[Model]):

        # create save dir
        self.labels_path = img_path + "/labels"
        os.makedirs(self.labels_path, exist_ok=True)
        self.load_annotation = True
        self.autosave = True

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


    def create_bounding_box(self, detection_result, frame, label):
        for r in detection_result:

            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, label + " " + str(confidence), org, font, fontScale, color, thickness)

    def get_masks_from_result(self, result, frame_ref):
        masks = []
        for r in result:
            if r.masks is not None and len(r.masks.data) > 0:
                
                mask = r.masks.data[0].cpu().numpy()
                if mask is not None and mask.size > 0:
                    mask_resized = cv2.resize(mask, (frame_ref.shape[1], frame_ref.shape[0]))
                    masks.append(mask_resized)

        return masks
    

    def create_masks_in_frames(self, result, frame, label):
        
        for r in result:
            if r.masks is not None and len(r.masks.data) > 0:
                
                mask = r.masks.data[0].cpu().numpy()
                if mask is not None and mask.size > 0:
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                    mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    mask_colored = np.zeros_like(frame, dtype=np.uint8)
                    mask_colored[:, :, 1] = mask_resized  
                    alpha = 0.5  
                    frame = cv2.addWeighted(frame, 1, mask_colored, alpha, 0)

                    # Encontrar o contorno da máscara
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        if cv2.contourArea(contour) > 100:  
                            x, y, w, h = cv2.boundingRect(contour)
                            x1, y1 = x, y
                            x2, y2 = x + w, y + h
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2
                            confidence_mask = r.boxes.conf[0].item() if r.boxes is not None and len(r.boxes) > 0 else 0.0
                            confidence_mask =  math.ceil((confidence_mask*100))/100
                            cv2.putText(frame, label + " " + str(confidence_mask), org, font, fontScale, color, thickness)

        return frame