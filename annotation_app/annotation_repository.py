from typing import List
import numpy as np
import torch
import os
from annotation_app.annotation_cell import AnnotationCell
from annotation_app.tools import Tools
from entities.entities import BoundingBox, PolygonalMask, Point
import cv2

class AnnotationRepository:

    def __init__(self, labels, img_path):
        self.labels = labels
        self.img_path = img_path
        self.labels_path = img_path + "/labels"

    def create_classes_file(self):
        with open(self.labels_path + "/classes.txt", "w") as f:
            for label in self.labels:
                f.write(f"{label}\n")

    def create_work_dir(self):
        os.makedirs(self.labels_path, exist_ok=True)
        # self.load_annotation = True
        self.autosave = True


    def save_annotations(self, annotations: List[AnnotationCell], labels: List[str], file_index: int):
        
        self.create_classes_file()

        files: List[str] = []
        for annotation in annotations:
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
                                
                                label_num = labels.index(box.label)
                                txt_line = f"{label_num} {x_center:.6f} {y_center:.6f} {box_w_norm:.6f} {box_h_norm:.6f}\n"
                                f.write(txt_line)

                    # save masks
                    for class_mask in annotation.classes_masks:
                        for mask in class_mask:
                            excluded_mask = False
                            for excluded_mask in annotation.excluded_classes_masks:
                                excluded_points_list = [(p.x, p.y) for p in excluded_mask.points]
                                excluded_mask_array = np.array(excluded_points_list, dtype=np.float32)

                                mask_points_list = [(p.x, p.y) for p in mask.points]
                                mask_array = np.array(mask_points_list, dtype=np.float32)

                                if mask_array.shape == excluded_mask_array.shape:
                                    excluded_mask = True
                            
                            if not excluded_mask:
                                label_num = labels.index(mask.label)
                                points_str_proto = (f"{p.x/w} {p.y/h}" for p in mask.points)
                                points_str = " ".join(points_str_proto)
                                txt_line = f"{label_num} {points_str}\n"
                                f.write(txt_line)
            print("Save done!")

    def load_labels(self):
        label_extensions = {".txt"}
        labels_list = [
            f for f in os.listdir(self.labels_path)
            if os.path.splitext(f)[1].lower() in label_extensions
        ]
        self.label_list_sorted = sorted(labels_list, key=Tools.natural_sort)


    def filter_workdir(self):
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        folder_list = [
            f for f in os.listdir(self.img_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        return folder_list
    
    def check_if_annotated(self, id):
        return id in [l.split(".")[0] for l in self.label_list_sorted]

    def get_num_annotations(self):
        num = len(os.listdir(self.labels_path))
        return num if num > 0 else 0

    def load_annotation(self, img, id):
        img_boxes = []
        classes_masks = []

        path = os.path.join(self.labels_path, id + ".txt")

        if os.path.isfile(path):
            with open(path, "r") as f:
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
                            points=poly,
                            confidence=1.0
                        )
                        classes_masks.append([mask])
        return img_boxes, classes_masks
    
    def load_img(self, img_path, folder_list, file_index):
        file = folder_list[file_index]
        id = file.split(".")[0]
        img_original = cv2.imread(os.path.join(img_path, file))
        img = img_original.copy()

        return img, id