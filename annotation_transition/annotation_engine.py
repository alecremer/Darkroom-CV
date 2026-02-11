from annotation_transition.annotation_renderer.annotation_action import AnnotationAction
from annotation_transition.opencv.annotation_view import AnnotationView
from types.entities import BoundingBox, PolygonalMask, Rectangle, Point
from annotation_cell import AnnotationCell
from typing import List, Any
import torch
from matplotlib.path import Path
import numpy as np

class AnnotationEngine:
    
    def __init__(self):
        self.current_annotation = AnnotationCell()
        self.current_label: str = ""
        self.file_index: int = 0
        self.annotation: List[AnnotationCell] = []

    def resize_rectangle_by_original_img(self, img, original_img, rectangle: Rectangle) -> Rectangle:
        
        h, w = img.shape[:2]
        h_original, w_original = original_img.shape[:2]

        x1 = (rectangle.p0.x/w)*w_original
        x2 = (rectangle.p.x/w)*w_original
        y1 = (rectangle.p0.y/h)*h_original
        y2 = (rectangle.p.y/h)*h_original

        return Rectangle(Point(x1, y1), Point(x2, y2))

    def annotate_bbox(self, rect: Rectangle, label: str):

        x1, y1, x2, y2 = rect
        bb = BoundingBox(
            label=label,
            box=torch.tensor([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=torch.float32),
            confidence=1.0
        )
        self.annotation[self.file_index].classes_boxes.append([bb])

    def annotate_bbox_from_construct_rect(self):
        img = self.current_annotation.img
        original_img = self.current_annotation.original_img
        rectangle = self.construct_rectangle
        rectangle_normalized = self.resize_rectangle_by_original_img(img, original_img, rectangle)

        self.annotate_bbox(rectangle_normalized, self.current_label)        

    

    def select_label(self, p: Point):
        label = self.view.select_label(p)
        if label: self.current_label = label


    def exclude_box_from_annotation(self, p: Point):
        x, y = p
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

    def exclude_polygon_from_annotations(self, p: Point):
        x, y = p
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

    

    def execute(self, action: AnnotationAction, payload: Any):
        
        if action is AnnotationAction.ANNOTATE_BBOX:
            self.construct_rectangle.p = payload
            self.annotate_bbox_from_construct_rect()

        elif action is AnnotationAction.CANCEL_CONSTRUCT_POLY:
            self.cancel_construct_poly()

        elif action is AnnotationAction.SELECT_LABEL:
            self.select_label(payload)

        elif action is AnnotationAction.START_CONSTRUCT_RECTANGLE:
            self.start_construct_rectangle(payload)

        elif action is AnnotationAction.EXCLUDE_CLICKED_ENTITY:
            self.exclude_box_from_annotation(payload)
            self.exclude_polygon_from_annotations(payload)