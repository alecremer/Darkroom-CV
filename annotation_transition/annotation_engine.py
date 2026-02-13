from annotation_transition.annotation_cell import AnnotationCell
from entities.entities import BoundingBox, PolygonalMask, Rectangle, Point
from typing import List, Any
import torch
from matplotlib.path import Path
import numpy as np

class AnnotationEngine:
    
    def resize_rectangle_by_original_img(self, img, original_img, rectangle: Rectangle) -> Rectangle:
        
        h, w = img.shape[:2]
        h_original, w_original = original_img.shape[:2]

        x1 = (rectangle.p0.x/w)*w_original
        x2 = (rectangle.p.x/w)*w_original
        y1 = (rectangle.p0.y/h)*h_original
        y2 = (rectangle.p.y/h)*h_original

        return Rectangle(Point(x1, y1), Point(x2, y2))

    def annotate_bbox(self, rect: Rectangle, label: str, annotation: List[AnnotationCell], file_index: int):

        x1, y1, x2, y2 = rect.to_coords()
        bb = BoundingBox(
            label=label,
            box=torch.tensor([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=torch.float32),
            confidence=1.0
        )
        annotation[file_index].classes_boxes.append([bb])

    def annotate_mask(self, mask: PolygonalMask, annotation: List[AnnotationCell], file_index: int):
        annotation[file_index].classes_masks.append([mask])
        
    # def done_or_create_polygon(self):
    #     mask = PolygonalMask(
    #         label=self.current_label,
    #         points=self.poly
    #     )
    #     self.annotation[self.file_index].classes_masks.append([mask])


    # def annotate_bbox_from_construct_rect(self):
    #     img = self.current_annotation.img
    #     original_img = self.current_annotation.original_img
    #     rectangle = self.construct_rectangle
    #     rectangle_normalized = self.resize_rectangle_by_original_img(img, original_img, rectangle)

    #     self.annotate_bbox(rectangle_normalized, self.current_label)        

    

    def select_label(self, label: str, labels: List[str], current_label: str):
        if label in labels: return label
        return current_label

    def exclude_box_from_annotation(self, p: Point, annotation: List[AnnotationCell], file_index: int):
        x, y = p
        for classes_boxes in annotation[file_index].classes_boxes:
            for i, bounding_boxes in enumerate(classes_boxes): 
                x1, y1, x2, y2 = bounding_boxes.box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    
                    excluded_box = False
                    for bb in annotation[file_index].excluded_classes_boxes:
                        excluded_box =  torch.allclose(bb.box.to(torch.float32).cpu(), bounding_boxes.box.to(torch.float32).cpu(), atol=1e-3)
                        if excluded_box:
                            break
                        
                    # error when users excludes all annotations
                    if excluded_box:
                        annotation[file_index].excluded_classes_boxes.remove(bounding_boxes)
                    else:
                        annotation[file_index].excluded_classes_boxes.append(bounding_boxes)

    def exclude_polygon_from_annotations(self, p: Point, annotation: List[AnnotationCell], file_index: int):
        x, y = p
        for classes_masks in annotation[file_index].classes_masks:
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
                    for excluded_mask in annotation[file_index].excluded_classes_masks:
                        excluded_points_list = [(p.x, p.y) for p in excluded_mask.points]
                        excluded_mask_array = np.array(excluded_points_list, dtype=np.float32)

                        if clicked_array.shape == excluded_mask_array.shape:
                            excluded_mask = True
                        
                        if excluded_mask:
                            break

                    if excluded_mask:
                        annotation[file_index].excluded_classes_masks.remove(masks)
                    else:
                        annotation[file_index].excluded_classes_masks.append(masks)

    def reset_annotation_cell(self, annotation: List[AnnotationCell], file_index: int):
        annotation[self.file_index].classes_boxes = [[]]
        annotation[self.file_index].classes_masks = [[]]
        annotation[self.file_index].excluded_classes_boxes = []
        annotation[self.file_index].excluded_classes_masks = []