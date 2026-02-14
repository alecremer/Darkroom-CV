import cv2
from typing import List
from annotation_transition.annotation_cell import AnnotationCell
from annotation_transition.renderer.render_data import RenderData
from rendering.opencv_renderer_primitives import OpencvRenderPrimitives
from entities.entities import BoundingBox, Rectangle
import math
import numpy as np

class AnnotationOverlay:

    def __init__(self):
        self.excluded_color = (150, 150, 150)
        self.construct_poly_color = (128, 128, 255)

    def draw_construct_box(self, img, rect: Rectangle):
        color = (0, 255, 0)
        thickness = 2
        OpencvRenderPrimitives.draw_rectangle(img, rect, color, thickness)

    def render_annotation(self, img, data: RenderData):
        current_annotation = data.current_annotation
        # self.current_annotation.img = self.current_annotation.original_img.copy()

        # for list of boxes, render boxes
        for bb in current_annotation.classes_boxes:
            self.bounding_box_to_image_box(img, bb)
        
        # render excluded boxes
        self.bounding_box_to_image_box(img, current_annotation.excluded_classes_boxes, self.excluded_color)

        # img_copy = current_annotation.img

        # render construct polygon
        if len(data.construct_poly)>0:
            img = OpencvRenderPrimitives.render_poly(data.construct_poly, img, self.construct_poly_color)
            # img_copy = self.render_poly(self.poly, img_copy, (128, 128, 255))
        
        # classes_masks = data.annotations[data.annotation_index].classes_masks
        classes_masks = data.current_annotation.classes_masks
        for masks in classes_masks:
            if len(masks) > 0:
                for mask in masks:
                    if mask:
                        poly = mask.points
                        img = OpencvRenderPrimitives.render_poly(poly, img)
        
        # excluded_masks = data.annotations[data.annotation_index].excluded_classes_masks
        excluded_masks = data.current_annotation.excluded_classes_masks
        for mask in excluded_masks:
            if mask:
                poly = mask.points
                img = OpencvRenderPrimitives.render_poly(poly, img, self.excluded_color)
            # img_copy = self.current_annotation.img.copy()
        
        self.draw_guide_lines(img, data.mouse_xy.x, data.mouse_xy.y)
        OpencvRenderPrimitives.resize_and_show(img)

    def draw_guide_lines(self, img, x, y):
        h, w = img.shape[:2]
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)   # vertical
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)   # horizontal

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

    def render_bounding_box(self, detection_result, frame, label):
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