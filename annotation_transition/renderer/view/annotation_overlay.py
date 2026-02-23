import cv2
from typing import List
from annotation_transition.annotation_cell import AnnotationCell
from annotation_transition.renderer.render_data import RenderData
from annotation_transition.renderer.view.draw_state_mapper import DrawStateMapper
from rendering.opencv_renderer_primitives import OpencvRenderPrimitives
from entities.entities import BoundingBox, PolygonalMask, Rectangle
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
        return img

    def render_annotation(self, img, data: RenderData):
        current_annotation = data.current_annotation

        # for list of boxes, render boxes
        for bb in current_annotation.classes_boxes:
            self.bounding_box_to_image_box(img, bb)
       
        # img_copy = current_annotation.img

        # render construct polygon
        if len(data.construct_poly)>0:
            img = OpencvRenderPrimitives.render_poly(data.construct_poly, img, self.construct_poly_color)
        
        # render masks
        classes_masks = data.current_annotation.classes_masks
        img = self._render_masks(img, classes_masks)
        
        
        excluded_masks = data.current_annotation.excluded_classes_masks
        for mask in excluded_masks:
            if mask:
                poly = mask.points
                img = OpencvRenderPrimitives.render_poly(poly, img, self.excluded_color)
         
        # render excluded boxes
        self.bounding_box_to_image_box(img, current_annotation.excluded_classes_boxes, self.excluded_color)


        self.draw_guide_lines(img, data.mouse_xy.x, data.mouse_xy.y)
        return img

    def _render_masks(self, img, classes_masks):
        for masks in classes_masks:
            if len(masks) > 0:
                for mask in masks:
                    if mask:
                        poly = mask.points
                        img = OpencvRenderPrimitives.render_poly(poly, img)
                        
                        # get poly x, y min points
                        x = min(p.x for p in poly)
                        y = min(p.y for p in poly)

                        # object details
                        org = [int(x) + 2, int(y) + 15]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.6
                        thickness = 2
                        text_color = (255, 0, 0)
                        mask: PolygonalMask
                        text = mask.label
                        conf = mask.confidence

                        cv2.putText(img, text  + " " + str(conf), org, font, fontScale, text_color, thickness)
        return img
    
    def draw_text_box(self, img, text: str,x_anchor_percent: float, y_anchor_percent: float, 
                      box_x_length: int, box_y_length: int, 
                      font_scale: float = 0.5,
                      thickness: int = 2, x_padding: int = 15, y_padding: int = 5, 
                      text_color = (255, 255, 255), box_color = (50, 50, 50), alpha: float = 0.8):
        h, w = img.shape[:2]
        x = x_anchor_percent * w
        y = y_anchor_percent * h
        # object details
        org = [int(x), int(y)]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # draw contrast box
        x0b, y0b = (int(x - x_padding), int(y - y_padding - box_y_length))
        x1b, y1b = (int(x + box_x_length + x_padding), int(y  + y_padding))

        overlay = img.copy()
        cv2.rectangle(overlay, (x0b, y0b), (x1b, y1b), box_color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        cv2.putText(img, text, org, font, font_scale, text_color, thickness)
        return img
    

    def draw_state(self, img, data: RenderData):

        text, text_color = DrawStateMapper.map(data.draw_state)    
        img = self.draw_text_box(img, text, 0.77, 0.05, 35, 10, 0.5, 2, 15, 5, text_color)
        return img
    
    def draw_number_of_imgs(self, img, data: RenderData):
        text = f"{data.num_imgs_annotated}/{data.num_imgs_total}"
        img = self.draw_text_box(img, text, 0.8, 0.99,
                                 60, 15, x_padding=5, y_padding=5)
        img = self.draw_text_box(img, str(data.file_index + 1), 0.7, 0.99,
                                 10, 15, x_padding=5, y_padding=5)
        return img

    def draw_lasso_pixel_dist(self, img, data: RenderData):

        h, w = img.shape[:2]
        x = 0.77 * w
        y = 0.08 * h
        # object details
        org = [int(x), int(y)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 2

        # draw contrast box
        x_padding = 15
        box_x_length = 20
        box_y_length = 5
        y_padding = 5
        x0b, y0b = (int(x - x_padding), int(y - y_padding - box_y_length))
        x1b, y1b = (int(x + box_x_length + x_padding), int(y  + y_padding))
        # box_color = (10, 10, 10)
        box_color = (50, 50, 50)
        overlay = img.copy()
        alpha = 0.8
        cv2.rectangle(overlay, (x0b, y0b), (x1b, y1b), box_color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        text_color = (255, 255, 255)
        text = str(data.pixel_lasso_dist)
        
        cv2.putText(img, text, org, font, fontScale, text_color, thickness)
        return img


    def draw_guide_lines(self, img, x, y):
        h, w = img.shape[:2]
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)   # vertical
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)   # horizontal

    def bounding_box_to_image_box(self, img, bounding_boxes: List[BoundingBox], color = (214, 211, 19)):
        if bounding_boxes == [[]]: return
        
        for bb in bounding_boxes:
            x1, y1, x2, y2 = bb.box
            
            # put box in cam
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # object details
            org = [int(x1) + 2, int(y1) + 15]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
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