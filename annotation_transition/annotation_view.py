from rendering.opencv_renderer_primitives import OpencvRenderPrimitives
from dataclasses import dataclass
from types.entities import Point, Rectangle, BoundingBox
from typing import List
import cv2
import math

@dataclass
class BtnView:
    rect: Rectangle
    text: str


class AnnotationView:

    def __init__(self, labels):
        self.show_ui: bool = False
        self.label_btns: List[BtnView] = []

        self.build_label_btns(labels)

    def build_label_btns(self, labels: List[str]):
        for i, label in enumerate(labels):
            
            x0, y0 = (10, 10 + i*40)
            x, y = (200, 40 + i*40)

            rect =Rectangle(Point(x0, y0), Point(x, y))
            btn = BtnView(rect, label)
            
            self.label_btns.append(btn)

    def draw_label_buttons(self, img, labels, current_label):

        for i, btn in enumerate(self.label_btns):
            color = (50, 50, 50)
            text_color = (0,255,0) if btn.text == current_label else (255,255,255)
            text_x0, text_y0 = (20, 35 + i*40)

            OpencvRenderPrimitives.draw_btn(img, btn.rect, color, text_color, text_x0, text_y0, btn.text, 0.8, 2)


    def draw_construct_rectangle(self, img, rect: Rectangle):
        img_copy = img.copy()
        color = (0, 255, 0)
        thickness = 2
        OpencvRenderPrimitives.draw_rectangle(img_copy, rect, color, thickness)

    def select_label(self, p: Point) -> str:
        x, y = p
        if self.show_ui:
            for btn in self.label_btns:
                x0, y0, x1, y1 = btn.rect.to_coords()
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return btn.text

        return None
    
    def create_bounding_box_from_result(self, detection_result, frame, labels_to_annotate = None):
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
    
    #TODO move from here
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

    #TODO move from here
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

    #TODO move from here
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