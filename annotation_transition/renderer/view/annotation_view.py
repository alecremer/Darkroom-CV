from annotation_transition.renderer.view.button_handler import Button
from rendering.opencv_renderer_primitives import OpencvRenderPrimitives
from dataclasses import dataclass
from entities.entities import Point, Rectangle, BoundingBox
from typing import List
import cv2
import math

@dataclass
class BtnView:
    rect: Rectangle
    text: str


class AnnotationView:

    def __init__(self, btns_templates: List[Button]):
        self.show_ui: bool = False
        self.label_btns: List[BtnView] = []
        self.btns_templates = btns_templates

        labels = [label.text for label in btns_templates]
        self.build_label_btns(labels)

    def build_label_btns(self, labels: List[str]):

        for i, label in enumerate(labels):
            
            x0, y0 = (10, 10 + i*40)
            x, y = (200, 40 + i*40)

            rect =Rectangle(Point(x0, y0), Point(x, y))
            btn = BtnView(rect, label)
            
            self.label_btns.append(btn)
        
        for btn_view, btn in zip(self.label_btns, self.btns_templates):
            btn.rect = btn_view.rect

    def draw_label_buttons(self, img, current_label):

        for i, btn in enumerate(self.label_btns):
            color = (50, 50, 50)
            text_color = (0,255,0) if btn.text == current_label else (255,255,255)
            text_x0, text_y0 = (20, -8 + btn.rect.p.y)

            OpencvRenderPrimitives.draw_btn(img, btn.rect, color, text_color, text_x0, text_y0, btn.text, 0.8, 2)


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
    
    def toggle_show_ui(self):
        self.show_ui = not self.show_ui
    