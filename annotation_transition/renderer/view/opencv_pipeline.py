from typing import Any, List
import cv2
from mouse_handler import MouseHandler, InputContext
from keyboard_handler import KeyboardHandler
from annotation_view import AnnotationView
from annotation_overlay import AnnotationOverlay
from annotation_transition.renderer.render_data import RenderData

class OpencvPipeline:

    def __init__(self, data: RenderData, mouse_handler: MouseHandler, keyboard_handler: KeyboardHandler, view: AnnotationView, overlay: AnnotationOverlay, labels: List[str]):
        
        self.keyboard_handler = keyboard_handler
        self.view = view
        self.overlay = overlay
        self.data = data
        self.labels = labels
        
        self.input_context = InputContext()
        self.keyboard_handler.build()
        self.view.build_label_btns(labels)

        self.window_name = "Annotation"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_handler.mouse_callback, param=self.input_context)

    def routine(self, img: Any):

        key = cv2.waitKey(10) & 0xFF  
        self.keyboard_handler.routine(key)
        
        if self.data.show_ui:
            self.view.draw_label_buttons(img, self.labels, self.data.label)

        if self.data.construct_box:
            self.overlay.draw_construct_box(img, self.data.construct_box)

        self.overlay.render_annotation(self.data)

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("killing...")
            #TODO: kill loop here

            
