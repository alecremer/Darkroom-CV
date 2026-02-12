from typing import Any, List
import cv2
from annotation_transition.renderer.keyboard_handler import KeyboardHandler
from annotation_transition.renderer.mouse_handler import InputContext, MouseHandler
from annotation_transition.renderer.render_data import RenderData
from annotation_transition.renderer.view.annotation_overlay import AnnotationOverlay
from annotation_transition.renderer.view.annotation_view import AnnotationView

class OpencvPipeline:

    def __init__(self, data: RenderData, mouse_handler: MouseHandler, keyboard_handler: KeyboardHandler, view: AnnotationView, overlay: AnnotationOverlay, labels: List[str]):
        
        self.keyboard_handler = keyboard_handler
        self.view = view
        self.overlay = overlay
        self.data = data
        self.labels = labels
        
        self.input_context = InputContext(1920)
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
        
        screen_res = 1080, 720 #TODO find a best way 
        scale_width = screen_res[0] / self.data.current_annotation.original_img.shape[1]
        scale_height = screen_res[1] / self.data.current_annotation.original_img.shape[0]
        scale = min(scale_width, scale_height, 1.0) 
        self.input_context.resize_scale = scale

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("killing...")
            #TODO: kill loop here

            
