from typing import Any, List
import cv2
import numpy as np
from annotation_app.renderer.draw_state import DrawState
from annotation_app.renderer.view.button_handler import ButtonHandler
from annotation_app.renderer.view.keyboard_handler import KeyboardHandler
from annotation_app.renderer.view.mouse_handler import InputContext, MouseHandler
from annotation_app.renderer.render_data import RenderData
from annotation_app.renderer.view.annotation_overlay import AnnotationOverlay
from annotation_app.renderer.view.annotation_view import AnnotationView
from dataset_analysis.ceav import CEAV
from rendering.opencv_renderer_primitives import OpencvRenderPrimitives

class OpencvPipeline:

    def __init__(self, data: RenderData, mouse_handler: MouseHandler, keyboard_handler: KeyboardHandler, view: AnnotationView, overlay: AnnotationOverlay, labels: List[str]):
        
        self.keyboard_handler = keyboard_handler
        self.view = view
        self.overlay = overlay
        self.data = data
        self.labels = labels
        
        self.input_context = InputContext(1920)
        self.keyboard_handler.build()

        self.window_name = "Annotation"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_handler.mouse_callback, param=self.input_context)

    def routine(self, img: Any):

        key = cv2.waitKey(10) & 0xFF
        
        img = self.data.current_annotation.original_img.copy()
        self.keyboard_handler.routine(key)
        
        img = self.overlay.render_annotation(img, self.data)

        if self.data.draw_state is not DrawState.IDLE:
            img = self.overlay.draw_state(img, self.data)

        if self.data.draw_state == DrawState.DRAWING_MASK_LASSO:
            img = self.overlay.draw_lasso_pixel_dist(img, self.data)

        if self.data.construct_box:
            img = self.overlay.draw_construct_box(img, self.data.construct_box)

        img = self.overlay.draw_number_of_imgs(img, self.data)

        if self.data.show_ui:
            self.view.draw_label_buttons(img, self.data.label)


        OpencvRenderPrimitives.resize_and_show(img)


        screen_res = 1080, 720 #TODO find a best way 
        scale_width = screen_res[0] / self.data.current_annotation.original_img.shape[1]
        scale_height = screen_res[1] / self.data.current_annotation.original_img.shape[0]
        scale = min(scale_width, scale_height, 1.0) 
        self.input_context.resize_scale = scale

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("killing...")
            #TODO: kill loop here

            
