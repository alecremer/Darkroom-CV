import cv2
from mouse_handler import MouseHandler, InputContext
from keyboard_handler import KeyboardHandler

class OpencvPipeline:

    def __init__(self, mouse_handler: MouseHandler, keyboard_handler: KeyboardHandler):
        window_name = "Annotation"
        self.input_context = InputContext()
        self.keyboard_handler = keyboard_handler

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_handler.mouse_callback, param=self.input_context)

        