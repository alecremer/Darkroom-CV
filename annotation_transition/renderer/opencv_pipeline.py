import cv2
from mouse_handler import MouseHandler, InputContext
from keyboard_handler import KeyboardHandler

class OpencvPipeline:

    def __init__(self, mouse_handler: MouseHandler, keyboard_handler: KeyboardHandler):
        self.window_name = "Annotation"
        self.input_context = InputContext()
        self.keyboard_handler = keyboard_handler
        self.keyboard_handler.build()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_handler.mouse_callback, param=self.input_context)

    def routine(self):

        key = cv2.waitKey(10) & 0xFF  
        self.keyboard_handler.routine(key)
        
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("killing...")
            #TODO: kill loop here
