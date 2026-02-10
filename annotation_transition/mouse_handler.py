import cv2
from command_policy import CommandPolicy, PolicyResult
from types.entities import Point
from command_intent import CommandIntent
from annotation_action import AnnotationAction
from annotation_engine import AnnotationEngine
from rendering.opencv_renderer_primitives import OpencvRenderPrimitives
from dataclasses import dataclass

@dataclass
class InputContext:
    resize_scale: float

class MouseHandler:

    def __init__(self, engine: AnnotationEngine, policy: CommandPolicy):

        self.engine = engine
        self.policy = policy

    def mouse_callback_default(self, intent: CommandIntent, x, y):
        result: PolicyResult = self.policy.decide(self.draw_state, intent)

        if x and y:
            self.engine.execute(result.action, Point(x, y))
        else:
            self.engine.execute(result.action)

        if result.next_state:
            self.draw_state = result.next_state


    def mouse_move_callback(self, x, y):
        
        self.mouse_callback_default(CommandIntent.MOVE, x, y)


    def lmb_up_callback(self, x, y):

        self.mouse_callback_default(CommandIntent.LMB_UP, x, y)


    def rmb_up_callback(self):

        self.mouse_callback_default(CommandIntent.RMB_UP)


    def lmb_down_callback(self, x, y):

        self.engine.execute(AnnotationAction.SELECT_LABEL, Point(x, y))

        self.mouse_callback_default(CommandIntent.LMB_DOWN)


    def mouse_click_annotate_callback(self, event, x, y, flags, param: InputContext):

        self.x_y_mouse = OpencvRenderPrimitives.normalize_by_scale(x, y, param.resize_scale)

        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_move_callback(x, y)
            

        elif event == cv2.EVENT_LBUTTONUP:
            self.lmb_up_callback(x, y)
                

        if event == cv2.EVENT_RBUTTONUP:
            self.rmb_up_callback()


        if event == cv2.EVENT_LBUTTONDOWN:
            self.lmb_down_callback(x, y)