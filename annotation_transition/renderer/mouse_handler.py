import cv2
from annotation_transition.renderer.action_dispatcher import ActionDispatcher
from annotation_transition.renderer.interaction_policy import InteractionPolicy, PolicyResult
from types.entities import Point
from annotation_transition.renderer.input_intent import InputIntent
from annotation_transition.renderer.annotation_action import AnnotationAction
from annotation_engine import AnnotationEngine
from rendering.opencv_renderer_primitives import OpencvRenderPrimitives
from dataclasses import dataclass
from annotation_controller import AnnotationController
from action_handler import ActionHandler

@dataclass
class InputContext:
    resize_scale: float

class MouseHandler:

    def __init__(self, controller: AnnotationController, command_dispatcher: ActionDispatcher, policy: InteractionPolicy):

        self.command_dispatcher = command_dispatcher
        self.policy = policy
        self.controller = controller

    def mouse_callback_default(self, intent: InputIntent, x, y):
        result: PolicyResult = self.policy.decide(self.controller.draw_state, intent)

        if x and y:
            self.command_dispatcher.dispatch(result.action, Point(x, y))
        else:
            self.command_dispatcher.dispatch(result.action)

        if result.next_state:
            self.controller.draw_state = result.next_state


    def mouse_move_callback(self, x, y):
        
        self.mouse_callback_default(InputIntent.MOVE, x, y)


    def lmb_up_callback(self, x, y):

        self.mouse_callback_default(InputIntent.LMB_UP, x, y)


    def rmb_up_callback(self):

        self.mouse_callback_default(InputIntent.RMB_UP)


    def lmb_down_callback(self, x, y):

        self.command_dispatcher.dispatch(AnnotationAction.SELECT_LABEL, Point(x, y))
        self.mouse_callback_default(InputIntent.LMB_DOWN)


    def mouse_callback(self, event, x, y, flags, param: InputContext):

        self.x_y_mouse = OpencvRenderPrimitives.normalize_by_scale(x, y, param.resize_scale)

        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_move_callback(x, y)
            

        elif event == cv2.EVENT_LBUTTONUP:
            self.lmb_up_callback(x, y)
                

        if event == cv2.EVENT_RBUTTONUP:
            self.rmb_up_callback()


        if event == cv2.EVENT_LBUTTONDOWN:
            self.lmb_down_callback(x, y)