import cv2
from annotation_transition.renderer.action_dispatcher import ActionDispatcher
from annotation_transition.renderer.interaction_policy import InteractionPolicy, PolicyResult
from annotation_transition.renderer.render_data import RenderData
from annotation_transition.renderer.view.button_handler import ButtonHandler
from entities.entities import Point
from annotation_transition.renderer.input_intent import InputIntent
from annotation_transition.renderer.annotation_action import AnnotationAction
from rendering.opencv_renderer_primitives import OpencvRenderPrimitives
from dataclasses import dataclass

@dataclass
class InputContext:
    resize_scale: float

class MouseHandler:

    def __init__(self, data: RenderData, command_dispatcher: ActionDispatcher, policy: InteractionPolicy, button_handler: ButtonHandler):

        self.command_dispatcher = command_dispatcher
        self.policy = policy
        self.data = data
        self.btn_handler = button_handler


    def mouse_callback_default(self, intent: InputIntent, x = None, y = None):
        result: PolicyResult = self.policy.decide(self.data.draw_state, intent)

        if x and y:
            self.data.mouse_xy = Point(x, y)
            self.command_dispatcher.dispatch(result.action, Point(x, y))
        else:
            self.command_dispatcher.dispatch(result.action)

        if result.next_state:
            self.data.draw_state = result.next_state


    def mouse_move_callback(self, x, y):
        
        self.mouse_callback_default(InputIntent.MOVE, x, y)


    def lmb_up_callback(self, x, y):

        
        
        self.mouse_callback_default(InputIntent.LMB_UP, x, y)

    def rmb_up_callback(self):

        self.mouse_callback_default(InputIntent.RMB_UP)


    def lmb_down_callback(self, x, y):
        
        if self.data.show_ui:
            self.btn_handler.handle_btns(Point(x, y))
        
        self.mouse_callback_default(InputIntent.LMB_DOWN, x, y)

    def wheel_callback(self, intent: InputIntent, flags):
        result: PolicyResult = self.policy.decide(self.data.draw_state, intent)

        self.command_dispatcher.dispatch(result.action, flags)

        if result.next_state:
            self.data.draw_state = result.next_state


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

        if event == cv2.EVENT_MOUSEWHEEL:
            self.wheel_callback(InputIntent.WHEEL, flags)