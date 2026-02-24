from annotation_app.renderer.action_handler import ActionHandler
from annotation_app.renderer.action_dispatcher import ActionDispatcher
from annotation_app.renderer.annotation_action import AnnotationAction
from annotation_app.renderer.command_adapter import CommandAdapter
from annotation_app.renderer.interaction_policy import InteractionPolicy
from annotation_app.renderer.view.button_handler import Button, ButtonHandler
from annotation_app.renderer.view.keyboard_handler import KeyboardHandler
from annotation_app.renderer.view.mouse_handler import MouseHandler
from annotation_app.renderer.render_data import RenderData
from annotation_app.renderer.view.annotation_overlay import AnnotationOverlay
from annotation_app.renderer.view.annotation_view import AnnotationView
from typing import List
from annotation_app.renderer.view.opencv_pipeline import OpencvPipeline

class AnnotationController:

    def __init__(self, labels: List[str], comm: callable):
        
        self.data: RenderData = RenderData()
        self.overlay = AnnotationOverlay()
        self.quit: bool = False


        self.action_handler = ActionHandler(self.data, self.set_quit)
        self.command_adapter = CommandAdapter(comm)
        self.action_dispatcher = ActionDispatcher(self.action_handler, self.command_adapter, self.data)
        
        btns = self.build_btns(labels)
        self.view = AnnotationView(btns)
        btn_handler = ButtonHandler()
        btn_handler.register_btns(btns)

        self.policy = InteractionPolicy()
        self.mouse_handler = MouseHandler(self.data, self.action_dispatcher, self.policy, btn_handler)
        self.keyboard_handler = KeyboardHandler(self.action_dispatcher)
        self.view_pipeline = OpencvPipeline(self.data, self.mouse_handler, self.keyboard_handler, self.view, self.overlay, labels)

        d = self.command_adapter.send(AnnotationAction.UPDATE)
        self.data.update_from(d)
        print(self.data.label)


    def build_btns(self, labels: List[str]):
        
        def label_btn_action(label):
            def action():
                self.action_dispatcher.dispatch(AnnotationAction.SELECT_LABEL, label)
            return action
         
        btns: List[Button] = []
        for label in labels:
            btn_label = label
            btn = Button(btn_label, None, label_btn_action(label))
            btns.append(btn)
        
        return btns

    def set_quit(self):
        self.quit = True

    def run(self):
        while not self.quit:
            self.view_pipeline.routine(self.data.img)