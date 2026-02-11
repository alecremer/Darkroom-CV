from dataclasses import dataclass
from typing import List
from annotation_transition.renderer.action_dispatcher import ActionDispatcher
from annotation_action import AnnotationAction
@dataclass
class KeyCommand:
    key: str
    action: callable

class KeyboardHandler:

    def __init__(self, action_dispatcher: ActionDispatcher):
        self.action_dispatcher = action_dispatcher
        self.commands = List[KeyCommand]

    def build(self):
        cmds: List[KeyCommand] = []
        cmds.append(KeyCommand('g', self.action_dispatcher.dispatch(AnnotationAction.NEXT_IMG)))
        cmds.append(KeyCommand('d', self.action_dispatcher.dispatch(AnnotationAction.PREVIOUS_IMG)))
        cmds.append(KeyCommand('t', self.action_dispatcher.dispatch(AnnotationAction.RESET_ANNOTATION_CELL)))
        cmds.append(KeyCommand('e', self.action_dispatcher.dispatch(AnnotationAction.TOGGLE_SHOW_UI)))
        cmds.append(KeyCommand('s', self.action_dispatcher.dispatch(AnnotationAction.SAVE_ANNOTATIONS)))
        cmds.append(KeyCommand('w', self.action_dispatcher.dispatch(AnnotationAction.UNDO_MASK_POINT)))
        cmds.append(KeyCommand('f', self.action_dispatcher.dispatch(AnnotationAction.DONE_MASK)))
        cmds.append(KeyCommand('q', self.action_dispatcher.dispatch(AnnotationAction.QUIT)))
        cmds.append(KeyCommand('r', self.action_dispatcher.dispatch(AnnotationAction.CREATE_BOX)))

      

        self.commands = cmds

    def _key_matches(self, key_a: str, key_b: str) -> bool:
        return str.lower(key_a) == str.lower(ord(key_b))

    def routine(self, pressed_key: str) -> None:
        for cmd in self.commands:
            cmd: KeyCommand
            if self._key_matches(pressed_key, cmd.key):
                cmd.action()