from dataclasses import dataclass
from typing import List
from annotation_transition.renderer.action_dispatcher import ActionDispatcher
from annotation_transition.renderer.annotation_action import AnnotationAction
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
        cmds.append(KeyCommand('g', lambda: self.action_dispatcher.dispatch(AnnotationAction.NEXT_IMG)))
        cmds.append(KeyCommand('d', lambda: self.action_dispatcher.dispatch(AnnotationAction.PREVIOUS_IMG)))
        cmds.append(KeyCommand('t', lambda: self.action_dispatcher.dispatch(AnnotationAction.RESET_ANNOTATION_CELL)))
        cmds.append(KeyCommand('e', lambda: self.action_dispatcher.dispatch(AnnotationAction.TOGGLE_SHOW_UI)))
        cmds.append(KeyCommand('s', lambda: self.action_dispatcher.dispatch(AnnotationAction.SAVE_ANNOTATIONS)))
        cmds.append(KeyCommand('w', lambda: self.action_dispatcher.dispatch(AnnotationAction.UNDO_MASK_POINT)))
        cmds.append(KeyCommand('f', lambda: self.action_dispatcher.dispatch(AnnotationAction.DONE_MASK)))
        cmds.append(KeyCommand('q', lambda: self.action_dispatcher.dispatch(AnnotationAction.QUIT)))
        cmds.append(KeyCommand('r', lambda: self.action_dispatcher.dispatch(AnnotationAction.CREATE_BOX)))

      

        self.commands = cmds

    def _key_matches(self, key_a: str, key_b: str) -> bool:
        # if not isinstance(key_a, str) or not isinstance(key_b, str): 
            # return False
        
        return key_a == ord(str.lower(key_b)) or key_a == ord(str.upper(key_b))

    def routine(self, pressed_key: str) -> None:
        for cmd in self.commands:
            cmd: KeyCommand
            if self._key_matches(pressed_key, cmd.key):
                cmd.action()