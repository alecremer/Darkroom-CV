from dataclasses import dataclass
from typing import List
from annotation_app.renderer.action_dispatcher import ActionDispatcher
from annotation_app.renderer.annotation_action import AnnotationAction
from enum import Enum

@dataclass
class KeyCommand:
    key: str
    action: callable
    description: str


class KeyboardHandler:

    def __init__(self, action_dispatcher: ActionDispatcher):
        self.action_dispatcher = action_dispatcher
        self.commands = List[KeyCommand]
        self.special_keys = {"esc": 27}

    def build(self):
        cmds: List[KeyCommand] = []
        cmds.append(KeyCommand('f', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.NEXT_IMG),
        "Next image")
        )
        cmds.append(KeyCommand('s', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.PREVIOUS_IMG),
        "Previous image")
        )
        cmds.append(KeyCommand('r', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.RESET_ANNOTATION_CELL),
        "Reset image annotations")
        )
        cmds.append(KeyCommand('w', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.TOGGLE_SHOW_UI),
        "Show/Hide images")
        )
        cmds.append(KeyCommand('a', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.SAVE_ANNOTATIONS),
        "Save annotations")
        )
        cmds.append(KeyCommand('q', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.CANCEL_CONSTRUCT_MASK),
        "Cancel mask")
        )
        cmds.append(KeyCommand('d', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.DONE_OR_CREATE_MASK),
        "Draw/done mask")
        )
        cmds.append(KeyCommand('e', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.START_CONSTRUCT_RECTANGLE),
        "Draw rectangle")
        )
        cmds.append(KeyCommand('c', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.START_CONSTRUCT_MASK_LASSO),
        "Lasso mask")
        )
        cmds.append(KeyCommand('esc', 
        lambda: self.action_dispatcher.dispatch(AnnotationAction.QUIT),
        "Quit")
        )

    

        self.commands = cmds

    def _key_matches(self, key_code: str, key_name: str) -> bool:
        key_name = key_name.lower()

        if key_name in self.special_keys:
            return key_code == self.special_keys[key_name]
        
        return (key_code == ord(key_name.lower()) or key_code == ord(key_name.upper()))

    def routine(self, pressed_key: str) -> None:
        for cmd in self.commands:
            cmd: KeyCommand
            if self._key_matches(pressed_key, cmd.key):
                cmd.action()

    def key_presentation(self):
        print("--------------------------------------------")
        print("Commands")
        print("--------------------------------------------")
        for key in self.commands:
            key: KeyCommand = key
            print(f"{key.key}: {key.description}")

