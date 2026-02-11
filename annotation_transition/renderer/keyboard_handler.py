from dataclasses import dataclass
from typing import List
from annotation_transition.renderer.action_dispatcher import ActionDispatcher

@dataclass
class KeyCommand:
    key: str
    action: callable

class KeyboardHandler:

    def __init__(self, action_dispatcher: ActionDispatcher):
        self.annotation_engine = action_dispatcher
        self.commands = List[KeyCommand]

    def build(self):
        cmds: List[KeyCommand] = []
        cmds.append(KeyCommand('g', self.annotation_engine.next_img))
        cmds.append(KeyCommand('d', self.annotation_engine.previous_img))
        cmds.append(KeyCommand('t', self.annotation_engine.reset_annotation_cell))
        cmds.append(KeyCommand('e', self.annotation_engine.toggle_show_ui))
        cmds.append(KeyCommand('s', self.annotation_engine.save_annotations))
        cmds.append(KeyCommand('w', self.annotation_engine.undo_polygon_point))
        cmds.append(KeyCommand('f', self.annotation_engine.done_or_create_polygon))
        cmds.append(KeyCommand('q', self.annotation_engine.quit))
        cmds.append(KeyCommand('r', self.annotation_engine.set_create_rectangle))

        self.commands = cmds

    def _key_matches(self, key_a: str, key_b: str) -> bool:
        return str.lower(key_a) == str.lower(ord(key_b))

    def routine(self, pressed_key: str) -> None:
        for cmd in self.commands:
            cmd: KeyCommand
            if self._key_matches(pressed_key, cmd.key):
                cmd.action()