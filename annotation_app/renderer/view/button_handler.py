from dataclasses import dataclass
from typing import List

from entities.entities import Point, Rectangle


@dataclass
class Button:
    text: str
    rect: Rectangle
    action: callable

class ButtonHandler:

    def __init__(self):

        self.btns: List[Button] = []

    def register_btn(self, text: str, rect: Rectangle, action: callable):
        self.btns.append(Button(text, rect, action))

    def register_btns(self, btns: List[Button]):
        for btn in btns:
            self.register_btn(btn.text, btn.rect, btn.action)

    def handle_btns(self, p: Point):
        
        x, y = p
        for btn in self.btns:
            x0, y0, x1, y1 = btn.rect.to_coords()
            if x0 <= x <= x1 and y0 <= y <= y1:
                btn.action()