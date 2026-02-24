from enum import Enum

class RenderIntent(Enum):
    START_CONSTRUCT_BOX = 10,
    DRAW_CONSTRUCT_BOX = 20,
    CANCEL_CONSTRUCT_BOX = 30,
    SELECT_LABEL = 40