from enum import Enum

class DrawState(Enum):
    IDLE = 0,
    STARTING_RECTANGLE = 10
    DRAWING_RECTANGLE = 20,
    STARTING_MASK = 30,
    DRAWING_MASK = 40,
