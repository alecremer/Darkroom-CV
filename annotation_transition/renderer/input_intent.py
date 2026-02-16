from enum import Enum

class InputIntent(Enum):
    
    IDLE = 0,
    MOVE = 10,
    LMB_UP = 20,
    LMB_DOWN = 30,
    RMB_UP = 40,
    RMB_DOWN = 50,
    WHEEL = 60,