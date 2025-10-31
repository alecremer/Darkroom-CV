from enum import Enum
from dataclasses import dataclass
from typing import Any

class Model(Enum):
    YOLO: str = "YOLO"
    VITMAE_SEG: str = "VITMAE_SEG"

@dataclass
class TrainedModel:
    model: Any | None = None
    model_type: str | None = None
