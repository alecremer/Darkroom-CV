from enum import Enum
from dataclasses import dataclass
from typing import Any
from entities.model_tasks import Task

class ModelType(Enum):
    YOLO: str = "YOLO"
    SETR_PUP: str = "SETR-PUP"
    SWIN_UNET: str = "SWIN-UNET"

@dataclass
class Model:
    model: Any | None = None
    model_type: str | None = None
    label: str = None
    confidence: float = None
    task: Task = None
