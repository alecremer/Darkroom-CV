from enum import Enum
from dataclasses import dataclass
from typing import Any
from model_tasks import Task

class ModelType(Enum):
    YOLO: str = "YOLO"
    VITMAE_SEG: str = "VITMAE_SEG"

@dataclass
class Model:
    model: Any | None = None
    model_type: str | None = None
    label: str
    confidence: float
    task: Task
