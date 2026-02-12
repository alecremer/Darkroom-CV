from dataclasses import dataclass
from entities.model_types import ModelType, Model

@dataclass
class DetectModelConfig:

    weights_path: str
    label: str
    confidence: float
    device: str
    test_path: str | None
    segmentation: bool = False
    model: ModelType = ModelType.YOLO
