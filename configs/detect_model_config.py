from dataclasses import dataclass
from model_types import Model, TrainedModel

@dataclass
class DetectModelConfig:

    weights_path: str
    label: str
    confidence: float
    device: str
    test_path: str | None
    segmentation: bool = False
    model: Model = Model.YOLO
