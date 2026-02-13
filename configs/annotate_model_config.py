from dataclasses import dataclass

@dataclass
class AnnotateModelConfig:

    weights_path: str
    labels: str
    confidence: float