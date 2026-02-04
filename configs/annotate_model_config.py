from dataclasses import dataclass

@dataclass
class AnnotateModelConfig:

    weights_path: str
    labels_to_annotate: str
    annotate_confidence: float