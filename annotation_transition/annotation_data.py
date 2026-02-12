from dataclasses import dataclass, field
from typing import Any, List
from annotation_transition.annotation_cell import AnnotationCell

@dataclass
class AnnotationData:
    annotations: List[AnnotationCell] = field(default_factory=list)
    current_annotation: AnnotationCell | None = None
    file_index: int = 0
    img: Any | None = None
    label: str = ""