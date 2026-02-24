from dataclasses import dataclass, field
from typing import Any, List
from annotation_app.annotation_cell import AnnotationCell

@dataclass
class AnnotationData:
    annotations: List[AnnotationCell] = field(default_factory=list)
    current_annotation: AnnotationCell | None = None
    file_index: int = 0
    img: Any | None = None
    label: str = ""
    labels: List[str] = field(default_factory=list)
    num_imgs_total: int = 0
    num_imgs_annotated: int = 0