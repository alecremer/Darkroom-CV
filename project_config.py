from dataclasses import dataclass
from typing import Any

@dataclass
class ProjectConfig:
    train: Any
    detect: Any
    annotate: Any