from dataclasses import dataclass
from typing import Any
from annotation_transition.render_intent import RenderIntent

@dataclass
class RenderCommand:
    intent: RenderIntent
    payload: Any