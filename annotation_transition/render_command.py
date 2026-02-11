from dataclasses import dataclass
from render_intent import RenderIntent
from typing import Any

@dataclass
class RenderCommand:
    intent: RenderIntent
    payload: Any