from types.model_types import Model
from types.model_tasks import Task

class InferenceModel:
    model: Model
    label: str
    confidence: float
    task: Task
