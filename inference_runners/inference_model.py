from entities.model_types import Model
from entities.model_tasks import Task

class InferenceModel:
    model: Model
    label: str
    confidence: float
    task: Task
