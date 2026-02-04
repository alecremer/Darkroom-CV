from model_types import Model
from model_tasks import Task

class InferenceModel:
    model: Model
    label: str
    confidence: float
    task: Task
