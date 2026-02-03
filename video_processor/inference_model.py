from model_types import TrainedModel
from model_tasks import Task

class InferenceModel:
    model: TrainedModel
    label: str
    confidence: float
    task: Task
