from mae.mae_seg import MAE_Seg
from typing import List
from model_types import ModelType, Model
from ultralytics import YOLO
from configs.detect_model_config import DetectModelConfig
from model_tasks import Task
class ModelsLoader:

    @classmethod
    def _load_vitmae_seg(self, encoder_path, head_path) -> MAE_Seg:

        model = MAE_Seg()
        model.load_vitmae_seg(encoder_path, head_path)

        return model

    @classmethod
    def load_models(self, model_configs: List[DetectModelConfig]) -> List[Model]:

        weight_paths = [m.weights_path for m in model_configs]
        print(f"weights_path: {weight_paths}")
        models_loaded = []

        for config in model_configs:
        # for config_weights_paths in filter(None, weight_paths):
            weight_paths = [c for c in config.weights_path]
            for p in weight_paths:
                path = p["path"]
                model_type = p["type"]

                if path:
                    print(path)
                    model = Model()
                    model.model_type = model_type
                    model.confidence = config.confidence
                    model.label = config.label

                    if model_type == ModelType.YOLO.value:
                        
                        model.model = YOLO(path)
                        model.model.verbose = False
                        models_loaded.append(model)
                        model.task = model.model.task

                    elif model_type == ModelType.VITMAE_SEG.value:
                        encoder_path = path["backbone"]
                        head_path = path["head"]
                        model.model = self._load_vitmae_seg(encoder_path, head_path)
                        model.task = Task.SEGMENTATION

                        models_loaded.append(model)

                    else:
                        raise("Unsupported model", model_type)

        # if not models_trained:
            # raise Exception("weights paths are empty")

        return models_loaded