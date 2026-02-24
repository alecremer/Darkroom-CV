from inference_runners.model_wrapper import ModelWrapper
from inference_runners.setr_pup_wrapper import SetrPupWrapper
from inference_runners.swin_unet_wrapper import SwinUnetWrapper
from setr_pup.setr_pup_inference import SetrPupInference
from typing import List
from entities.model_types import ModelType, Model
from ultralytics import YOLO
from configs.detect_model_config import DetectModelConfig
from entities.model_tasks import Task
class ModelsLoader:

    
    

    @classmethod
    def load_models(self, model_configs: List[DetectModelConfig]) -> List[Model]:

        weight_paths = [m.weights_path for m in model_configs]
        models_loaded = []

        for config in model_configs:
        # for config_weights_paths in filter(None, weight_paths):
            weight_paths = [c for c in config.weights_path]
            for p in weight_paths:
                path = p["path"]
                model_type = p["type"]

                if path:
                    model = Model()
                    model.model_type = model_type
                    model.confidence = config.confidence
                    # model.label = config.label

                    if model_type == ModelType.YOLO.value:
                        
                        model.model = YOLO(path)
                        model.model.verbose = False
                        models_loaded.append(model)
                        model.task = model.model.task

                    elif model_type == ModelType.SETR_PUP.value:
                        encoder_path = path["backbone"]
                        head_path = path["head"]
                        model.model: ModelWrapper = SetrPupWrapper(encoder_path, head_path) # type: ignore #TODO: multiclasses support
                        model.task = Task.SEGMENTATION

                        models_loaded.append(model)

                    elif model_type == ModelType.SWIN_UNET.value:
                        model.model: ModelWrapper = SwinUnetWrapper(path) # type: ignore #TODO: multiclasses support
                        model.task = Task.SEGMENTATION

                        models_loaded.append(model)

                    else:
                        raise("Unsupported model", model_type)

        # if not models_trained:
            # raise Exception("weights paths are empty")

        return models_loaded