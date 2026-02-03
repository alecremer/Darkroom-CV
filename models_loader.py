from mae.mae_seg import MAE_Seg
from typing import List
from model_types import Model, TrainedModel
from ultralytics import YOLO

class ModelsLoader:
    def _load_vitmae_seg(self, encoder_path, head_path) -> MAE_Seg:

        model = MAE_Seg()
        model.load_vitmae_seg(encoder_path, head_path)

        return model

    def _set_trained_models(self, weight_paths) -> List[TrainedModel]:

        print(f"weights_path: {weight_paths}")
        models_trained = []

            
        for config_weights_paths in filter(None, weight_paths):
            for p in config_weights_paths:
                path = p["path"]
                model_type = p["type"]

                if path:
                    print(path)
                    if model_type == Model.YOLO.value:
                        
                        model_trained = TrainedModel()
                        model_trained.model = YOLO(path)
                        model_trained.model.verbose = False
                        model_trained.model_type = model_type
                        
                        models_trained.append(model_trained)

                    elif model_type == Model.VITMAE_SEG.value:
                        encoder_path = path["backbone"]
                        head_path = path["head"]
                        model_trained = TrainedModel()
                        model_trained.model = self._load_vitmae_seg(encoder_path, head_path)
                        model_trained.model_type = model_type

                        models_trained.append(model_trained)

                    else:
                        raise("Unsupported model", model_type)

        # if not models_trained:
            # raise Exception("weights paths are empty")

        return models_trained