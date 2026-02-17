
import os
from pathlib import Path
from typing import List
import yaml
from configs.train_model_config import TrainModelConfig
from mae.mae_train import MAE_Train
from ultralytics import YOLO

class TrainEngine:

    def train_one_model(self, train_cfg: TrainModelConfig):

        # parse
        # epochs = train_cfg.epochs
        path = train_cfg.dataset_path
        device = train_cfg.device
        model = train_cfg.model
        results_folder_name = train_cfg.results_folder_name
        model_config = train_cfg.model_par_config

        val_path: str
        train_path: str
        test_path: str
        root_path: str
        
        # get train paths
        with open(os.path.join(path, "data.yaml"), 'r') as f:
            data = yaml.safe_load(f)

            if 'path' in data:
                root_path = Path(data['path'])
                val_path = data['val']
                train_path = data['train']
                test_path = root_path, data['test']

        # map
        if model == 'vit-mae':
            mae_train = MAE_Train()
            mae_train.train(root_path, train_path, val_path, model_config)
        
        # YOLO
        else:
            model = YOLO("models/" + model)

            if train_cfg.model_par_config:

                results = model.train(data=(path + "/data.yaml"), device=device, 
                                project=path + "/runs", name=results_folder_name, **model_config)
            # else:
            #     results = model.train(data=(path + "/data.yaml"), epochs=epochs, device=device, 
            #                     project=path + "/runs", name=results_folder_name, patience=50)
            model.val()

    def train(self, train_cfg_list: List[TrainModelConfig]):

        if train_cfg_list and len(train_cfg_list) > 0:
            
            for cfg in train_cfg_list:

                self.train_one_model(cfg)

        else:
            raise ValueError("train configuration could not be empty")