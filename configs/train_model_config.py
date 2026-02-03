from dataclasses import dataclass

@dataclass
class TrainModelConfig:

    dataset_path: str
    epochs: int
    label: str
    device: str
    model: str
    results_folder_name: str
    model_par_config: dict